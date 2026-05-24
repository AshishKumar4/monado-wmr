// Copyright 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Offline controller-VIO replay: drive the REAL constellation tracker + ESKF over recorded raw
 *        frames + controller IMU, so the optical front-end and the IMU<->visual drift can be studied and
 *        tuned WITHOUT a headset. Inputs are all recorded artifacts (no synthetic data, no production
 *        code changes — the harness drives the tracker through its public API with fake devices):
 *          - euroc raw frames (mav0/cam0..3/data/, the four mosaic ROIs, timestamp-aligned)
 *          - per-unit camera calibration  (wmr/hmd-cameras.json, dumped by the driver)
 *          - per-unit controller LED model (wmr/controller_<serial>.json -> ControllerLeds)
 *          - controller IMU               (a capture's telemetry/imu.bin, manifest-driven)
 *
 * Stage A (this file so far): load + validate every input. Stage B wires the fake devices + tracker and
 * replays; Stage C emits the trajectory + IMU<->visual drift analysis.
 */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <fstream>

#include <cjson/cJSON.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "xrt/xrt_defines.h"
#include "xrt/xrt_tracking.h"
#include "tracking/t_constellation_tracking.h"
#include "tracking/t_led_models.h"

namespace {

namespace fs = std::filesystem;

// ---- small helpers ---------------------------------------------------------

std::string
read_file(const std::string &path)
{
	std::ifstream f(path, std::ios::binary);
	if (!f) {
		return {};
	}
	return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

enum t_camera_distortion_model
distortion_model_from_string(const char *s)
{
	if (strcmp(s, "T_DISTORTION_WMR") == 0) return T_DISTORTION_WMR;
	if (strcmp(s, "T_DISTORTION_OPENCV_RADTAN_5") == 0) return T_DISTORTION_OPENCV_RADTAN_5;
	if (strcmp(s, "T_DISTORTION_OPENCV_RADTAN_8") == 0) return T_DISTORTION_OPENCV_RADTAN_8;
	if (strcmp(s, "T_DISTORTION_OPENCV_RADTAN_14") == 0) return T_DISTORTION_OPENCV_RADTAN_14;
	if (strcmp(s, "T_DISTORTION_FISHEYE_KB4") == 0) return T_DISTORTION_FISHEYE_KB4;
	return T_DISTORTION_WMR;
}

double
jnum(const cJSON *o, const char *k)
{
	const cJSON *v = cJSON_GetObjectItemCaseSensitive(o, k);
	return cJSON_IsNumber(v) ? v->valuedouble : 0.0;
}

// ---- camera group (wmr/hmd-cameras.json, the format dumped by the driver) ----

bool
load_camera_group(const std::string &path, struct t_constellation_camera_group *out)
{
	std::string txt = read_file(path);
	cJSON *root = cJSON_Parse(txt.c_str());
	if (root == nullptr) {
		fprintf(stderr, "camera group: parse failed: %s\n", path.c_str());
		return false;
	}
	bool ok = true;
	const cJSON *cams = cJSON_GetObjectItemCaseSensitive(root, "cameras");
	out->cam_count = cJSON_GetArraySize(cams);
	int i = 0;
	const cJSON *c = nullptr;
	cJSON_ArrayForEach(c, cams)
	{
		if (i >= XRT_TRACKING_MAX_SLAM_CAMS) {
			break;
		}
		struct t_constellation_camera *cam = &out->cams[i];
		struct t_camera_calibration *cal = &cam->calibration;
		cal->image_size_pixels.w = (int)jnum(c, "width");
		cal->image_size_pixels.h = (int)jnum(c, "height");
		const cJSON *K = cJSON_GetObjectItemCaseSensitive(c, "intrinsics");
		for (int r = 0; r < 3; r++) {
			const cJSON *row = cJSON_GetArrayItem(K, r);
			for (int col = 0; col < 3; col++) {
				cal->intrinsics[r][col] = cJSON_GetArrayItem(row, col)->valuedouble;
			}
		}
		cal->distortion_model = distortion_model_from_string(
		    cJSON_GetObjectItemCaseSensitive(c, "distortion_model")->valuestring);
		const cJSON *dist = cJSON_GetObjectItemCaseSensitive(c, "distortion");
		for (int k = 0; k < XRT_DISTORTION_MAX_DIM && k < cJSON_GetArraySize(dist); k++) {
			cal->distortion_parameters_as_array[k] = cJSON_GetArrayItem(dist, k)->valuedouble;
		}
		const cJSON *P = cJSON_GetObjectItemCaseSensitive(c, "P_imu_cam");
		const cJSON *pos = cJSON_GetObjectItemCaseSensitive(P, "position");
		const cJSON *ori = cJSON_GetObjectItemCaseSensitive(P, "orientation");
		cam->P_imu_cam.position = {(float)cJSON_GetArrayItem(pos, 0)->valuedouble,
		                          (float)cJSON_GetArrayItem(pos, 1)->valuedouble,
		                          (float)cJSON_GetArrayItem(pos, 2)->valuedouble};
		cam->P_imu_cam.orientation = {(float)cJSON_GetArrayItem(ori, 0)->valuedouble,
		                             (float)cJSON_GetArrayItem(ori, 1)->valuedouble,
		                             (float)cJSON_GetArrayItem(ori, 2)->valuedouble,
		                             (float)cJSON_GetArrayItem(ori, 3)->valuedouble};
		const cJSON *roi = cJSON_GetObjectItemCaseSensitive(c, "roi");
		cam->roi.offset.w = (int)jnum(roi, "x");
		cam->roi.offset.h = (int)jnum(roi, "y");
		cam->roi.extent.w = (int)jnum(roi, "w");
		cam->roi.extent.h = (int)jnum(roi, "h");
		cam->blob_min_threshold = (uint8_t)jnum(c, "blob_min_threshold");
		cam->blob_detect_threshold = (uint8_t)jnum(c, "blob_detect_threshold");
		cam->min_threshold = (uint8_t)jnum(c, "min_threshold");
		cam->slam_tracking_index = (size_t)jnum(c, "slam_tracking_index");
		i++;
	}
	cJSON_Delete(root);
	return ok && out->cam_count > 0;
}

// ---- controller LED model (wmr/controller_<serial>.json -> ControllerLeds) ----
// Mirrors wmr_controller_base_get_led_model: P_device_model = identity, LED pos/normal from the factory
// blob, radius 3 mm. (Bounding points are only used for SLAM masks, not the pose solve; left empty.)

bool
load_led_model(const std::string &path, uint8_t device_id, struct t_constellation_led_model *out)
{
	std::string txt = read_file(path);
	cJSON *root = cJSON_Parse(txt.c_str());
	if (root == nullptr) {
		fprintf(stderr, "led model: parse failed: %s\n", path.c_str());
		return false;
	}
	const cJSON *ci = cJSON_GetObjectItemCaseSensitive(root, "CalibrationInformation");
	const cJSON *leds = cJSON_GetObjectItemCaseSensitive(ci, "ControllerLeds");
	int n = cJSON_GetArraySize(leds);
	if (n <= 0) {
		cJSON_Delete(root);
		return false;
	}
	t_constellation_led_model_init(device_id, nullptr, out, (uint8_t)n, 0);
	for (int i = 0; i < n; i++) {
		const cJSON *led = cJSON_GetArrayItem(leds, i);
		const cJSON *p = cJSON_GetObjectItemCaseSensitive(led, "Position");
		const cJSON *nm = cJSON_GetObjectItemCaseSensitive(led, "Normal");
		out->leds[i].id = (uint8_t)i;
		out->leds[i].pos = {(float)cJSON_GetArrayItem(p, 0)->valuedouble,
		                    (float)cJSON_GetArrayItem(p, 1)->valuedouble,
		                    (float)cJSON_GetArrayItem(p, 2)->valuedouble};
		out->leds[i].dir = {(float)cJSON_GetArrayItem(nm, 0)->valuedouble,
		                    (float)cJSON_GetArrayItem(nm, 1)->valuedouble,
		                    (float)cJSON_GetArrayItem(nm, 2)->valuedouble};
		out->leds[i].radius_mm = 3.0f;
	}
	cJSON_Delete(root);
	return true;
}

// ---- euroc frame index (the four cams share a timestamp = one mosaic frame) ----

struct MosaicFrame
{
	int64_t t_ns;
	std::string cam_png[XRT_TRACKING_MAX_SLAM_CAMS];
};

std::vector<MosaicFrame>
index_euroc(const std::string &mav0, int cam_count)
{
	std::vector<MosaicFrame> frames;
	std::ifstream csv(mav0 + "/cam0/data.csv");
	std::string line;
	std::getline(csv, line); // header
	while (std::getline(csv, line)) {
		size_t comma = line.find(',');
		if (comma == std::string::npos) {
			continue;
		}
		MosaicFrame mf{};
		mf.t_ns = std::stoll(line.substr(0, comma));
		std::string fname = line.substr(comma + 1);
		fname.erase(std::remove_if(fname.begin(), fname.end(), [](char c) { return c == '\r' || c == '\n' || c == ' '; }),
		            fname.end());
		bool all = true;
		for (int c = 0; c < cam_count; c++) {
			mf.cam_png[c] = mav0 + "/cam" + std::to_string(c) + "/data/" + fname;
			if (!fs::exists(mf.cam_png[c])) {
				all = false;
			}
		}
		if (all) {
			frames.push_back(mf);
		}
	}
	return frames;
}

// ---- controller IMU (telemetry/imu.bin, manifest-driven offsets) ----

struct ImuRow
{
	int64_t t_ns;
	float ax, ay, az, gx, gy, gz;
};

std::vector<ImuRow>
load_imu(const std::string &telemetry_dir, int device_id)
{
	std::vector<ImuRow> out;
	std::string mtxt = read_file(telemetry_dir + "/manifest.json");
	cJSON *root = cJSON_Parse(mtxt.c_str());
	if (root == nullptr) {
		return out;
	}
	const cJSON *imu = cJSON_GetObjectItemCaseSensitive(cJSON_GetObjectItemCaseSensitive(root, "streams"), "imu");
	int row_size = (int)jnum(imu, "row_size");
	const cJSON *fields = cJSON_GetObjectItemCaseSensitive(imu, "fields");
	int off_t = -1, off_dev = -1, off_ax = -1, off_gx = -1; // manifest-driven, no hardcoded layout
	const cJSON *fld = nullptr;
	cJSON_ArrayForEach(fld, fields)
	{
		const char *nm = cJSON_GetObjectItemCaseSensitive(fld, "name")->valuestring;
		int off = (int)jnum(fld, "offset");
		if (strcmp(nm, "t_mono_ns") == 0) off_t = off;
		else if (strcmp(nm, "device_id") == 0) off_dev = off;
		else if (strcmp(nm, "ax") == 0) off_ax = off;
		else if (strcmp(nm, "gx") == 0) off_gx = off;
	}
	cJSON_Delete(root);
	if (row_size <= 0 || off_t < 0 || off_dev < 0 || off_ax < 0 || off_gx < 0) {
		return out;
	}
	std::string bin = read_file(telemetry_dir + "/imu.bin");
	for (size_t p = 0; p + row_size <= bin.size(); p += row_size) {
		const char *r = bin.data() + p;
		if ((uint8_t)r[off_dev] != device_id) {
			continue;
		}
		ImuRow row{};
		memcpy(&row.t_ns, r + off_t, 8);
		memcpy(&row.ax, r + off_ax, 4);
		memcpy(&row.ay, r + off_ax + 4, 4);
		memcpy(&row.az, r + off_ax + 8, 4);
		memcpy(&row.gx, r + off_gx, 4);
		memcpy(&row.gy, r + off_gx + 4, 4);
		memcpy(&row.gz, r + off_gx + 8, 4);
		out.push_back(row);
	}
	return out;
}

} // namespace

int
main(int argc, char **argv)
{
	if (argc < 6) {
		fprintf(stderr,
		        "usage: %s <euroc mav0 dir> <hmd-cameras.json> <controller json> <telemetry dir> "
		        "<device_id 1=left 2=right>\n",
		        argv[0]);
		return 2;
	}
	const std::string mav0 = argv[1], cam_json = argv[2], ctrl_json = argv[3], telem = argv[4];
	const int device_id = atoi(argv[5]);

	struct t_constellation_camera_group cams = {};
	if (!load_camera_group(cam_json, &cams)) {
		fprintf(stderr, "FAILED to load camera group\n");
		return 1;
	}
	printf("camera group: %d cams\n", cams.cam_count);
	for (int i = 0; i < cams.cam_count; i++) {
		const struct t_camera_calibration *cal = &cams.cams[i].calibration;
		printf("  cam%d %dx%d fx=%.1f fy=%.1f roi=(%d,%d %dx%d)\n", i, cal->image_size_pixels.w,
		       cal->image_size_pixels.h, cal->intrinsics[0][0], cal->intrinsics[1][1],
		       cams.cams[i].roi.offset.w, cams.cams[i].roi.offset.h, cams.cams[i].roi.extent.w,
		       cams.cams[i].roi.extent.h);
	}

	struct t_constellation_led_model led = {};
	if (!load_led_model(ctrl_json, (uint8_t)device_id, &led)) {
		fprintf(stderr, "FAILED to load LED model\n");
		return 1;
	}
	printf("LED model: %u LEDs (e.g. led0 pos=%.4f,%.4f,%.4f)\n", led.num_leds, led.leds[0].pos.x,
	       led.leds[0].pos.y, led.leds[0].pos.z);

	std::vector<MosaicFrame> frames = index_euroc(mav0, cams.cam_count);
	printf("euroc frames: %zu mosaic frames", frames.size());
	if (!frames.empty()) {
		double dur = (frames.back().t_ns - frames.front().t_ns) / 1e9;
		printf(" over %.1fs (%.1f Hz)", dur, dur > 0 ? frames.size() / dur : 0.0);
		cv::Mat im = cv::imread(frames.front().cam_png[0], cv::IMREAD_GRAYSCALE);
		printf("; cam0 frame0 = %dx%d", im.cols, im.rows);
	}
	printf("\n");

	std::vector<ImuRow> imu = load_imu(telem, device_id);
	printf("controller IMU (device %d): %zu samples", device_id, imu.size());
	if (!imu.empty()) {
		double dur = (imu.back().t_ns - imu.front().t_ns) / 1e9;
		printf(" over %.1fs (%.0f Hz)", dur, dur > 0 ? imu.size() / dur : 0.0);
	}
	printf("\n");

	t_constellation_led_model_clear(&led);
	printf("Stage A: all inputs loaded + validated.\n");
	return 0;
}
