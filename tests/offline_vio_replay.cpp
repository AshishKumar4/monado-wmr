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

#include <chrono>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>

#include <cjson/cJSON.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "xrt/xrt_defines.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_frame.h"
#include "xrt/xrt_tracking.h"
#include "math/m_api.h"
#include "math/m_imu_3dof.h"
#include "util/u_frame.h"
#include "util/u_g2_telemetry.h"
#include "tracking/t_constellation_tracking.h"
#include "tracking/t_led_models.h"
#include "tracking/t_tracker_kalman_fusion_c.h"

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

struct FrameRow
{
	int64_t t_ns;
	int64_t hw_ts_ns;
	uint8_t cam_id;
	uint32_t frame_seq;
	uint16_t n_blobs;
	uint16_t exposure;
};

std::vector<FrameRow>
load_frame_telemetry(const std::string &telemetry_dir)
{
	std::vector<FrameRow> out;
	std::string mtxt = read_file(telemetry_dir + "/manifest.json");
	cJSON *root = cJSON_Parse(mtxt.c_str());
	if (root == nullptr) {
		return out;
	}
	const cJSON *frame = cJSON_GetObjectItemCaseSensitive(cJSON_GetObjectItemCaseSensitive(root, "streams"), "frame");
	int row_size = (int)jnum(frame, "row_size");
	const cJSON *fields = cJSON_GetObjectItemCaseSensitive(frame, "fields");
	int off_t = -1, off_hw = -1, off_cam = -1, off_seq = -1, off_n = -1, off_exposure = -1;
	const cJSON *fld = nullptr;
	cJSON_ArrayForEach(fld, fields)
	{
		const char *nm = cJSON_GetObjectItemCaseSensitive(fld, "name")->valuestring;
		int off = (int)jnum(fld, "offset");
		if (strcmp(nm, "t_mono_ns") == 0) off_t = off;
		else if (strcmp(nm, "hw_ts_ns") == 0) off_hw = off;
		else if (strcmp(nm, "cam_id") == 0) off_cam = off;
		else if (strcmp(nm, "frame_seq") == 0) off_seq = off;
		else if (strcmp(nm, "n_blobs") == 0) off_n = off;
		else if (strcmp(nm, "exposure") == 0) off_exposure = off;
	}
	cJSON_Delete(root);
	if (row_size <= 0 || off_t < 0 || off_hw < 0 || off_cam < 0 || off_seq < 0 || off_n < 0 ||
	    off_exposure < 0) {
		return out;
	}
	std::string bin = read_file(telemetry_dir + "/frame.bin");
	for (size_t p = 0; p + row_size <= bin.size(); p += row_size) {
		const char *r = bin.data() + p;
		FrameRow row{};
		memcpy(&row.t_ns, r + off_t, 8);
		memcpy(&row.hw_ts_ns, r + off_hw, 8);
		row.cam_id = (uint8_t)r[off_cam];
		memcpy(&row.frame_seq, r + off_seq, 4);
		memcpy(&row.n_blobs, r + off_n, 2);
		memcpy(&row.exposure, r + off_exposure, 2);
		out.push_back(row);
	}
	return out;
}

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

// ---- recorded head pose (gt) for the fake HMD: places the cameras in the gravity-aligned world ----

struct HeadPose
{
	int64_t t_ns;
	struct xrt_pose pose;
};

// Recorded live SLAM head pose (telemetry head_pose.bin: packed rows u64 t_mono_ns + f32 px,py,pz,qx,qy,qz,qw
// = 36 bytes, OpenXR world). This is the TRUE camera->world the constellation used live; using it makes the
// replay faithful (vs the IMU-only reconstruction, which drifts and breaks prior-dependent paths). Empty if
// the capture predates head-pose recording.
std::vector<HeadPose>
load_head_pose_telemetry(const std::string &telem)
{
	std::vector<HeadPose> v;
	std::ifstream f(telem + "/head_pose.bin", std::ios::binary);
	char buf[36];
	while (f.read(buf, sizeof(buf))) {
		uint64_t t = 0;
		float p[7];
		memcpy(&t, buf, 8);
		memcpy(p, buf + 8, 28);
		HeadPose h{};
		h.t_ns = (int64_t)t;
		h.pose.position = {p[0], p[1], p[2]};
		h.pose.orientation = {p[3], p[4], p[5], p[6]};
		v.push_back(h);
	}
	return v;
}

std::vector<HeadPose>
load_gt(const std::string &mav0)
{
	std::vector<HeadPose> v;
	std::ifstream csv(mav0 + "/gt/data.csv");
	std::string line;
	std::getline(csv, line); // header: t, px,py,pz, qw,qx,qy,qz
	while (std::getline(csv, line)) {
		long long t = 0;
		double px, py, pz, qw, qx, qy, qz;
		if (sscanf(line.c_str(), "%lld,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &t, &px, &py, &pz, &qw, &qx, &qy,
		           &qz) == 8) {
			HeadPose h{};
			h.t_ns = t;
			h.pose.position = {(float)px, (float)py, (float)pz};
			h.pose.orientation = {(float)qx, (float)qy, (float)qz, (float)qw}; // xrt order x,y,z,w
			v.push_back(h);
		}
	}
	return v;
}

// euroc head IMU (mav0/imu0/data.csv: t, wx,wy,wz, ax,ay,az)
std::vector<ImuRow>
load_euroc_imu(const std::string &mav0)
{
	std::vector<ImuRow> out;
	std::ifstream csv(mav0 + "/imu0/data.csv");
	std::string line;
	std::getline(csv, line);
	while (std::getline(csv, line)) {
		long long t = 0;
		double wx, wy, wz, ax, ay, az;
		if (sscanf(line.c_str(), "%lld,%lf,%lf,%lf,%lf,%lf,%lf", &t, &wx, &wy, &wz, &ax, &ay, &az) == 7) {
			ImuRow r{};
			r.t_ns = t;
			r.ax = (float)ax;
			r.ay = (float)ay;
			r.az = (float)az;
			out.push_back(r);
		}
	}
	return out;
}

// Align the gt world to OpenXR Y-up. The euroc gt world is gravity-aligned but in a different
// convention than the live get_tracked_pose supplied, so the constellation's gravity prior (which gates
// the ab-initio search) is wrong. Derive the constant correction R_fix that maps the gt world's gravity-
// up to +Y from the head IMU's measured gravity (frame-independent), and apply it to every head pose.
void
apply_gravity_fix(std::vector<HeadPose> &gt, const std::vector<ImuRow> &himu)
{
	if (gt.empty() || himu.empty()) {
		return;
	}
	std::vector<float> vx, vy, vz;
	size_t gi = 0;
	for (const ImuRow &s : himu) {
		while (gi + 1 < gt.size() &&
		       llabs(gt[gi + 1].t_ns - s.t_ns) <= llabs(gt[gi].t_ns - s.t_ns)) {
			gi++;
		}
		struct xrt_vec3 a = {s.ax, s.ay, s.az}; // specific force ~ world-up in the head-IMU frame
		math_vec3_normalize(&a);
		struct xrt_vec3 up_world;
		math_quat_rotate_vec3(&gt[gi].pose.orientation, &a, &up_world); // -> up in the gt world frame
		vx.push_back(up_world.x);
		vy.push_back(up_world.y);
		vz.push_back(up_world.z);
	}
	auto med = [](std::vector<float> &v) {
		std::sort(v.begin(), v.end());
		return v[v.size() / 2];
	};
	struct xrt_vec3 up = {med(vx), med(vy), med(vz)}; // median is robust to motion accel
	math_vec3_normalize(&up);
	struct xrt_quat Rfix;
	struct xrt_vec3 plusY = {0.0f, 1.0f, 0.0f};
	math_quat_from_vec_a_to_vec_b(&up, &plusY, &Rfix);
	struct xrt_pose fix = {Rfix, {0.0f, 0.0f, 0.0f}};
	for (HeadPose &h : gt) {
		struct xrt_pose corrected;
		math_pose_transform(&fix, &h.pose, &corrected);
		h.pose = corrected;
	}
	printf("gravity fix: gt-world up was (%.3f,%.3f,%.3f) -> rotated to +Y across %zu head poses\n", up.x,
	       up.y, up.z, gt.size());
}

// ---- G2_DUMP_FRAMES reader: per-cam controller PGMs (cam<id>_t<ts>_e<exp>_s<seq>_n<blobs>.pgm) ----
// Group the 4 cams of one source frame by seq; the harness reassembles them into the mosaic by ROI.
struct PgmFrame
{
	long long t_ns;
	int exposure;
	int n_blobs;
	std::string path;
};

std::vector<MosaicFrame>
index_pgm_dump(const std::string &dir, int cam_count, const std::string &telemetry_dir, bool *ok)
{
	*ok = false;
	auto field = [](const std::string &n, const char *tag) -> long long {
		size_t p = n.find(tag);
		return p == std::string::npos ? -1 : atoll(n.c_str() + p + strlen(tag));
	};
	std::map<std::pair<long long, int>, std::vector<PgmFrame>> by_key;
	std::map<long long, int> seq_counts;
	std::error_code ec;
	for (auto it = fs::directory_iterator(dir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
		const std::string name = it->path().filename().string();
		if (name.rfind("cam", 0) != 0 || it->path().extension() != ".pgm") {
			continue;
		}
		const int cam = name[3] - '0';
		const long long ts = field(name, "_t"), seq = field(name, "_s"), exposure = field(name, "_e"),
		                n_blobs = field(name, "_n");
		if (cam < 0 || cam >= cam_count || ts < 0 || seq < 0 || exposure < 0 || n_blobs < 0) {
			continue;
		}
		by_key[{seq, cam}].push_back(PgmFrame{ts, (int)exposure, (int)n_blobs, it->path().string()});
		seq_counts[seq]++;
	}

	size_t duplicate_keys = 0;
	for (const auto &kv : by_key) {
		if (kv.second.size() != 1) {
			if (duplicate_keys < 8) {
				fprintf(stderr, "PGM dump invalid: duplicate seq/cam seq=%lld cam=%d has %zu files\n",
				        kv.first.first, kv.first.second, kv.second.size());
				for (const PgmFrame &pgm : kv.second) {
					fprintf(stderr, "  %s t=%lld n=%d\n", pgm.path.c_str(), pgm.t_ns, pgm.n_blobs);
				}
			}
			duplicate_keys++;
		}
	}
	if (duplicate_keys != 0) {
		fprintf(stderr, "PGM dump invalid: %zu duplicate (seq,cam) keys in %s\n", duplicate_keys, dir.c_str());
		return {};
	}

	size_t incomplete_groups = 0;
	for (const auto &kv : seq_counts) {
		if (kv.second != cam_count) {
			if (incomplete_groups < 8) {
				fprintf(stderr, "PGM dump invalid: seq=%lld has %d camera files, expected %d\n", kv.first,
				        kv.second, cam_count);
			}
			incomplete_groups++;
		}
	}
	if (incomplete_groups != 0) {
		fprintf(stderr, "PGM dump invalid: %zu non-%d-camera source-frame groups in %s\n", incomplete_groups,
		        cam_count, dir.c_str());
		return {};
	}

	std::map<std::pair<long long, int>, FrameRow> telemetry_by_key;
	std::map<long long, int> telemetry_seq_counts;
	for (const FrameRow &row : load_frame_telemetry(telemetry_dir)) {
		if (row.cam_id >= cam_count) {
			continue;
		}
		const auto key = std::make_pair((long long)row.frame_seq, (int)row.cam_id);
		if (telemetry_by_key.find(key) != telemetry_by_key.end()) {
			fprintf(stderr, "frame telemetry invalid: duplicate seq/cam seq=%u cam=%u\n", row.frame_seq,
			        row.cam_id);
			return {};
		}
		telemetry_by_key[key] = row;
		telemetry_seq_counts[row.frame_seq]++;
	}
	if (!telemetry_by_key.empty()) {
		size_t mismatch_count = 0;
		for (const auto &kv : telemetry_seq_counts) {
			if (kv.second != cam_count) {
				if (mismatch_count < 8) {
					fprintf(stderr, "frame telemetry invalid: seq=%lld has %d camera rows, expected %d\n",
					        kv.first, kv.second, cam_count);
				}
				mismatch_count++;
			}
		}
		for (const auto &kv : by_key) {
			const auto it = telemetry_by_key.find(kv.first);
			if (it == telemetry_by_key.end()) {
				if (mismatch_count < 8) {
					fprintf(stderr, "PGM dump invalid: seq=%lld cam=%d is not present in frame.bin\n",
					        kv.first.first, kv.first.second);
				}
				mismatch_count++;
				continue;
			}
			const PgmFrame &pgm = kv.second.front();
			const FrameRow &row = it->second;
			if (pgm.t_ns != row.hw_ts_ns || pgm.n_blobs != row.n_blobs || pgm.exposure != row.exposure) {
				if (mismatch_count < 8) {
					fprintf(stderr,
					        "PGM/frame.bin mismatch: seq=%lld cam=%d pgm(t=%lld,n=%d,e=%d) "
					        "frame(t=%lld,n=%u,e=%u)\n",
					        kv.first.first, kv.first.second, pgm.t_ns, pgm.n_blobs, pgm.exposure,
					        (long long)row.hw_ts_ns, row.n_blobs, row.exposure);
				}
				mismatch_count++;
			}
		}
		for (const auto &kv : telemetry_by_key) {
			if (by_key.find(kv.first) == by_key.end()) {
				if (mismatch_count < 8) {
					fprintf(stderr, "PGM dump invalid: frame.bin seq=%lld cam=%d has no matching PGM\n",
					        kv.first.first, kv.first.second);
				}
				mismatch_count++;
			}
		}
		if (mismatch_count != 0) {
			fprintf(stderr, "PGM dump invalid: %zu mismatch(es) against telemetry frame.bin in %s\n",
			        mismatch_count, dir.c_str());
			return {};
		}
	}

	std::vector<MosaicFrame> out;
	for (const auto &kv : seq_counts) {
		const long long seq = kv.first;
		MosaicFrame mf{};
		bool have_all = true;
		long long t_ns = -1;
		for (int c = 0; c < cam_count; c++) {
			const auto it = by_key.find({seq, c});
			if (it == by_key.end()) {
				have_all = false;
				break;
			}
			const PgmFrame &pgm = it->second.front();
			if (t_ns < 0) {
				t_ns = pgm.t_ns;
			} else if (t_ns != pgm.t_ns) {
				fprintf(stderr, "PGM dump invalid: seq=%lld mixes timestamps across cameras\n", seq);
				return {};
			}
			mf.cam_png[c] = pgm.path;
		}
		if (have_all) {
			mf.t_ns = t_ns;
			out.push_back(mf);
		}
	}
	std::sort(out.begin(), out.end(), [](const MosaicFrame &a, const MosaicFrame &b) { return a.t_ns < b.t_ns; });
	*ok = true;
	return out;
}

// Head ORIENTATION from the recorded HMD IMU (telemetry device 0) via Monado's 3DOF filter: gyro
// integration (gives yaw) + gravity anchoring (roll/pitch). Far better than a gravity-only up-align,
// which has no yaw and leaves the camera world misoriented -> systematically wrong matcher prior (it
// drove the harness's spurious ~76-deg orientation error vs live's ~3 deg). Position is left 0 (no SLAM
// head trajectory; the matcher works in the camera frame, and the ESKF tracks the controller
// consistently in this gravity-aligned, slowly-yaw-drifting world).
std::vector<HeadPose>
head_poses_from_imu(const std::vector<ImuRow> &hmd_imu)
{
	std::vector<HeadPose> out;
	struct m_imu_3dof dof;
	m_imu_3dof_init(&dof, M_IMU_3DOF_USE_GRAVITY_DUR_300MS);
	for (const ImuRow &s : hmd_imu) {
		const struct xrt_vec3 a = {s.ax, s.ay, s.az};
		const struct xrt_vec3 g = {s.gx, s.gy, s.gz};
		m_imu_3dof_update(&dof, (uint64_t)s.t_ns, &a, &g);
		HeadPose h{};
		h.t_ns = s.t_ns;
		h.pose.position = {0.0f, 0.0f, 0.0f};
		h.pose.orientation = dof.rot;
		out.push_back(h);
	}
	m_imu_3dof_close(&dof);
	return out;
}

// ---- fake xrt_device shared no-ops ----
xrt_result_t
dev_noop_update(struct xrt_device *)
{
	return XRT_SUCCESS;
}
void
dev_noop_destroy(struct xrt_device *)
{
}

// ---- fake HMD: returns the recorded head pose at the queried time (nearest sample) ----
struct FakeHmd
{
	struct xrt_device base;
	std::vector<HeadPose> gt;
};

xrt_result_t
hmd_get_pose(struct xrt_device *xdev, enum xrt_input_name, int64_t t, struct xrt_space_relation *rel)
{
	FakeHmd *h = reinterpret_cast<FakeHmd *>(xdev);
	struct xrt_pose pose = XRT_POSE_IDENTITY;
	if (!h->gt.empty()) {
		size_t best = 0;
		int64_t bd = INT64_MAX;
		for (size_t i = 0; i < h->gt.size(); i++) {
			int64_t d = h->gt[i].t_ns > t ? h->gt[i].t_ns - t : t - h->gt[i].t_ns;
			if (d < bd) {
				bd = d;
				best = i;
			}
		}
		pose = h->gt[best].pose;
	}
	rel->pose = pose;
	rel->relation_flags = (enum xrt_space_relation_flags)(
	    XRT_SPACE_RELATION_ORIENTATION_VALID_BIT | XRT_SPACE_RELATION_POSITION_VALID_BIT |
	    XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT | XRT_SPACE_RELATION_POSITION_TRACKED_BIT);
	return XRT_SUCCESS;
}

// ---- fake controller: ESKF prior via get_tracked_pose + the constellation feed callbacks ----
struct FakeController
{
	struct xrt_device base;
	struct KalmanFusionInterfaceWrapper *kf;
	uint8_t device_id; // 1=left, 2=right
	std::vector<struct t_constellation_led> leds;
	std::mutex cap_lock;
	bool opt_valid = false;
	struct xrt_pose opt_pose {};
	struct FakeHmd *hmd = nullptr; // body-lock reference (mirrors the production wcb->hmd_xdev)
};

// Live head pose at time @p t (same world frame as the controller) — the body-lock reference + arm-reach
// gate origin, exactly as the production driver queries the HMD's tracked pose.
static const struct xrt_pose *
ctrl_head_pose(FakeController *c, int64_t t, struct xrt_pose *out)
{
	if (c->hmd == nullptr) {
		return nullptr;
	}
	struct xrt_space_relation rel = {};
	hmd_get_pose(reinterpret_cast<struct xrt_device *>(c->hmd), XRT_INPUT_GENERIC_TRACKER_POSE, t, &rel);
	*out = rel.pose;
	return out;
}

xrt_result_t
ctrl_get_pose(struct xrt_device *xdev, enum xrt_input_name, int64_t t, struct xrt_space_relation *rel)
{
	FakeController *c = reinterpret_cast<FakeController *>(xdev);
	struct xrt_pose hp;
	kalman_fusion_get_prediction(c->kf, t, rel, ctrl_head_pose(c, t, &hp)); // IMU prior the matcher refines
	return XRT_SUCCESS;
}

bool
cb_get_led_model(struct xrt_device *xdev, struct t_constellation_led_model *led_model)
{
	FakeController *c = reinterpret_cast<FakeController *>(xdev);
	t_constellation_led_model_init(c->device_id, nullptr, led_model, (uint8_t)c->leds.size(), 0);
	for (size_t i = 0; i < c->leds.size(); i++) {
		led_model->leds[i] = c->leds[i];
	}
	return true;
}
void
cb_push_pose(struct xrt_device *xdev, timepoint_ns t, const struct xrt_pose *pose)
{
	FakeController *c = reinterpret_cast<FakeController *>(xdev);
	struct xrt_pose_sample s = {};
	s.pose = *pose;
	s.timestamp_ns = t;
	struct xrt_pose hp;
	kalman_fusion_process_pose(c->kf, &s, nullptr, nullptr, 15, ctrl_head_pose(c, t, &hp));
	std::lock_guard<std::mutex> lk(c->cap_lock);
	c->opt_valid = true;
	c->opt_pose = *pose;
}
void
cb_push_leds(struct xrt_device *xdev, timepoint_ns t, const struct xrt_pose *P_xrworld_cam,
             const struct t_constellation_cam_calib *cc, const struct t_constellation_led_obs *leds, size_t n)
{
	FakeController *c = reinterpret_cast<FakeController *>(xdev);
	struct kalman_led_camera_view view = {cc->fx,
	                                      cc->fy,
	                                      cc->cx,
	                                      cc->cy,
	                                      P_xrworld_cam->orientation,
	                                      P_xrworld_cam->position};
	struct kalman_led_observation obs[64];
	size_t m = n < 64 ? n : 64;
	for (size_t i = 0; i < m; i++) {
		obs[i].observed_px = leds[i].obs_px;
		obs[i].led_obj = leds[i].led_obj;
		obs[i].pos_var_px2 = leds[i].pos_var_px2;
	}
	// NULL variance => uniform fallback; per-blob variance is carried in obs when available.
	struct xrt_pose hp;
	kalman_fusion_process_led_observations(c->kf, t, obs, m, &view, nullptr, 8.0f, true,
	                                       ctrl_head_pose(c, t, &hp));
}
void
cb_push_position(struct xrt_device *xdev,
                 timepoint_ns t,
                 const struct xrt_vec3 *position,
                 const struct xrt_vec3 *position_variance)
{
	FakeController *c = reinterpret_cast<FakeController *>(xdev);
	struct xrt_pose hp;
	kalman_fusion_process_position(c->kf, t, position, position_variance, ctrl_head_pose(c, t, &hp));
}
bool
cb_get_unc(struct xrt_device *xdev, double *ps, double *os, double *ys)
{
	return kalman_fusion_get_pose_uncertainty(reinterpret_cast<FakeController *>(xdev)->kf, ps, os, ys);
}
bool
cb_get_predicted_pose(struct xrt_device *xdev, timepoint_ns when_ns, struct xrt_space_relation *out)
{
	// The matcher's raw prior — the filter's honest estimate, NOT the body-lock report (== production driver).
	kalman_fusion_get_predicted_pose(reinterpret_cast<FakeController *>(xdev)->kf, when_ns, out);
	return true;
}
bool
cb_predict_gate(struct xrt_device *xdev, const struct xrt_pose *P_xrworld_cam,
                const struct t_constellation_cam_calib *cc, const struct xrt_vec3 *led_obj, float out_zhat[2],
                float out_S[4])
{
	FakeController *c = reinterpret_cast<FakeController *>(xdev);
	struct kalman_led_camera_view view = {cc->fx,
	                                      cc->fy,
	                                      cc->cx,
	                                      cc->cy,
	                                      P_xrworld_cam->orientation,
	                                      P_xrworld_cam->position};
	struct kalman_led_observation obs = {};
	obs.led_obj = *led_obj; // observed_px unused by the gate predictor
	return kalman_fusion_predict_led_gate(c->kf, &obs, &view, out_zhat, out_S);
}
void
cb_noop_frame(struct xrt_device *, uint64_t, uint64_t)
{
}
void
cb_noop_bright(struct xrt_device *, uint8_t)
{
}

// ---- reassemble the four cam tiles into the mosaic the tracker splits by ROI ----
struct xrt_frame *
assemble_mosaic(const MosaicFrame &mf, const struct t_constellation_camera_group &cams, uint64_t seq)
{
	int W = 0, H = 0;
	for (int c = 0; c < cams.cam_count; c++) {
		W = std::max(W, cams.cams[c].roi.offset.w + cams.cams[c].roi.extent.w);
		H = std::max(H, cams.cams[c].roi.offset.h + cams.cams[c].roi.extent.h);
	}
	struct xrt_frame *f = nullptr;
	u_frame_create_one_off(XRT_FORMAT_L8, (uint32_t)W, (uint32_t)H, &f);
	memset(f->data, 0, f->size);
	const uint16_t expo = 6000; // header row metadata; constellation reads exposure from data[6..7]
	f->data[6] = (uint8_t)(expo >> 8);
	f->data[7] = (uint8_t)(expo & 0xff);
	for (int c = 0; c < cams.cam_count; c++) {
		cv::Mat im = cv::imread(mf.cam_png[c], cv::IMREAD_GRAYSCALE);
		if (im.empty()) {
			continue;
		}
		const int ox = cams.cams[c].roi.offset.w, oy = cams.cams[c].roi.offset.h;
		const int rw = std::min(im.cols, cams.cams[c].roi.extent.w);
		const int rh = std::min(im.rows, cams.cams[c].roi.extent.h);
		for (int r = 0; r < rh; r++) {
			memcpy(f->data + (size_t)(oy + r) * f->stride + ox, im.ptr<uint8_t>(r), rw);
		}
	}
	f->timestamp = mf.t_ns;
	f->source_timestamp = mf.t_ns;
	f->source_sequence = seq;
	return f;
}

} // namespace

// Per-controller state: each device gets its own ESKF, LED model, IMU stream, and CSV output. The
// tracker is single (one constellation_tracker_create) — each FakeController is registered via
// t_constellation_tracker_add_device so the multi-device code paths (predictive-ROI per-device gate,
// per-device blob labelling, per-device pose attempts) exercise.
struct CtrlSession
{
	std::unique_ptr<FakeController> ctrl;
	std::vector<ImuRow> imu;
	size_t imu_ii = 0;
	FILE *csv = nullptr;
	int opt_frames = 0;
	int locked = 0;
};

int
main(int argc, char **argv)
{
	if (argc < 5) {
		fprintf(stderr,
		        "usage: %s <frames_dir> <hmd-cameras.json> <telemetry_dir> "
		        "<left_ctrl.json> [right_ctrl.json] [out_dir]\n"
		        "  Replays all listed controllers simultaneously on the same tracker (the production\n"
		        "  multi-device path). First controller becomes device_id=1 (left), second device_id=2\n"
		        "  (right). out_dir, if given, receives one CSV per device (dev<id>.csv).\n",
		        argv[0]);
		return 2;
	}
	const std::string mav0 = argv[1], cam_json = argv[2], telem = argv[3];
	// Controller json paths (1..N). Optional trailing arg is the out_dir if it's not a .json file.
	std::vector<std::string> ctrl_jsons;
	std::string out_dir;
	for (int i = 4; i < argc; i++) {
		std::string a = argv[i];
		bool looks_like_json = a.size() >= 5 && a.compare(a.size() - 5, 5, ".json") == 0;
		if (looks_like_json) {
			ctrl_jsons.push_back(a);
		} else {
			out_dir = a; // last non-json arg = output dir
			break;
		}
	}
	if (ctrl_jsons.empty() || ctrl_jsons.size() > 2) {
		fprintf(stderr, "expected 1-2 controller json paths (got %zu)\n", ctrl_jsons.size());
		return 2;
	}

	struct t_constellation_camera_group cams = {};
	if (!load_camera_group(cam_json, &cams)) {
		fprintf(stderr, "FAILED to load camera group\n");
		return 1;
	}

	// Load all controllers (device_id assigned by position: first=1, second=2).
	std::vector<CtrlSession> sessions(ctrl_jsons.size());
	for (size_t s = 0; s < ctrl_jsons.size(); s++) {
		const uint8_t device_id = (uint8_t)(s + 1);
		struct t_constellation_led_model m = {};
		if (!load_led_model(ctrl_jsons[s], device_id, &m)) {
			fprintf(stderr, "FAILED to load LED model for device %u from %s\n", device_id,
			        ctrl_jsons[s].c_str());
			return 1;
		}
		sessions[s].ctrl = std::make_unique<FakeController>();
		sessions[s].ctrl->device_id = device_id;
		sessions[s].ctrl->leds.assign(m.leds, m.leds + m.num_leds);
		t_constellation_led_model_clear(&m);
		sessions[s].imu = load_imu(telem, device_id);
	}
	// Two frame sources: a G2_DUMP_FRAMES dir of controller PGMs (real LED frames), or a euroc mav0
	// dir (SLAM frames). Auto-detect by the euroc cam0/data layout.
	const bool euroc_mode = fs::exists(mav0 + "/cam0/data");
	std::vector<HeadPose> gt;
	std::vector<MosaicFrame> frames;
	if (euroc_mode) {
		gt = load_gt(mav0);
		apply_gravity_fix(gt, load_euroc_imu(mav0)); // align gt world to Y-up
		frames = index_euroc(mav0, cams.cam_count);
	} else {
		bool pgm_ok = false;
		frames = index_pgm_dump(mav0, cams.cam_count, telem, &pgm_ok); // real controller LED frames
		if (!pgm_ok) {
			fprintf(stderr,
			        "PGM dump failed integrity checks; recapture into an empty directory or remove stale frames\n");
			return 1;
		}
		gt = load_head_pose_telemetry(telem);          // true SLAM head pose, if the capture recorded it
		if (gt.empty()) {
			gt = head_poses_from_imu(load_imu(telem, 0)); // fallback: IMU-only reconstruction (drifts;
			                                              // prior-dependent paths e.g. the flip veto are unfaithful)
			printf("note: no head_pose.bin -> IMU-derived head pose (drifts; flip-veto A/B unreliable)\n");
		} else {
			printf("using RECORDED SLAM head pose (%zu samples) -> faithful camera->world\n", gt.size());
		}
	}
	// Keep only frames within the UNION of all controllers' IMU windows. Any frame that falls inside
	// at least one device's IMU window is replayed (per-device IMU readiness is enforced inside the
	// loop below). This preserves single-device behaviour and lets dual-device replay cover the full
	// motion span when one IMU stream starts slightly earlier than the other.
	{
		int64_t lo_union = INT64_MAX, hi_union = INT64_MIN;
		bool any = false;
		for (const CtrlSession &cs : sessions) {
			if (cs.imu.empty()) {
				continue;
			}
			any = true;
			lo_union = std::min(lo_union, cs.imu.front().t_ns - 100000000);
			hi_union = std::max(hi_union, cs.imu.back().t_ns + 100000000);
		}
		if (any) {
			std::vector<MosaicFrame> kept;
			for (const MosaicFrame &mf : frames) {
				if (mf.t_ns >= lo_union && mf.t_ns <= hi_union) {
					kept.push_back(mf);
				}
			}
			frames.swap(kept);
		}
	}
	{
		size_t total_leds = 0, total_imu = 0;
		for (const CtrlSession &cs : sessions) {
			total_leds += cs.ctrl->leds.size();
			total_imu += cs.imu.size();
		}
		printf("inputs: %d cams, %zu controllers (LED total %zu), %zu head poses, %zu frames, %zu IMU samples [%s]\n",
		       cams.cam_count, sessions.size(), total_leds, gt.size(), frames.size(), total_imu,
		       euroc_mode ? "euroc" : "pgm-dump");
		for (const CtrlSession &cs : sessions) {
			printf("  device %u: %zu LEDs, %zu IMU samples\n", cs.ctrl->device_id, cs.ctrl->leds.size(),
			       cs.imu.size());
		}
	}
	if (frames.empty()) {
		fprintf(stderr, "no frames to replay\n");
		return 1;
	}
	bool any_imu = false;
	for (const CtrlSession &cs : sessions) {
		if (!cs.imu.empty()) {
			any_imu = true;
			break;
		}
	}
	if (!any_imu) {
		fprintf(stderr, "no IMU to replay (any device)\n");
		return 1;
	}

	FakeHmd hmd = {};
	hmd.base.get_tracked_pose = hmd_get_pose;
	hmd.base.update_inputs = dev_noop_update;
	hmd.base.destroy = dev_noop_destroy;
	hmd.base.device_type = XRT_DEVICE_TYPE_GENERIC_TRACKER;
	snprintf(hmd.base.str, sizeof(hmd.base.str), "offline-hmd");
	hmd.gt = gt;

	// Initialise each controller's xrt_device + ESKF. The body-lock reference (hmd) is shared.
	for (CtrlSession &cs : sessions) {
		FakeController &ctrl = *cs.ctrl;
		ctrl.kf = kalman_fusion_create();
		ctrl.hmd = &hmd; // mirrors wmr_controller_attach_to_hmd's wcb->hmd_xdev
		ctrl.base.get_tracked_pose = ctrl_get_pose;
		ctrl.base.update_inputs = dev_noop_update;
		ctrl.base.destroy = dev_noop_destroy;
		ctrl.base.device_type = ctrl.device_id == 1 ? XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER
		                                             : XRT_DEVICE_TYPE_RIGHT_HAND_CONTROLLER;
		snprintf(ctrl.base.str, sizeof(ctrl.base.str), "offline-controller-%u", ctrl.device_id);
	}

	// Optional: emit a standard telemetry capture of the replay (G2_REPLAY_TELEMETRY=<dir>), so the
	// constellation's pose_attempt/frame/event streams are written and the existing analysis tools
	// (analyze.py, plot_session.py, imu_vs_optical_drift.py) work on the replay just like a live session.
	g2_telem_init(getenv("G2_REPLAY_TELEMETRY"));

	struct xrt_frame_context xfctx = {};
	struct t_constellation_tracker *tracker = nullptr;
	struct xrt_frame_sink *sink = nullptr;
	if (t_constellation_tracker_create(&xfctx, &hmd.base, &cams, &tracker, &sink, nullptr) != 0) {
		fprintf(stderr, "tracker create failed\n");
		return 1;
	}
	struct t_constellation_tracked_device_callbacks cbs = {};
	cbs.get_led_model = cb_get_led_model;
	cbs.notify_frame_received = cb_noop_frame;
	cbs.push_observed_pose = cb_push_pose;
	cbs.push_observed_position = cb_push_position;
	cbs.push_brightness_update = cb_noop_bright;
	cbs.push_observed_leds = cb_push_leds;
	cbs.get_pose_uncertainty = cb_get_unc;
	cbs.get_predicted_pose = cb_get_predicted_pose;
	cbs.predict_led_gate = cb_predict_gate;
	for (CtrlSession &cs : sessions) {
		t_constellation_tracker_add_device(tracker, &cs.ctrl->base, &cbs);
	}

	// Open per-device CSVs. out_dir is created if missing.
	if (!out_dir.empty()) {
		std::error_code ec;
		fs::create_directories(out_dir, ec);
		for (CtrlSession &cs : sessions) {
			char path[1024];
			snprintf(path, sizeof(path), "%s/dev%u.csv", out_dir.c_str(), cs.ctrl->device_id);
			cs.csv = fopen(path, "w");
			if (cs.csv != nullptr) {
				fprintf(cs.csv,
				        "t_ns,opt_valid,opt_px,opt_py,opt_pz,opt_qx,opt_qy,opt_qz,opt_qw,"
				        "pred_px,pred_py,pred_pz,pred_qx,pred_qy,pred_qz,pred_qw,pred_tracked,"
				        "hmd_px,hmd_py,hmd_pz,pred_to_hmd_m\n");
			} else {
				fprintf(stderr, "WARN: could not open %s for write\n", path);
			}
		}
	}

	uint64_t seq = 0;
	for (const MosaicFrame &mf : frames) {
		// Feed each device's IMU samples up to this frame's time so its matcher prior
		// (kalman_fusion_get_prediction) is current. Per-device streams advance independently.
		for (CtrlSession &cs : sessions) {
			while (cs.imu_ii < cs.imu.size() && cs.imu[cs.imu_ii].t_ns <= mf.t_ns) {
				const ImuRow &r = cs.imu[cs.imu_ii];
				struct xrt_imu_sample s = {};
				s.timestamp_ns = r.t_ns;
				s.accel_m_s2 = {r.ax, r.ay, r.az};
				s.gyro_rad_secs = {r.gx, r.gy, r.gz};
				kalman_fusion_process_imu_data(cs.ctrl->kf, &s, nullptr, nullptr);
				// Live head pose for the out-of-view body anchor (== production driver).
				struct xrt_pose hp_anchor;
				kalman_fusion_update_body_anchor(cs.ctrl->kf,
				                                 ctrl_head_pose(cs.ctrl.get(), s.timestamp_ns, &hp_anchor));
				cs.imu_ii++;
			}
			std::lock_guard<std::mutex> lk(cs.ctrl->cap_lock);
			cs.ctrl->opt_valid = false;
		}

		struct xrt_frame *f = assemble_mosaic(mf, cams, seq++);
		// Completion BARRIER (replaces the old fixed 8ms sleep, which was a race: a frame whose
		// ab-initio search ran past 8ms was read as "no optical pose" -> the yield % became
		// machine-load-dependent, audit H1). The tracker increments frames_completed exactly once per
		// frame at every pipeline exit; we drive one frame at a time, so only one is ever in flight.
		// Spin-wait (with a tiny yield/sleep) until THIS frame is fully processed before reading opt_*.
		const uint64_t done_before = t_constellation_tracker_debug_frames_completed(tracker);
		xrt_sink_push_frame(sink, f);
		{
			const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
			while (t_constellation_tracker_debug_frames_completed(tracker) < done_before + 1) {
				if (std::chrono::steady_clock::now() > deadline) {
					fprintf(stderr, "WARN: frame %lld completion barrier timed out\n",
					        (long long)mf.t_ns);
					break;
				}
				std::this_thread::sleep_for(std::chrono::microseconds(100));
			}
		}
		xrt_frame_reference(&f, nullptr); // release after processing

		// Snapshot per-device pose state and write CSVs.
		for (CtrlSession &cs : sessions) {
			struct xrt_space_relation fused = {};
			struct xrt_pose hp_pred;
			const struct xrt_pose *hp_pred_ptr = ctrl_head_pose(cs.ctrl.get(), mf.t_ns, &hp_pred);
			kalman_fusion_get_prediction(cs.ctrl->kf, mf.t_ns, &fused, hp_pred_ptr);
			const bool ptracked =
			    (fused.relation_flags & XRT_SPACE_RELATION_POSITION_TRACKED_BIT) != 0;
			cs.locked += ptracked ? 1 : 0;
			bool ov;
			struct xrt_pose op;
			{
				std::lock_guard<std::mutex> lk(cs.ctrl->cap_lock);
				ov = cs.ctrl->opt_valid;
				op = cs.ctrl->opt_pose;
			}
			cs.opt_frames += ov ? 1 : 0;
			if (cs.csv != nullptr) {
				fprintf(cs.csv, "%lld,%d,", (long long)mf.t_ns, ov ? 1 : 0);
				if (ov) {
					fprintf(cs.csv, "%.5f,%.5f,%.5f,%.6f,%.6f,%.6f,%.6f,", op.position.x,
					        op.position.y, op.position.z, op.orientation.x, op.orientation.y,
					        op.orientation.z, op.orientation.w);
				} else {
					fprintf(cs.csv, "nan,nan,nan,nan,nan,nan,nan,");
				}
				fprintf(cs.csv, "%.5f,%.5f,%.5f,%.6f,%.6f,%.6f,%.6f,%d,",
				        fused.pose.position.x, fused.pose.position.y, fused.pose.position.z,
				        fused.pose.orientation.x, fused.pose.orientation.y, fused.pose.orientation.z,
				        fused.pose.orientation.w, ptracked ? 1 : 0);
				if (hp_pred_ptr != nullptr) {
					const double dx = fused.pose.position.x - hp_pred_ptr->position.x;
					const double dy = fused.pose.position.y - hp_pred_ptr->position.y;
					const double dz = fused.pose.position.z - hp_pred_ptr->position.z;
					fprintf(cs.csv, "%.5f,%.5f,%.5f,%.5f\n", hp_pred_ptr->position.x,
					        hp_pred_ptr->position.y, hp_pred_ptr->position.z,
					        std::sqrt(dx * dx + dy * dy + dz * dz));
				} else {
					fprintf(cs.csv, "nan,nan,nan,nan\n");
				}
			}
		}
	}

	for (CtrlSession &cs : sessions) {
		if (cs.csv != nullptr) {
			fclose(cs.csv);
			cs.csv = nullptr;
		}
	}
	printf("replay done: %zu frames\n", frames.size());
	for (const CtrlSession &cs : sessions) {
		printf("  device %u: optical pose on %d (%.0f%%) | position-tracked on %d (%.0f%%)\n",
		       cs.ctrl->device_id, cs.opt_frames, 100.0 * cs.opt_frames / std::max<size_t>(frames.size(), 1),
		       cs.locked, 100.0 * cs.locked / std::max<size_t>(frames.size(), 1));
	}
	if (!out_dir.empty()) {
		printf("wrote per-device CSVs to %s/\n", out_dir.c_str());
	}

	xrt_frame_context_destroy_nodes(&xfctx); // stops the tracker threads
	g2_telem_shutdown();                     // flush + close the replay telemetry (no-op if disabled)
	for (CtrlSession &cs : sessions) {
		kalman_fusion_destroy(cs.ctrl->kf);
	}
	return 0;
}
