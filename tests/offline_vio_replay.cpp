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
#include <mutex>
#include <thread>

#include <cjson/cJSON.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "xrt/xrt_defines.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_frame.h"
#include "xrt/xrt_tracking.h"
#include "math/m_api.h"
#include "util/u_frame.h"
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
};

xrt_result_t
ctrl_get_pose(struct xrt_device *xdev, enum xrt_input_name, int64_t t, struct xrt_space_relation *rel)
{
	FakeController *c = reinterpret_cast<FakeController *>(xdev);
	kalman_fusion_get_prediction(c->kf, t, rel); // the IMU-dead-reckoned prior the matcher refines
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
	kalman_fusion_process_pose(c->kf, &s, nullptr, nullptr, 15);
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
	}
	struct xrt_vec2 var = {1.5f, 1.5f};
	kalman_fusion_process_led_observations(c->kf, t, obs, m, &view, &var, 8.0f, true);
}
bool
cb_get_unc(struct xrt_device *xdev, double *ps, double *os)
{
	return kalman_fusion_get_pose_uncertainty(reinterpret_cast<FakeController *>(xdev)->kf, ps, os);
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

int
main(int argc, char **argv)
{
	if (argc < 6) {
		fprintf(stderr,
		        "usage: %s <euroc mav0 dir> <hmd-cameras.json> <controller json> <telemetry dir> "
		        "<device_id 1=left 2=right> [out.csv]\n",
		        argv[0]);
		return 2;
	}
	const std::string mav0 = argv[1], cam_json = argv[2], ctrl_json = argv[3], telem = argv[4];
	const int device_id = atoi(argv[5]);
	const std::string out_csv = argc > 6 ? argv[6] : "";

	struct t_constellation_camera_group cams = {};
	if (!load_camera_group(cam_json, &cams)) {
		fprintf(stderr, "FAILED to load camera group\n");
		return 1;
	}
	std::vector<struct t_constellation_led> leds;
	{
		struct t_constellation_led_model m = {};
		if (!load_led_model(ctrl_json, (uint8_t)device_id, &m)) {
			fprintf(stderr, "FAILED to load LED model\n");
			return 1;
		}
		leds.assign(m.leds, m.leds + m.num_leds);
		t_constellation_led_model_clear(&m);
	}
	std::vector<HeadPose> gt = load_gt(mav0);
	std::vector<MosaicFrame> frames = index_euroc(mav0, cams.cam_count);
	std::vector<ImuRow> imu = load_imu(telem, device_id);
	printf("inputs: %d cams, %zu LEDs, %zu gt head poses, %zu frames, %zu IMU (device %d)\n", cams.cam_count,
	       leds.size(), gt.size(), frames.size(), imu.size(), device_id);
	if (frames.empty() || imu.empty()) {
		fprintf(stderr, "no frames or IMU to replay\n");
		return 1;
	}

	FakeHmd hmd = {};
	hmd.base.get_tracked_pose = hmd_get_pose;
	hmd.base.update_inputs = dev_noop_update;
	hmd.base.destroy = dev_noop_destroy;
	hmd.base.device_type = XRT_DEVICE_TYPE_GENERIC_TRACKER;
	snprintf(hmd.base.str, sizeof(hmd.base.str), "offline-hmd");
	hmd.gt = gt;

	FakeController ctrl = {};
	ctrl.kf = kalman_fusion_create();
	ctrl.device_id = (uint8_t)device_id;
	ctrl.leds = leds;
	ctrl.base.get_tracked_pose = ctrl_get_pose;
	ctrl.base.update_inputs = dev_noop_update;
	ctrl.base.destroy = dev_noop_destroy;
	ctrl.base.device_type =
	    device_id == 1 ? XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER : XRT_DEVICE_TYPE_RIGHT_HAND_CONTROLLER;
	snprintf(ctrl.base.str, sizeof(ctrl.base.str), "offline-controller");

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
	cbs.push_brightness_update = cb_noop_bright;
	cbs.push_observed_leds = cb_push_leds;
	cbs.get_pose_uncertainty = cb_get_unc;
	t_constellation_tracker_add_device(tracker, &ctrl.base, &cbs);

	FILE *csv = out_csv.empty() ? nullptr : fopen(out_csv.c_str(), "w");
	if (csv != nullptr) {
		fprintf(csv, "t_ns,opt_valid,opt_px,opt_py,opt_pz,opt_qx,opt_qy,opt_qz,opt_qw,"
		             "pred_px,pred_py,pred_pz,pred_qx,pred_qy,pred_qz,pred_qw,pred_tracked\n");
	}

	size_t ii = 0;
	uint64_t seq = 0;
	int opt_frames = 0, locked = 0;
	for (const MosaicFrame &mf : frames) {
		// Feed IMU up to this frame's time so the matcher's prior (get_prediction) is current.
		while (ii < imu.size() && imu[ii].t_ns <= mf.t_ns) {
			struct xrt_imu_sample s = {};
			s.timestamp_ns = imu[ii].t_ns;
			s.accel_m_s2 = {imu[ii].ax, imu[ii].ay, imu[ii].az};
			s.gyro_rad_secs = {imu[ii].gx, imu[ii].gy, imu[ii].gz};
			kalman_fusion_process_imu_data(ctrl.kf, &s, nullptr, nullptr);
			ii++;
		}
		{
			std::lock_guard<std::mutex> lk(ctrl.cap_lock);
			ctrl.opt_valid = false;
		}
		struct xrt_frame *f = assemble_mosaic(mf, cams, seq++);
		xrt_sink_push_frame(sink, f);
		std::this_thread::sleep_for(std::chrono::milliseconds(8)); // let the fast thread process
		xrt_frame_reference(&f, nullptr);                          // release after processing

		struct xrt_space_relation fused = {};
		kalman_fusion_get_prediction(ctrl.kf, mf.t_ns, &fused);
		const bool ptracked = (fused.relation_flags & XRT_SPACE_RELATION_POSITION_TRACKED_BIT) != 0;
		locked += ptracked ? 1 : 0;
		bool ov;
		struct xrt_pose op;
		{
			std::lock_guard<std::mutex> lk(ctrl.cap_lock);
			ov = ctrl.opt_valid;
			op = ctrl.opt_pose;
		}
		opt_frames += ov ? 1 : 0;
		if (csv != nullptr) {
			fprintf(csv, "%lld,%d,", (long long)mf.t_ns, ov ? 1 : 0);
			if (ov) {
				fprintf(csv, "%.5f,%.5f,%.5f,%.6f,%.6f,%.6f,%.6f,", op.position.x, op.position.y,
				        op.position.z, op.orientation.x, op.orientation.y, op.orientation.z,
				        op.orientation.w);
			} else {
				fprintf(csv, "nan,nan,nan,nan,nan,nan,nan,");
			}
			fprintf(csv, "%.5f,%.5f,%.5f,%.6f,%.6f,%.6f,%.6f,%d\n", fused.pose.position.x,
			        fused.pose.position.y, fused.pose.position.z, fused.pose.orientation.x,
			        fused.pose.orientation.y, fused.pose.orientation.z, fused.pose.orientation.w,
			        ptracked ? 1 : 0);
		}
	}
	if (csv != nullptr) {
		fclose(csv);
	}
	printf("replay done: %zu frames | optical pose on %d (%.0f%%) | position-tracked on %d (%.0f%%)\n",
	       frames.size(), opt_frames, 100.0 * opt_frames / frames.size(), locked,
	       100.0 * locked / frames.size());
	if (!out_csv.empty()) {
		printf("wrote %s\n", out_csv.c_str());
	}

	xrt_frame_context_destroy_nodes(&xfctx); // stops the tracker threads
	kalman_fusion_destroy(ctrl.kf);
	return 0;
}
