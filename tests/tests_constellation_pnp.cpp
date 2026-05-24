// Copyright 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Decoupled unit tests for the constellation PnP mirror-twin enumeration
 *        (ransac_pnp_pose_with_twin). Synthetic geometry with KNOWN ground truth —
 *        no filter, no head pose, no captured data — so a wrong solve fails rather
 *        than agreeing with itself. Validates the core property the front-end relies
 *        on: for the near-coplanar few-LED geometry that causes the controller flip,
 *        BOTH pose solutions (the true one and its mirror twin) are recovered, so the
 *        caller can pick the prior-consistent one instead of committing a flip.
 */
#include "catch_amalgamated.hpp"

#include "internal/ransac_pnp.h"
#include "internal/blobwatch.h"
#include "internal/camera_model.h"
#include "tracking/t_led_models.h"
#include "math/m_api.h"

#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr int MODEL_ID = 2;
constexpr double FX = 400.0, CX = 320.0;

//! Geodesic angle (deg) between two quaternions (q and -q identified).
double
quat_angle_deg(const struct xrt_quat &a, const struct xrt_quat &b)
{
	double d = std::fabs((double)a.x * b.x + (double)a.y * b.y + (double)a.z * b.z + (double)a.w * b.w);
	d = std::min(1.0, d);
	return 2.0 * std::acos(d) * 180.0 / M_PI;
}

struct xrt_quat
quat_from_rvec(const cv::Mat &rvec)
{
	const double a = cv::norm(rvec);
	struct xrt_vec3 axis = {0.f, 0.f, 1.f};
	if (a > 1e-9) {
		axis.x = (float)(rvec.at<double>(0) / a);
		axis.y = (float)(rvec.at<double>(1) / a);
		axis.z = (float)(rvec.at<double>(2) / a);
	}
	struct xrt_quat q;
	math_quat_from_angle_vector(a, &axis, &q);
	return q;
}

//! A zero-distortion pinhole camera (RADTAN_8 with all coeffs 0 == pinhole undistort).
struct camera_model
make_pinhole()
{
	struct camera_model cam = {};
	cam.width = 640;
	cam.height = 480;
	cam.calib.fx = cam.calib.fy = (float)FX;
	cam.calib.cx = cam.calib.cy = (float)CX;
	cam.calib.model = T_DISTORTION_OPENCV_RADTAN_8;
	return cam;
}

//! Project object points through (rvec,tvec) into pixels (pinhole, no distortion).
std::vector<cv::Point2f>
project(const std::vector<cv::Point3f> &obj, const cv::Mat &rvec, const cv::Mat &tvec)
{
	cv::Mat K = (cv::Mat_<double>(3, 3) << FX, 0, CX, 0, FX, CX, 0, 0, 1);
	std::vector<cv::Point2f> px;
	cv::projectPoints(obj, rvec, tvec, K, cv::Mat::zeros(4, 1, CV_64F), px);
	return px;
}

//! Build LED model + blobs for an object point set viewed at (rvec,tvec).
void
build_scene(const std::vector<cv::Point3f> &obj,
            const cv::Mat &rvec,
            const cv::Mat &tvec,
            std::vector<t_constellation_led> &leds,
            t_constellation_led_model &model,
            std::vector<blob> &blobs)
{
	const std::vector<cv::Point2f> px = project(obj, rvec, tvec);
	leds.resize(obj.size());
	blobs.assign(obj.size(), blob{});
	for (size_t i = 0; i < obj.size(); i++) {
		leds[i] = t_constellation_led{};
		leds[i].id = (uint8_t)i;
		leds[i].pos = {obj[i].x, obj[i].y, obj[i].z};
		blobs[i].x = px[i].x;
		blobs[i].y = px[i].y;
		blobs[i].led_id = LED_MAKE_ID(MODEL_ID, (uint16_t)i);
	}
	model = t_constellation_led_model{};
	model.id = MODEL_ID;
	model.leds = leds.data();
	model.num_leds = (uint8_t)leds.size();
}

} // namespace

TEST_CASE("near-coplanar few-LED solve recovers BOTH twins; truth is one of them")
{
	// A tilted planar LED patch (z=0 in object frame) — the geometry that admits the mirror two-fold.
	const std::vector<cv::Point3f> obj = {
	    {-0.03f, -0.02f, 0.f}, {0.03f, -0.02f, 0.f}, {0.03f, 0.02f, 0.f}, {-0.03f, 0.02f, 0.f}, {0.f, 0.f, 0.f}};
	const cv::Mat rvec_true = (cv::Mat_<double>(3, 1) << 0.5, 0.25, 0.1); // ~33 deg, off-axis -> ambiguous
	const cv::Mat tvec_true = (cv::Mat_<double>(3, 1) << 0.02, -0.01, 0.45);

	std::vector<t_constellation_led> leds;
	t_constellation_led_model model;
	std::vector<blob> blobs;
	build_scene(obj, rvec_true, tvec_true, leds, model, blobs);
	struct camera_model cam = make_pinhole();

	struct xrt_pose pose = {};
	pose.orientation.w = 1.f;
	struct xrt_pose twin = {};
	bool has_twin = false;
	int nleds = 0, ninliers = 0;
	const bool ok = ransac_pnp_pose_with_twin(&pose, blobs.data(), (int)blobs.size(), &model, &cam, &nleds,
	                                          &ninliers, &twin, &has_twin);
	REQUIRE(ok);

	const struct xrt_quat qt = quat_from_rvec(rvec_true);
	const double a_primary = quat_angle_deg(pose.orientation, qt);
	const double a_twin = has_twin ? quat_angle_deg(twin.orientation, qt) : 1e3;

	// The truth must be recovered as ONE of the candidates (reprojection cannot say which — that's the
	// prior's job downstream; here we only require that the correct mode was enumerated, not lost).
	REQUIRE(std::min(a_primary, a_twin) < 5.0);
	// The tilted planar geometry genuinely has a second mode: a distinct twin must be reported.
	REQUIRE(has_twin);
	REQUIRE(quat_angle_deg(pose.orientation, twin.orientation) > 20.0);
}

TEST_CASE("well-conditioned non-coplanar solve is unambiguous (recovers truth, no spurious twin)")
{
	// LEDs spread in depth (z) -> well-conditioned -> no mirror ambiguity.
	const std::vector<cv::Point3f> obj = {{-0.03f, -0.02f, 0.00f},  {0.03f, -0.02f, 0.02f},
	                                      {0.03f, 0.02f, -0.015f},  {-0.03f, 0.02f, 0.018f},
	                                      {0.0f, 0.0f, 0.03f},      {0.01f, -0.03f, -0.02f}};
	const cv::Mat rvec_true = (cv::Mat_<double>(3, 1) << 0.3, -0.4, 0.2);
	const cv::Mat tvec_true = (cv::Mat_<double>(3, 1) << -0.01, 0.02, 0.5);

	std::vector<t_constellation_led> leds;
	t_constellation_led_model model;
	std::vector<blob> blobs;
	build_scene(obj, rvec_true, tvec_true, leds, model, blobs);
	struct camera_model cam = make_pinhole();

	struct xrt_pose pose = {};
	pose.orientation.w = 1.f;
	struct xrt_pose twin = {};
	bool has_twin = false;
	int nleds = 0, ninliers = 0;
	const bool ok = ransac_pnp_pose_with_twin(&pose, blobs.data(), (int)blobs.size(), &model, &cam, &nleds,
	                                          &ninliers, &twin, &has_twin);
	REQUIRE(ok);

	const struct xrt_quat qt = quat_from_rvec(rvec_true);
	// Unique solve: the primary must BE the truth (orientation + position), tight.
	REQUIRE(quat_angle_deg(pose.orientation, qt) < 3.0);
	REQUIRE(std::fabs(pose.position.x - tvec_true.at<double>(0)) < 0.01);
	REQUIRE(std::fabs(pose.position.y - tvec_true.at<double>(1)) < 0.01);
	REQUIRE(std::fabs(pose.position.z - tvec_true.at<double>(2)) < 0.01);
	// If a twin is reported at all here it must not be a gross (mirror-scale) alternative.
	if (has_twin) {
		REQUIRE(quat_angle_deg(pose.orientation, twin.orientation) < 20.0);
	}
}

TEST_CASE("degenerate inputs are rejected, not crashed")
{
	struct camera_model cam = make_pinhole();
	struct xrt_pose pose = {};
	pose.orientation.w = 1.f;
	struct xrt_pose twin = {};
	bool has_twin = true;
	int nleds = -1, ninliers = -1;

	SECTION("fewer than 4 LEDs -> false")
	{
		const std::vector<cv::Point3f> obj = {{-0.03f, 0.f, 0.f}, {0.03f, 0.f, 0.f}, {0.f, 0.03f, 0.f}};
		const cv::Mat rvec = (cv::Mat_<double>(3, 1) << 0.1, 0.1, 0.1);
		const cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.45);
		std::vector<t_constellation_led> leds;
		t_constellation_led_model model;
		std::vector<blob> blobs;
		build_scene(obj, rvec, tvec, leds, model, blobs);
		const bool ok = ransac_pnp_pose_with_twin(&pose, blobs.data(), (int)blobs.size(), &model, &cam, &nleds,
		                                          &ninliers, &twin, &has_twin);
		REQUIRE_FALSE(ok);
		REQUIRE_FALSE(has_twin);
	}

	SECTION("zero blobs -> false")
	{
		t_constellation_led_model model = {};
		model.id = MODEL_ID;
		model.num_leds = 0;
		const bool ok =
		    ransac_pnp_pose_with_twin(&pose, nullptr, 0, &model, &cam, &nleds, &ninliers, &twin, &has_twin);
		REQUIRE_FALSE(ok);
		REQUIRE_FALSE(has_twin);
	}
}
