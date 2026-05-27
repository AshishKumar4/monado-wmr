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
#include "internal/joint_pnp.h"
#include "internal/pose_metrics.h"
#include "internal/correspondence_search.h"
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

// ===========================================================================
// THE unified anisotropic prior-orientation split (pose_metrics_prior_orient_split):
// the single mechanism that feeds the soft mirror-flip cost selecting the correct P3P
// twin at BOTH the front-end twin site and the ab-initio search. Decoupled (synthetic
// quaternions, known truth). Asserts the two anisotropy claims the design rests on:
//   - a TILT flip is separated by the (driftless) tilt component even with NO yaw prior;
//   - a PURE-YAW difference is attributed entirely to yaw (tilt ~0), so the tilt term
//     contributes nothing and the yaw term carries the cost.
// ===========================================================================

namespace {
//! Quaternion from an axis-angle (deg) about a unit axis.
struct xrt_quat
quat_axis_deg(double ax, double ay, double az, double deg)
{
	struct xrt_vec3 axis = {(float)ax, (float)ay, (float)az};
	math_vec3_normalize(&axis);
	struct xrt_quat q;
	math_quat_from_angle_vector(deg * M_PI / 180.0, &axis, &q);
	return q;
}
} // namespace

TEST_CASE("unified prior split: a TILT flip is rejected by the driftless tilt term with NO yaw prior")
{
	// World-up is +Y; the split axis is the world-up expressed in the working frame (here: identity frame,
	// so up = +Y). The prior is the controller upright facing the camera.
	const struct xrt_vec3 up = {0.f, 1.f, 0.f};
	const struct xrt_quat q_prior = quat_axis_deg(0, 1, 0, 0); // identity (upright)

	// The CORRECT candidate differs from the prior by a pure YAW (about up) of 40 deg — and ALSO a tiny
	// real tilt of 3 deg. Its TILT term must stay small (driftless agreement) regardless of the yaw.
	struct xrt_quat q_yaw40 = quat_axis_deg(0, 1, 0, 40);
	struct xrt_quat q_tilt3 = quat_axis_deg(1, 0, 0, 3);
	struct xrt_quat q_correct;
	math_quat_rotate(&q_yaw40, &q_tilt3, &q_correct); // yaw . small-tilt

	// The MIRROR TWIN tilts the controller ~120 deg about a horizontal axis (the classic flip) — possibly
	// combined with some yaw. Its TILT term must blow up.
	struct xrt_quat q_flip = quat_axis_deg(1, 0.2, 0, 120);
	struct xrt_quat q_twin;
	math_quat_rotate(&q_yaw40, &q_flip, &q_twin);

	double tilt_c = 0, yaw_c = 0, tilt_t = 0, yaw_t = 0;
	pose_metrics_prior_orient_split(&q_correct, &q_prior, &up, &tilt_c, &yaw_c);
	pose_metrics_prior_orient_split(&q_twin, &q_prior, &up, &tilt_t, &yaw_t);

	const double TILT_TOL = 30.0 * M_PI / 180.0; // GRAVITY_TILT_TOL
	INFO("correct: tilt=" << tilt_c * 57.3 << " yaw=" << yaw_c * 57.3
	                       << "  twin: tilt=" << tilt_t * 57.3 << " yaw=" << yaw_t * 57.3);
	// The correct twin's tilt is small (real motion), well within the bound — accepted on TILT ALONE.
	CHECK(tilt_c < TILT_TOL);
	CHECK(tilt_c < 5.0 * M_PI / 180.0); // ~the 3 deg real tilt; the 40 deg yaw does NOT leak into tilt
	// The flipped twin's tilt is huge -> rejected by the tilt term even though we used NO yaw information.
	CHECK(tilt_t > TILT_TOL);
	CHECK(tilt_t > 90.0 * M_PI / 180.0);
	// And the tilt term genuinely SEPARATES them (the whole point): driftless tilt is the discriminator.
	CHECK(tilt_t > tilt_c + 80.0 * M_PI / 180.0);
}

TEST_CASE("unified prior split: a PURE-YAW difference is attributed to yaw, not tilt (the tilt term stays 0)")
{
	const struct xrt_vec3 up = {0.f, 1.f, 0.f};
	const struct xrt_quat q_prior = quat_axis_deg(0, 1, 0, 0);

	// A pure-yaw mirror: 175 deg about world-up, ZERO tilt. The split must report tilt ~0 (gravity is blind
	// to a yaw flip, so the tilt term of the soft cost contributes nothing) and the whole error as YAW (so
	// the yaw term, scaled by the live yaw sigma, is what carries the cost).
	struct xrt_quat q_yawflip = quat_axis_deg(0, 1, 0, 175);
	double tilt = 0, yaw = 0;
	pose_metrics_prior_orient_split(&q_yawflip, &q_prior, &up, &tilt, &yaw);
	INFO("pure-yaw: tilt=" << tilt * 57.3 << " yaw=" << yaw * 57.3);
	CHECK(tilt < 1.0 * M_PI / 180.0);   // gravity blind to a yaw flip -> no tilt-term penalty
	CHECK(yaw > 170.0 * M_PI / 180.0);  // the full ~175 deg is correctly attributed to yaw

	// A legitimate, drifted-but-real yaw of 50 deg (no tilt): tilt ~0, yaw 50. Its yaw term stays modest
	// against a live yaw sigma, so a real drifted yaw is kept (not penalised away like a ~180 deg flip).
	struct xrt_quat q_yaw50 = quat_axis_deg(0, 1, 0, 50);
	pose_metrics_prior_orient_split(&q_yaw50, &q_prior, &up, &tilt, &yaw);
	INFO("real-yaw50: tilt=" << tilt * 57.3 << " yaw=" << yaw * 57.3);
	CHECK(tilt < 1.0 * M_PI / 180.0);
	CHECK(yaw == Catch::Approx(50.0 * M_PI / 180.0).margin(2.0 * M_PI / 180.0));
}

TEST_CASE("unified prior split: degenerate up-axis defers entirely to yaw (no tilt)")
{
	// A zero-length up (no gravity reference) must not crash and must put all difference into yaw,
	// so the tilt term contributes nothing and the decision falls to the yaw term alone.
	const struct xrt_vec3 up = {0.f, 0.f, 0.f};
	const struct xrt_quat q_prior = quat_axis_deg(0, 1, 0, 0);
	struct xrt_quat q_cand = quat_axis_deg(1, 0.3f, 0, 100);
	double tilt = 123.0, yaw = -1.0;
	pose_metrics_prior_orient_split(&q_cand, &q_prior, &up, &tilt, &yaw);
	CHECK(tilt == 0.0);
	CHECK(yaw > 90.0 * M_PI / 180.0);
	CHECK(std::isfinite(yaw));
}

TEST_CASE("unified associator end-to-end: real twin enumeration + the split picks the prior-consistent twin")
{
	// The real near-coplanar tilt-flip scene (same as the BOTH-twins test). The two recovered twins differ
	// mostly in TILT. With a prior at the TRUE tilt (but a STALE/absent yaw), the unified split must score
	// the correct twin's tilt low and the flipped twin's tilt high — i.e. select the correct mode with NO
	// fresh yaw, exactly the live flip-cure. The blobs/LEDs are in the camera (OpenCV) frame, so the split
	// axis is world-up expressed in that frame.
	const std::vector<cv::Point3f> obj = {
	    {-0.03f, -0.02f, 0.f}, {0.03f, -0.02f, 0.f}, {0.03f, 0.02f, 0.f}, {-0.03f, 0.02f, 0.f}, {0.f, 0.f, 0.f}};
	const cv::Mat rvec_true = (cv::Mat_<double>(3, 1) << 0.5, 0.25, 0.1);
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
	REQUIRE(ransac_pnp_pose_with_twin(&pose, blobs.data(), (int)blobs.size(), &model, &cam, &nleds, &ninliers,
	                                  &twin, &has_twin));
	REQUIRE(has_twin);

	const struct xrt_quat q_true = quat_from_rvec(rvec_true);
	// Identify which recovered solution is the truth and which is the flip.
	const bool primary_is_true = quat_angle_deg(pose.orientation, q_true) < quat_angle_deg(twin.orientation, q_true);
	const struct xrt_quat q_correct = primary_is_true ? pose.orientation : twin.orientation;
	const struct xrt_quat q_flip = primary_is_true ? twin.orientation : pose.orientation;

	// PRIOR: the true tilt with a DELIBERATELY WRONG yaw (rotate the truth by 35 deg about world-up) — the
	// offline/stale-yaw case. World-up in this OpenCV camera frame is roughly -Y (objects above project up);
	// the split is sign-agnostic in the axis, so either Y sign gives the same tilt/yaw magnitudes.
	const struct xrt_vec3 up = {0.f, 1.f, 0.f};
	struct xrt_quat yaw35 = quat_axis_deg(up.x, up.y, up.z, 35);
	struct xrt_quat q_prior;
	math_quat_rotate(&yaw35, &q_true, &q_prior);

	double tilt_correct = 0, yaw_correct = 0, tilt_flip = 0, yaw_flip = 0;
	pose_metrics_prior_orient_split(&q_correct, &q_prior, &up, &tilt_correct, &yaw_correct);
	pose_metrics_prior_orient_split(&q_flip, &q_prior, &up, &tilt_flip, &yaw_flip);

	const double TILT_TOL = 30.0 * M_PI / 180.0;
	INFO("correct tilt=" << tilt_correct * 57.3 << "  flip tilt=" << tilt_flip * 57.3);
	// The unified mechanism: the correct twin's tilt is within the driftless bound, the flip's is not —
	// so the tilt term ALONE (no fresh yaw) selects the correct twin and rejects the flip.
	CHECK(tilt_correct < TILT_TOL);
	CHECK(tilt_flip > TILT_TOL);
	CHECK(tilt_flip > tilt_correct);
}

// ===========================================================================
// FRAME regression: the split is about a CAMERA-frame up axis, so the candidate-
// vs-prior relative rotation MUST be the camera-frame (left) relative q_cand.q_prior^-1,
// not the object-frame q_prior^-1.q_cand. For a TILTED prior, a pure WORLD-yaw of the
// controller must give tilt = 0 (gravity is blind to yaw). The object-frame form leaks
// ~sin(tilt)*yaw into tilt (33 deg tilt + 90 deg yaw -> 45.3 deg false tilt), which would
// over-reject a legitimate tilted+yaw-drifted frame. This test passes ONLY on the
// camera-frame relative; it FAILS on the object-frame form (the bug it guards).
// ===========================================================================
TEST_CASE("prior split frame: a pure world-yaw on a TILTED prior gives tilt~0 (camera-frame relative)")
{
	const struct xrt_vec3 up = {0.f, 1.f, 0.f};

	// Each case: a prior TILTED about world-X by tilt_deg, and a candidate that is the prior with an
	// additional pure WORLD-yaw (about up) of yaw_deg LEFT-applied (a world-frame rotation). Gravity sees
	// no tilt change, so the split's tilt must be ~0 and the whole difference must land in yaw.
	struct
	{
		double tilt_deg, yaw_deg;
	} cases[] = {{33, 90}, {60, 40}, {45, 120}, {20, 90}};

	for (const auto &c : cases) {
		const struct xrt_quat q_prior = quat_axis_deg(1, 0, 0, c.tilt_deg);
		const struct xrt_quat q_yaw = quat_axis_deg(up.x, up.y, up.z, c.yaw_deg);
		struct xrt_quat q_cand;
		math_quat_rotate(&q_yaw, &q_prior, &q_cand); // world-yaw . prior == pure world-yaw of the prior

		double tilt = -1.0, yaw = -1.0;
		pose_metrics_prior_orient_split(&q_cand, &q_prior, &up, &tilt, &yaw);
		INFO("tilt_prior=" << c.tilt_deg << " yaw=" << c.yaw_deg << " -> split tilt=" << tilt * 57.3
		                   << " yaw=" << yaw * 57.3);
		// Camera-frame relative: a pure world-yaw is ALL yaw, ZERO tilt. (Object-frame would give
		// sin(tilt)*yaw of false tilt, e.g. 45.3 deg for 33/90 — the regression this guards.)
		CHECK(tilt < 0.5 * M_PI / 180.0);
		CHECK(yaw == Catch::Approx(c.yaw_deg * M_PI / 180.0).margin(1.0 * M_PI / 180.0));
	}
}

// ===========================================================================
// The soft mirror-flip cost (pose_metrics_prior_orient_cost +
// pose_metrics_score_is_better_pose_prior): the prior-consistency re-rank.
// Teeth: a near-180 flip with a FRESH GOOD prior is NOT selected even when
// it reprojects marginally BETTER (the prior penalty decides), while a real-but-
// drifted moderate yaw IS kept; and the decision FLIPS if the prior term is
// removed — proving the prior term is load-bearing, not the reprojection alone.
// Decoupled (synthetic scores + quaternions, known truth; no filter/capture).
// ===========================================================================

namespace {
// The production soft-cost knobs (mirror t_constellation_tracking.c, kept in sync via these locals so the
// test exercises the real decision policy through the public helpers, not a re-implementation).
constexpr double SIGMA_TILT = 30.0 * M_PI / 180.0; // GRAVITY_TILT_TOL
constexpr double HUBER_KNEE = 3.0;                 // PRIOR_GATE_SIGMA (sigmas)
constexpr double COST_WEIGHT = 1.0;                // FLIP_COST_WEIGHT

//! A GOOD pose_metrics with a given summed reprojection error over @p matched blobs.
struct pose_metrics
good_score(int matched, double reproj_sum_px2)
{
	struct pose_metrics s = {};
	s.match_flags = (enum pose_match_flags)(POSE_MATCH_GOOD | POSE_HAD_PRIOR);
	s.matched_blobs = matched;
	s.reprojection_error = reproj_sum_px2;
	return s;
}
} // namespace

TEST_CASE("soft flip cost: a near-180 flip is NOT selected even when it reprojects marginally better")
{
	const struct xrt_vec3 up = {0.f, 1.f, 0.f};
	const struct xrt_quat q_prior = quat_axis_deg(0, 1, 0, 0); // upright, facing camera

	// CORRECT candidate: a small real tilt + a small real yaw — both within the tracked envelope.
	struct xrt_quat q_yaw20 = quat_axis_deg(0, 1, 0, 20);
	struct xrt_quat q_tilt4 = quat_axis_deg(1, 0, 0, 4);
	struct xrt_quat q_correct;
	math_quat_rotate(&q_yaw20, &q_tilt4, &q_correct);

	// FLIP twin: ~150 deg tilt about a horizontal axis (the classic mirror flip).
	struct xrt_quat q_flip = quat_axis_deg(1, 0.15f, 0, 150);

	// FRESH yaw prior: a tight live yaw sigma (the fusion is confidently tracking yaw).
	const double sigma_yaw_fresh = 15.0 * M_PI / 180.0;

	const double cost_correct = pose_metrics_prior_orient_cost(&q_correct, &q_prior, &up, SIGMA_TILT,
	                                                           sigma_yaw_fresh, HUBER_KNEE, COST_WEIGHT);
	const double cost_flip = pose_metrics_prior_orient_cost(&q_flip, &q_prior, &up, SIGMA_TILT,
	                                                        sigma_yaw_fresh, HUBER_KNEE, COST_WEIGHT);
	INFO("cost_correct=" << cost_correct << "  cost_flip=" << cost_flip);
	CHECK(cost_flip > cost_correct + 5.0); // the flip's penalty dwarfs the correct twin's

	// The flip even reprojects MARGINALLY BETTER (mirror twins reproject near-identically; give the flip the
	// edge on raw reprojection) yet must NOT be selected — the prior penalty is the decider.
	struct pose_metrics primary = good_score(5, 6.0); // correct twin, summed px^2
	struct pose_metrics twin = good_score(5, 5.5);    // flip twin, slightly LOWER raw reprojection
	// WITH the prior term: the correct (primary) is kept — the flip is not taken.
	CHECK_FALSE(pose_metrics_score_is_better_pose_prior(&primary, cost_correct, &twin, cost_flip));
	// WITHOUT the prior term (both costs 0): the flip's lower reprojection wins — i.e. the soft cost is
	// what prevents the flip. This is the teeth: the prior term is load-bearing, not the reprojection.
	CHECK(pose_metrics_score_is_better_pose_prior(&primary, 0.0, &twin, 0.0));
}

TEST_CASE("soft flip cost: a real-but-drifted moderate yaw is KEPT, not penalised away")
{
	const struct xrt_vec3 up = {0.f, 1.f, 0.f};
	const struct xrt_quat q_prior = quat_axis_deg(0, 1, 0, 0);

	// A legitimate, drifted-but-real yaw of 55 deg with ~0 tilt — a real heading the gyro has drifted to.
	struct xrt_quat q_realyaw = quat_axis_deg(0, 1, 0, 55);
	// The mirror twin would be a ~180 deg pure-yaw flip of it.
	struct xrt_quat q_yawflip = quat_axis_deg(0, 1, 0, 235); // 55 + 180

	// Yaw prior is alive but LOOSE (drift has widened the live sigma) — so the drifted real yaw self-deweights.
	const double sigma_yaw_loose = 40.0 * M_PI / 180.0;

	const double cost_real = pose_metrics_prior_orient_cost(&q_realyaw, &q_prior, &up, SIGMA_TILT,
	                                                       sigma_yaw_loose, HUBER_KNEE, COST_WEIGHT);
	const double cost_yawflip = pose_metrics_prior_orient_cost(&q_yawflip, &q_prior, &up, SIGMA_TILT,
	                                                          sigma_yaw_loose, HUBER_KNEE, COST_WEIGHT);
	INFO("cost_real=" << cost_real << "  cost_yawflip=" << cost_yawflip);
	// The drifted-but-real yaw's penalty stays modest; the pure-yaw flip's is far larger -> the flip loses,
	// the real yaw is kept (never dropped, never out-ranked by its own flip).
	CHECK(cost_yawflip > cost_real + 5.0);

	struct pose_metrics primary = good_score(5, 6.0); // the real drifted-yaw pose
	struct pose_metrics twin = good_score(5, 6.0);    // its pure-yaw flip, equal reprojection
	CHECK_FALSE(pose_metrics_score_is_better_pose_prior(&primary, cost_real, &twin, cost_yawflip));
}

TEST_CASE("soft flip cost: an untracked (large) yaw sigma defers to reprojection (cold-start preserved)")
{
	const struct xrt_vec3 up = {0.f, 1.f, 0.f};
	const struct xrt_quat q_prior = quat_axis_deg(0, 1, 0, 0);

	// A pure-yaw flip with NO tilt and an untracked (half-turn) yaw sigma: the yaw term must vanish so the
	// flip is NOT penalised on yaw — at cold start the search must run on reprojection alone.
	struct xrt_quat q_yawflip = quat_axis_deg(0, 1, 0, 175);
	const double sigma_yaw_untracked = 180.0 * M_PI / 180.0; // FLIP_COST_YAW_SIGMA_MAX

	const double cost = pose_metrics_prior_orient_cost(&q_yawflip, &q_prior, &up, SIGMA_TILT,
	                                                  sigma_yaw_untracked, HUBER_KNEE, COST_WEIGHT);
	INFO("untracked pure-yaw cost=" << cost);
	// (175/180)^2 ~ 0.95 < knee^2 -> still quadratic, a tiny penalty; tilt ~0 contributes nothing.
	CHECK(cost < 2.0);

	// A TILT flip with the SAME untracked yaw, however, is still rejected: tilt is driftless, so its sigma
	// stays tight regardless of yaw trust -> huge penalty even at cold-yaw.
	struct xrt_quat q_tiltflip = quat_axis_deg(1, 0.15f, 0, 150);
	const double cost_tilt = pose_metrics_prior_orient_cost(&q_tiltflip, &q_prior, &up, SIGMA_TILT,
	                                                       sigma_yaw_untracked, HUBER_KNEE, COST_WEIGHT);
	INFO("untracked tilt-flip cost=" << cost_tilt);
	CHECK(cost_tilt > 10.0);
}

TEST_CASE("soft flip cost: stale-prior yaw_sigma at the CEILING still penalises a 180° flip (F1 guard)")
{
	const struct xrt_vec3 up = {0.f, 1.f, 0.f};
	const struct xrt_quat q_prior = quat_axis_deg(0, 1, 0, 0);

	// Post-dropout stale prior: yaw_sigma can inflate up to FLIP_COST_YAW_SIGMA_MAX. A near-180° flip
	// at the ceiling must still cost meaningfully more than legitimate fast yaw motion at the same
	// ceiling — otherwise the matcher coin-flips between the twins. With the 90° cap a 180° flip
	// yields d² = (180/90)² = 4 (Huber-quadratic at s=2 < knee=3) while a legitimate 60° yaw still
	// costs d² = (60/90)² ≈ 0.44 — a 9× margin. With the prior 180° cap the same flip cost d²=1.0
	// against legitimate 60° cost d²≈0.11 — same 9× ratio in cost-space but a tiny ABSOLUTE penalty
	// that loses to a marginally-better reprojection. The cap is the empirical knob for "how much
	// flip-cost a stale-prior frame should pay".
	const double cap = 90.0 * M_PI / 180.0; // current FLIP_COST_YAW_SIGMA_MAX after F1
	struct xrt_quat q_180_flip = quat_axis_deg(0, 1, 0, 175);
	struct xrt_quat q_60_motion = quat_axis_deg(0, 1, 0, 60);
	const double cost_flip = pose_metrics_prior_orient_cost(&q_180_flip, &q_prior, &up, SIGMA_TILT,
	                                                       cap, HUBER_KNEE, COST_WEIGHT);
	const double cost_motion = pose_metrics_prior_orient_cost(&q_60_motion, &q_prior, &up, SIGMA_TILT,
	                                                         cap, HUBER_KNEE, COST_WEIGHT);
	INFO("180° flip cost at cap=90°: " << cost_flip << "  60° motion cost: " << cost_motion);
	// The flip cost must clearly dominate the legitimate-motion cost.
	CHECK(cost_flip > 3.0);
	CHECK(cost_motion < 0.6);
	CHECK(cost_flip > cost_motion * 5.0);
}

TEST_CASE("soft flip cost: head-anchored yaw cue penalises a head-relative 180° flip (Fix A)")
{
	// Fix A adds a SECOND independent prior term: the candidate's head-relative orientation vs the last
	// accepted controller-in-head orientation. A 180° mirror flip in the WORLD frame is also a 180° flip
	// in HEAD frame (the head doesn't change between consecutive frames, the controller does), so the
	// head-anchored term penalises the same flip independently — doubling the discrimination at the gate.
	const struct xrt_vec3 imu_up = {0.f, 1.f, 0.f}; // world-up rotated into head/IMU frame (identity head)
	const struct xrt_quat q_last_head_rel = quat_axis_deg(0, 1, 0, 0); // controller-in-head at last frame
	// Production knobs (mirror t_constellation_tracking.c):
	const double SIGMA_TILT_HEAD = 180.0 * M_PI / 180.0; // HEAD_YAW_CUE_TILT_SIGMA -> effectively disable tilt
	const double SIGMA_YAW_HEAD = 60.0 * M_PI / 180.0;   // HEAD_YAW_CUE_YAW_SIGMA

	// Same controller still in head frame, but two candidates: the prior-consistent twin (~5° drift) and
	// the 180° mirror twin (the flip we want to reject).
	const struct xrt_quat q_cand_consistent = quat_axis_deg(0, 1, 0, 5);
	const struct xrt_quat q_cand_flipped = quat_axis_deg(0, 1, 0, 175);

	const double cost_consistent = pose_metrics_prior_orient_cost(
	    &q_cand_consistent, &q_last_head_rel, &imu_up, SIGMA_TILT_HEAD, SIGMA_YAW_HEAD, HUBER_KNEE,
	    COST_WEIGHT);
	const double cost_flipped = pose_metrics_prior_orient_cost(
	    &q_cand_flipped, &q_last_head_rel, &imu_up, SIGMA_TILT_HEAD, SIGMA_YAW_HEAD, HUBER_KNEE,
	    COST_WEIGHT);
	INFO("Fix A: head-rel flip cost=" << cost_flipped << "  consistent cost=" << cost_consistent);
	// 180° / 60° = 3 sigma -> Huber knee — penalty becomes linear at d=3, so cost should be substantial.
	CHECK(cost_flipped > 3.0);
	CHECK(cost_consistent < 0.05);
	CHECK(cost_flipped > cost_consistent * 50.0);
}

TEST_CASE("soft flip cost: head-anchored cue tolerates legitimate fast head-vs-controller motion (Fix A)")
{
	// Sanity bound for Fix A's yaw sigma choice (60°): a legitimate ~30° head-relative yaw delta between
	// consecutive frames (one frame at 30Hz of fast head-and-controller motion in opposite directions
	// totalling ~900°/s relative angular rate) must cost MUCH less than a 180° flip at the same sigma —
	// otherwise the cue would over-fire and reject legitimate motion as if it were a flip.
	const struct xrt_vec3 imu_up = {0.f, 1.f, 0.f};
	const struct xrt_quat q_last_head_rel = quat_axis_deg(0, 1, 0, 0);
	const double SIGMA_TILT_HEAD = 180.0 * M_PI / 180.0;
	const double SIGMA_YAW_HEAD = 60.0 * M_PI / 180.0;

	const struct xrt_quat q_legit_motion = quat_axis_deg(0, 1, 0, 30); // 30° head-rel yaw in 1 frame
	const struct xrt_quat q_flip = quat_axis_deg(0, 1, 0, 175);

	const double cost_legit = pose_metrics_prior_orient_cost(
	    &q_legit_motion, &q_last_head_rel, &imu_up, SIGMA_TILT_HEAD, SIGMA_YAW_HEAD, HUBER_KNEE,
	    COST_WEIGHT);
	const double cost_flip = pose_metrics_prior_orient_cost(
	    &q_flip, &q_last_head_rel, &imu_up, SIGMA_TILT_HEAD, SIGMA_YAW_HEAD, HUBER_KNEE, COST_WEIGHT);
	INFO("Fix A tolerance: legit 30° head-rel cost=" << cost_legit << "  flip cost=" << cost_flip);
	// 30°/60° = 0.5 sigma -> d²=0.25 — small, well within the gate.
	CHECK(cost_legit < 0.3);
	CHECK(cost_flip > cost_legit * 10.0);
}

TEST_CASE("soft flip cost: the Huber knee bounds a gross outlier to a linear penalty")
{
	const struct xrt_vec3 up = {0.f, 1.f, 0.f};
	const struct xrt_quat q_prior = quat_axis_deg(0, 1, 0, 0);
	const double sigma_yaw = 15.0 * M_PI / 180.0;

	// Two grossly-flipped candidates differing by an extra 30 deg of tilt: beyond the knee the penalty is
	// LINEAR in the standardized residual, so the larger one is more expensive but only linearly so (a
	// quadratic loss would blow up). Compare to the unbounded quadratic at the same distances.
	struct xrt_quat q_a = quat_axis_deg(1, 0, 0, 120);
	struct xrt_quat q_b = quat_axis_deg(1, 0, 0, 150);
	const double cost_a = pose_metrics_prior_orient_cost(&q_a, &q_prior, &up, SIGMA_TILT, sigma_yaw,
	                                                    HUBER_KNEE, COST_WEIGHT);
	const double cost_b = pose_metrics_prior_orient_cost(&q_b, &q_prior, &up, SIGMA_TILT, sigma_yaw,
	                                                    HUBER_KNEE, COST_WEIGHT);
	// s_a = 120/30 = 4, s_b = 150/30 = 5 (both beyond knee 3). Huber: k(2s-k) -> 3(8-3)=15, 3(10-3)=21.
	INFO("cost_a=" << cost_a << "  cost_b=" << cost_b);
	CHECK(cost_a == Catch::Approx(15.0).margin(0.5));
	CHECK(cost_b == Catch::Approx(21.0).margin(0.5));
	// The unbounded quadratic would be 16 and 25; Huber keeps the gross-outlier penalty strictly below it.
	CHECK(cost_b < 5.0 * 5.0 * COST_WEIGHT);
}

// ===========================================================================
// MULTI-CAMERA JOINT (generalised / non-central) PnP (joint_pnp_solve): pooling LED bearing rays
// from 2+ rigidly-mounted cameras through their extrinsics dissolves the single-camera front/back
// mirror two-fold BY CONSTRUCTION (the flip cannot fit both cameras). Decoupled synthetic geometry
// with KNOWN ground truth: a wrong solve fails rather than agreeing with itself. Asserts the core
// properties the front-end relies on:
//   - a two-camera solve recovers the TRUE pose (orientation + position) tightly;
//   - the mirror twin that a SINGLE near-planar camera admits is NOT a fixed point of the joint solve
//     (seeded AT the flip, the joint solve over two cameras pulls back to the truth);
//   - a single contributing camera (and too-few rays) DECLINES, so the caller keeps its single-cam path.
// ===========================================================================

namespace {

//! A camera placed in the rig (IMU) frame: extrinsic P_imu_cam + a pinhole model.
struct JCam
{
	struct xrt_pose P_imu_cam;
	struct camera_model cam;
};

//! Project an LED object point through P_imu_obj (object->rig) into camera @c, returning the blob pixel
//! (pinhole, no distortion) and whether it lands in front of the camera. The constellation convention is
//! OpenCV camera frame (+Z forward), and joint_pnp maps the undistorted bearing (nx,ny,1) through
//! P_imu_cam.orientation — so the camera frame here is the one whose +Z is P_imu_cam's local forward.
bool
project_into(const JCam &jc, const struct xrt_pose &P_imu_obj, const struct xrt_vec3 &led_obj,
             struct xrt_vec2 &px)
{
	// LED in rig frame, then into the camera frame: p_cam = R_cam_imu (p_imu - t_cam).
	struct xrt_vec3 p_imu;
	math_pose_transform_point(&P_imu_obj, &led_obj, &p_imu);
	struct xrt_pose P_cam_imu;
	math_pose_invert(&jc.P_imu_cam, &P_cam_imu);
	struct xrt_vec3 p_cam;
	math_pose_transform_point(&P_cam_imu, &p_imu, &p_cam);
	if (p_cam.z <= 1e-3f) {
		return false;
	}
	px.x = (float)(jc.cam.calib.fx * (p_cam.x / p_cam.z) + jc.cam.calib.cx);
	px.y = (float)(jc.cam.calib.fy * (p_cam.y / p_cam.z) + jc.cam.calib.cy);
	return true;
}

JCam
make_jcam(const struct xrt_pose &P_imu_cam)
{
	JCam jc{};
	jc.P_imu_cam = P_imu_cam;
	jc.cam.width = 640;
	jc.cam.height = 480;
	jc.cam.calib.fx = jc.cam.calib.fy = (float)FX;
	jc.cam.calib.cx = jc.cam.calib.cy = (float)CX;
	jc.cam.calib.model = T_DISTORTION_OPENCV_RADTAN_8;
	return jc;
}

//! A FLAT planar LED patch (the mirror-prone geometry that admits the front/back two-fold), object
//! frame, metres. Perfectly coplanar (z=0) so a single view is genuinely two-fold ambiguous.
std::vector<struct xrt_vec3>
planar_leds()
{
	return {{-0.04f, -0.03f, 0.f}, {0.04f, -0.03f, 0.f}, {0.04f, 0.03f, 0.f}, {-0.04f, 0.03f, 0.f},
	        {0.f, 0.f, 0.f},       {0.02f, -0.01f, 0.f}, {-0.02f, 0.01f, 0.f}, {0.f, 0.025f, 0.f}};
}

//! Build a labelled blob set for one camera viewing @p P_imu_obj; only LEDs that land in-front + in-frame
//! are emitted. Returns the blobs (each labelled with its true LED id for MODEL_ID).
std::vector<blob>
blobs_for_cam(const JCam &jc, const struct xrt_pose &P_imu_obj, const std::vector<struct xrt_vec3> &leds)
{
	std::vector<blob> out;
	for (size_t i = 0; i < leds.size(); i++) {
		struct xrt_vec2 px;
		if (!project_into(jc, P_imu_obj, leds[i], px)) {
			continue;
		}
		blob b{};
		b.x = px.x;
		b.y = px.y;
		b.pos_var_px2 = 1.0f; // a real blob always carries its measurement variance (>= 0.25 px^2 floor)
		b.led_id = LED_MAKE_ID(MODEL_ID, (uint16_t)i);
		out.push_back(b);
	}
	return out;
}

t_constellation_led_model
led_model_from(std::vector<t_constellation_led> &storage, const std::vector<struct xrt_vec3> &leds)
{
	storage.resize(leds.size());
	for (size_t i = 0; i < leds.size(); i++) {
		storage[i] = t_constellation_led{};
		storage[i].id = (uint8_t)i;
		storage[i].pos = leds[i];
	}
	t_constellation_led_model m{};
	m.id = MODEL_ID;
	m.leds = storage.data();
	m.num_leds = (uint8_t)storage.size();
	return m;
}

//! Two cameras ~10 cm apart on the rig (the G2 front-stereo baseline), both looking +Z.
void
two_cams(JCam &left, JCam &right)
{
	struct xrt_pose Pl = XRT_POSE_IDENTITY;
	Pl.position = {-0.05f, 0.f, 0.f};
	struct xrt_pose Pr = XRT_POSE_IDENTITY;
	Pr.position = {0.05f, 0.f, 0.f};
	left = make_jcam(Pl);
	right = make_jcam(Pr);
}

} // namespace

TEST_CASE("joint PnP: two cameras recover the true pose tightly")
{
	const auto leds = planar_leds();
	std::vector<t_constellation_led> storage;
	t_constellation_led_model model = led_model_from(storage, leds);

	JCam left, right;
	two_cams(left, right);

	// Truth: controller ~45 cm in front of the rig, tilted off-axis (the ambiguous attitude).
	struct xrt_pose P_imu_obj_true = XRT_POSE_IDENTITY;
	P_imu_obj_true.position = {0.01f, -0.02f, 0.45f};
	struct xrt_quat tilt = quat_axis_deg(1, 0.4, 0.2, 28);
	P_imu_obj_true.orientation = tilt;

	std::vector<blob> bl = blobs_for_cam(left, P_imu_obj_true, leds);
	std::vector<blob> br = blobs_for_cam(right, P_imu_obj_true, leds);
	REQUIRE(bl.size() >= 3);
	REQUIRE(br.size() >= 3);

	struct joint_pnp_view views[2];
	views[0] = {bl.data(), (int)bl.size(), &left.cam, left.P_imu_cam};
	views[1] = {br.data(), (int)br.size(), &right.cam, right.P_imu_cam};

	// Seed with a slightly perturbed prior (a real fusion prior is never exact).
	struct xrt_pose pose = P_imu_obj_true;
	struct xrt_quat perturb = quat_axis_deg(0, 1, 0.3, 10);
	math_quat_rotate(&perturb, &pose.orientation, &pose.orientation);
	pose.position.z += 0.03f;

	int nrays = 0, ninl = 0;
	const bool ok = joint_pnp_solve(&pose, views, 2, &model, &nrays, &ninl);
	REQUIRE(ok);
	INFO("recovered q vs truth = " << quat_angle_deg(pose.orientation, P_imu_obj_true.orientation) << " deg");
	CHECK(quat_angle_deg(pose.orientation, P_imu_obj_true.orientation) < 3.0);
	CHECK(std::fabs(pose.position.x - P_imu_obj_true.position.x) < 0.01f);
	CHECK(std::fabs(pose.position.y - P_imu_obj_true.position.y) < 0.01f);
	CHECK(std::fabs(pose.position.z - P_imu_obj_true.position.z) < 0.01f);
}

TEST_CASE("joint PnP: the single-camera mirror twin is dissolved by the second camera")
{
	const auto leds = planar_leds();
	std::vector<t_constellation_led> storage;
	t_constellation_led_model model = led_model_from(storage, leds);

	JCam left, right;
	two_cams(left, right);

	struct xrt_pose P_imu_obj_true = XRT_POSE_IDENTITY;
	P_imu_obj_true.position = {0.0f, 0.0f, 0.45f};
	P_imu_obj_true.orientation = quat_axis_deg(1, 0.3, 0.0, 30); // tilted -> the single-view planar mirror exists

	std::vector<blob> bl = blobs_for_cam(left, P_imu_obj_true, leds);
	std::vector<blob> br = blobs_for_cam(right, P_imu_obj_true, leds);
	REQUIRE(bl.size() >= 4);
	REQUIRE(br.size() >= 4);

	// First show the SINGLE LEFT camera is genuinely flip-ambiguous: its lone solve admits a mirror twin
	// (the front/back flip a planar patch always has from one view) that reprojects ~as well as the truth.
	// That is exactly the ambiguity the joint solve must remove.
	struct camera_model lcam = left.cam;
	struct xrt_pose lpose = {};
	lpose.orientation.w = 1.f;
	struct xrt_pose ltwin = {};
	bool has_twin = false;
	int n = 0, ninl0 = 0;
	REQUIRE(ransac_pnp_pose_with_twin(&lpose, bl.data(), (int)bl.size(), &model, &lcam, &n, &ninl0, &ltwin,
	                                  &has_twin));
	REQUIRE(has_twin); // the single view really is two-fold ambiguous here

	struct joint_pnp_view views[2];
	views[0] = {bl.data(), (int)bl.size(), &left.cam, left.P_imu_cam};
	views[1] = {br.data(), (int)br.size(), &right.cam, right.P_imu_cam};

	// Seed the joint solve from a realistic (gravity-anchored) prior: the correct branch but perturbed by
	// ~12 deg + 3 cm, as a live fusion prior is. The two-camera solve must converge to the truth AND, by
	// construction, never land on the single-view flip (which fits the left image but not the right).
	struct xrt_pose pose = P_imu_obj_true;
	struct xrt_quat perturb = quat_axis_deg(0.2, 1.0, 0.3, 12);
	math_quat_rotate(&perturb, &pose.orientation, &pose.orientation);
	pose.position.z += 0.03f;

	int nrays = 0, ninl = 0;
	const bool ok = joint_pnp_solve(&pose, views, 2, &model, &nrays, &ninl);
	REQUIRE(ok);
	const double out_err = quat_angle_deg(pose.orientation, P_imu_obj_true.orientation);
	const struct xrt_quat ltwin_imu_q = [&] {
		struct xrt_pose tw;
		math_pose_transform(&left.P_imu_cam, &ltwin, &tw);
		return tw.orientation;
	}();
	const double err_to_flip = quat_angle_deg(pose.orientation, ltwin_imu_q);
	INFO("joint solved err-to-truth = " << out_err << " deg, err-to-single-cam-flip = " << err_to_flip << " deg");
	// Landed on the TRUTH, far from the single-camera flip -> the second camera dissolved the mirror.
	CHECK(out_err < 4.0);
	CHECK(err_to_flip > 15.0);
}

TEST_CASE("joint PnP: a FLIPPED prior is recovered (dual-seed crosses the front/back barrier)")
{
	// The worst case the dual-seed exists for: the fusion PRIOR itself is on the wrong branch (a flip has
	// propagated). A naive single-seed LM would stay at the flip; the two-camera solve must still recover
	// the truth, because the mirror seed fits both cameras and the (flipped) prior seed does not.
	const auto leds = planar_leds();
	std::vector<t_constellation_led> storage;
	t_constellation_led_model model = led_model_from(storage, leds);

	JCam left, right;
	two_cams(left, right);

	struct xrt_pose P_imu_obj_true = XRT_POSE_IDENTITY;
	P_imu_obj_true.position = {0.0f, 0.0f, 0.45f};
	P_imu_obj_true.orientation = quat_axis_deg(1, 0.3, 0.0, 30);

	std::vector<blob> bl = blobs_for_cam(left, P_imu_obj_true, leds);
	std::vector<blob> br = blobs_for_cam(right, P_imu_obj_true, leds);
	REQUIRE(bl.size() >= 4);
	REQUIRE(br.size() >= 4);

	// Recover the real single-camera mirror twin of the LEFT view and seed the joint solve AT it (the
	// flipped-prior case). The twin is genuinely a wrong branch (>15 deg from the truth).
	struct camera_model lcam = left.cam;
	struct xrt_pose lpose = {};
	lpose.orientation.w = 1.f;
	struct xrt_pose ltwin = {};
	bool has_twin = false;
	int n = 0, ninl0 = 0;
	REQUIRE(ransac_pnp_pose_with_twin(&lpose, bl.data(), (int)bl.size(), &model, &lcam, &n, &ninl0, &ltwin,
	                                  &has_twin));
	REQUIRE(has_twin);
	struct xrt_pose pose; // the flipped prior, in the rig frame
	math_pose_transform(&left.P_imu_cam, &ltwin, &pose);
	const double seed_err = quat_angle_deg(pose.orientation, P_imu_obj_true.orientation);
	INFO("flipped-prior seed error = " << seed_err << " deg");
	REQUIRE(seed_err > 15.0);

	struct joint_pnp_view views[2] = {{bl.data(), (int)bl.size(), &left.cam, left.P_imu_cam},
	                                  {br.data(), (int)br.size(), &right.cam, right.P_imu_cam}};
	int nrays = 0, ninl = 0;
	REQUIRE(joint_pnp_solve(&pose, views, 2, &model, &nrays, &ninl));
	const double out_err = quat_angle_deg(pose.orientation, P_imu_obj_true.orientation);
	INFO("recovered-from-flipped-prior error = " << out_err << " deg");
	CHECK(out_err < 4.0); // the dual-seed crossed back to the truth
}

TEST_CASE("joint PnP: a single contributing camera declines (caller keeps the single-cam path)")
{
	const auto leds = planar_leds();
	std::vector<t_constellation_led> storage;
	t_constellation_led_model model = led_model_from(storage, leds);

	JCam left, right;
	two_cams(left, right);

	struct xrt_pose P_imu_obj_true = XRT_POSE_IDENTITY;
	P_imu_obj_true.position = {0.0f, 0.0f, 0.45f};
	P_imu_obj_true.orientation = quat_axis_deg(1, 0.2, 0.1, 25);

	std::vector<blob> bl = blobs_for_cam(left, P_imu_obj_true, leds);
	REQUIRE(bl.size() >= 4);
	std::vector<blob> empty; // the right camera sees nothing

	struct joint_pnp_view views[2];
	views[0] = {bl.data(), (int)bl.size(), &left.cam, left.P_imu_cam};
	views[1] = {empty.data(), 0, &right.cam, right.P_imu_cam};

	struct xrt_pose pose = P_imu_obj_true;
	int nrays = 0, ninl = 0;
	// Only one camera contributes -> the joint solve must decline (the mirror is NOT resolvable here),
	// regardless of how many rays that one camera has.
	CHECK_FALSE(joint_pnp_solve(&pose, views, 2, &model, &nrays, &ninl));

	// And num_views < 2 is rejected outright.
	CHECK_FALSE(joint_pnp_solve(&pose, views, 1, &model, &nrays, &ninl));
}

TEST_CASE("joint PnP: degenerate / empty inputs are rejected, not crashed")
{
	const auto leds = planar_leds();
	std::vector<t_constellation_led> storage;
	t_constellation_led_model model = led_model_from(storage, leds);
	JCam left, right;
	two_cams(left, right);
	struct xrt_pose pose = XRT_POSE_IDENTITY;
	pose.position.z = 0.45f;
	int nrays = -1, ninl = -1;

	SECTION("null args")
	{
		CHECK_FALSE(joint_pnp_solve(nullptr, nullptr, 2, &model, &nrays, &ninl));
	}
	SECTION("two views but each below the 3-ray floor")
	{
		struct xrt_pose P = XRT_POSE_IDENTITY;
		P.position.z = 0.45f;
		std::vector<blob> bl = blobs_for_cam(left, P, leds);
		std::vector<blob> br = blobs_for_cam(right, P, leds);
		bl.resize(2); // starve both views below 3 rays
		br.resize(2);
		struct joint_pnp_view views[2] = {{bl.data(), 2, &left.cam, left.P_imu_cam},
		                                  {br.data(), 2, &right.cam, right.P_imu_cam}};
		CHECK_FALSE(joint_pnp_solve(&pose, views, 2, &model, &nrays, &ninl));
	}
}

// ===========================================================================
// PER-BLOB MEASUREMENT NOISE (pos_var_px2) WIRING into the global blob<->LED assignment. The assignment is
// ranked by the per-blob measurement-noise-weighted (Mahalanobis) cost (dx^2+dy^2)/pos_var_px2 — the ML
// data-association cost — so a residual is judged relative to how uncertain that blob's centre is. Decoupled
// behavioural test through the PUBLIC matcher (pose_metrics_match_pose_to_blobs): two blobs compete for one
// LED; raw pixel distance alone picks the closer blob, but the weighting picks the one the LED is more
// statistically consistent with. The test has teeth: raw distance and the weighted cost choose DIFFERENT
// blobs, so a no-op (unwired) matcher would assign the other one and fail.
// ===========================================================================

TEST_CASE("pos_var_px2: the blob<->LED assignment is ranked by the measurement-noise-weighted cost")
{
	// One visible LED, single pinhole camera. dir faces the camera (-Z) so the LED passes the visibility
	// test; a real LED radius gives a non-degenerate gate.
	std::vector<t_constellation_led> leds(1);
	leds[0] = t_constellation_led{};
	leds[0].id = 0;
	leds[0].pos = {0.f, 0.f, 0.f};
	leds[0].dir = {0.f, 0.f, -1.f};
	leds[0].radius_mm = 4.0f;
	t_constellation_led_model model{};
	model.id = MODEL_ID;
	model.leds = leds.data();
	model.num_leds = 1;

	struct camera_model cam = make_pinhole();
	struct xrt_pose pose = XRT_POSE_IDENTITY;
	pose.position = {0.f, 0.f, 0.5f}; // 0.5 m in front -> projects to the principal point

	// Locate the LED's projected pixel + gate by running the matcher with no blobs.
	struct pose_metrics_blob_match_info probe{};
	pose_metrics_match_pose_to_blobs(&pose, nullptr, 0, &model, &cam, &probe);
	REQUIRE(probe.num_visible_leds == 1);
	const struct xrt_vec2 led_px = probe.visible_leds[0].pos_px;
	const double gate = probe.visible_leds[0].led_radius_px;
	REQUIRE(gate > 2.5); // both candidate blobs must sit inside the gate

	// blob 0: FAR (2.0 px) but very uncertain  -> Maha 4.0/16 = 0.25  (most consistent with its own noise)
	// blob 1: NEAR (1.0 px) but a tight centre -> Maha 1.0/0.5 = 2.0
	// raw distance picks blob 1 (1.0 < 4.0); the measurement-noise-weighted cost picks blob 0 (0.25 < 2.0).
	struct blob blobs[2];
	blobs[0] = blob{};
	blobs[0].x = led_px.x + 2.0f;
	blobs[0].y = led_px.y;
	blobs[0].pos_var_px2 = 16.0f;
	blobs[0].led_id = LED_INVALID_ID;
	blobs[1] = blob{};
	blobs[1].x = led_px.x + 1.0f;
	blobs[1].y = led_px.y;
	blobs[1].pos_var_px2 = 0.5f;
	blobs[1].led_id = LED_INVALID_ID;
	REQUIRE(2.0f < (float)gate); // blob 0 (the farther one) is genuinely inside the gate

	struct pose_metrics_blob_match_info mi{};
	pose_metrics_match_pose_to_blobs(&pose, blobs, 2, &model, &cam, &mi);
	REQUIRE(mi.num_visible_leds == 1);
	REQUIRE(mi.visible_leds[0].matched_blob != nullptr);
	// The weighted cost flipped the choice: the LED is assigned to blob 0 (far + uncertain), NOT the
	// raw-closest blob 1. Unwired (raw dx^2+dy^2) this would be blob 1 and the check fails.
	CHECK(mi.visible_leds[0].matched_blob == &blobs[0]);

	// And the accumulated reprojection_error stays in RAW px^2 (so the GOOD/STRONG thresholds keep their
	// calibration): it equals blob 0's raw squared distance (2.0^2 = 4.0), not the weighted cost.
	CHECK(mi.reprojection_error == Catch::Approx(4.0).margin(1e-3));
}

// ===========================================================================
// AB-INITIO EARLY PRUNE (correspondence_search admissible-bound prior prune): when a confident prior is
// trusted, a candidate whose cheap prior cost alone (an admissible lower bound on its combined cost, since
// reprojection >= 0) already loses every score_is_better branch against the current GOOD best is skipped
// before the expensive reprojection. The committed invariant is accept-IDENTICAL: the prune only reduces
// work, never changes which pose wins. Driven through the REAL engine on the near-coplanar tilt-flip scene
// (the geometry the controller flip lives in), so the property is tested end-to-end, not on a stub.
// ===========================================================================
TEST_CASE("ab-initio prune: skips flipped hypotheses without changing the accepted pose")
{
	// The near-coplanar tilt-flip scene (same geometry as the twin-enumeration tests): both the true pose
	// and its mirror twin reproject onto the blobs, so the prior — not reprojection — must decide.
	const std::vector<cv::Point3f> obj = {
	    {-0.03f, -0.02f, 0.f}, {0.03f, -0.02f, 0.f}, {0.03f, 0.02f, 0.f}, {-0.03f, 0.02f, 0.f}, {0.f, 0.f, 0.f}};
	const cv::Mat rvec_true = (cv::Mat_<double>(3, 1) << 0.5, 0.25, 0.1);
	const cv::Mat tvec_true = (cv::Mat_<double>(3, 1) << 0.02, -0.01, 0.45);

	std::vector<t_constellation_led> leds;
	t_constellation_led_model model;
	std::vector<blob> blobs;
	build_scene(obj, rvec_true, tvec_true, leds, model, blobs);
	struct camera_model cam = make_pinhole();

	// Each LED's object-frame normal points from the LED toward the camera, so every LED passes the engine's
	// facing test at the true (camera-facing) pose. Camera centre in the object frame = -R^T t.
	const struct xrt_quat q_true = quat_from_rvec(rvec_true);
	struct xrt_pose P_cam_obj_true = {q_true,
	                                  {(float)tvec_true.at<double>(0), (float)tvec_true.at<double>(1),
	                                   (float)tvec_true.at<double>(2)}};
	struct xrt_pose P_obj_cam_true;
	math_pose_invert(&P_cam_obj_true, &P_obj_cam_true);
	const struct xrt_vec3 cam_centre_obj = P_obj_cam_true.position;
	for (size_t i = 0; i < leds.size(); i++) {
		struct xrt_vec3 n = m_vec3_sub(cam_centre_obj, leds[i].pos);
		math_vec3_normalize(&n);
		leds[i].dir = n;
		leds[i].radius_mm = 5.0f;
	}
	// A real blob carries its measurement variance + size; the engine's gate + ranked assignment read them.
	for (auto &b : blobs) {
		b.pos_var_px2 = 1.0f;
		b.width = 3;
		b.height = 3;
	}

	struct t_constellation_search_model *smodel = t_constellation_search_model_new(&model);
	REQUIRE(smodel != nullptr);

	struct correspondence_search *cs = correspondence_search_new(&cam);
	REQUIRE(cs != nullptr);
	correspondence_search_set_blobs(cs, blobs.data(), (int)blobs.size());

	// The production soft-cost knobs (mirror t_constellation_tracking.c; the matcher is driven through the
	// real engine with the real policy values, not a re-implementation).
	const float SIGMA_TILT = 30.0f * (float)(M_PI / 180.0); // GRAVITY_TILT_TOL == MIN_ROT_ERROR
	const float SIGMA_YAW = 15.0f * (float)(M_PI / 180.0);  // a confident live yaw
	const float HUBER_KNEE = 3.0f;                          // PRIOR_GATE_SIGMA
	const float COST_WEIGHT = 1.0f;                         // FLIP_COST_WEIGHT
	const float POS_TOL = 0.60f;                            // MAX_POS_ERROR
	const float ROT_TOL = 60.0f * (float)(M_PI / 180.0);   // MAX_ROT_ERROR

	// PRIOR = the true camera->object pose, with a CONFIDENT (tight) yaw sigma: the prune bound is then
	// tight enough to skip the flipped twin once the true pose is the running best.
	struct xrt_pose prior = {};
	prior.orientation = q_true;
	prior.position = {(float)tvec_true.at<double>(0), (float)tvec_true.at<double>(1),
	                  (float)tvec_true.at<double>(2)};
	struct xrt_vec3 pos_thresh = {POS_TOL, POS_TOL, POS_TOL};
	struct xrt_vec3 rot_thresh = {ROT_TOL, ROT_TOL, ROT_TOL};
	struct xrt_vec3 up = {0.f, 1.f, 0.f};

	// SHALLOW search (depth 1, the small-constellation regime this 5-LED scene fits) and NO
	// STOP_FOR_STRONG_MATCH: let the search run past the first GOOD pose so the flipped candidates that
	// follow are the ones the prune skips (with the strong-stop the search would terminate before them).
	enum correspondence_search_flags flags = (enum correspondence_search_flags)(
	    CS_FLAG_SHALLOW_SEARCH | CS_FLAG_HAVE_POSE_PRIOR | CS_FLAG_TRUST_PRIOR_ORIENT);

	struct xrt_pose found = prior; // seeds mi.pose_prior
	struct pose_metrics score = {};
	const bool ok = correspondence_search_find_one_pose(cs, smodel, flags, &found, &pos_thresh, &rot_thresh,
	                                                    &up, SIGMA_TILT, SIGMA_YAW, HUBER_KNEE, COST_WEIGHT,
	                                                    &score);
	REQUIRE(ok);
	REQUIRE((score.match_flags & POSE_MATCH_GOOD) != 0);

	// 1) The accepted pose is the TRUE twin, not the mirror flip.
	CHECK(quat_angle_deg(found.orientation, q_true) < 8.0);

	// 2) The prune actually fired on this search (else the test proves nothing about it).
	CHECK(cs->num_pose_checks_pruned > 0u);

	// 3) Accept-invariance witness: independently score the true pose and its mirror twin through the SAME
	//    public scoring + comparator the engine uses; the accepted pose must be the lower-combined-cost GOOD
	//    twin — i.e. the prune did NOT drop the rightful winner. Recover the twin via the direct PnP.
	struct xrt_pose primary = {}, twin = {};
	primary.orientation.w = 1.f;
	bool has_twin = false;
	int nleds = 0, ninliers = 0;
	REQUIRE(ransac_pnp_pose_with_twin(&primary, blobs.data(), (int)blobs.size(), &model, &cam, &nleds,
	                                  &ninliers, &twin, &has_twin));
	REQUIRE(has_twin);
	auto combined_cost = [&](const struct xrt_pose &p) {
		struct pose_metrics s = {};
		struct xrt_pose pp = p;
		pose_metrics_evaluate_pose_with_prior(&s, &pp, false, &prior, &pos_thresh, &rot_thresh, blobs.data(),
		                                      (int)blobs.size(), &model, &cam, nullptr);
		const double pc = pose_metrics_prior_orient_cost(&p.orientation, &prior.orientation, &up, SIGMA_TILT,
		                                                 SIGMA_YAW, HUBER_KNEE, COST_WEIGHT);
		return std::make_pair(s, s.reprojection_error + pc);
	};
	auto [s_primary, c_primary] = combined_cost(primary);
	auto [s_twin, c_twin] = combined_cost(twin);
	const bool primary_good = (s_primary.match_flags & POSE_MATCH_GOOD) != 0;
	const bool twin_good = (s_twin.match_flags & POSE_MATCH_GOOD) != 0;
	REQUIRE((primary_good || twin_good));
	// The expected winner = the lower-combined-cost GOOD twin (the engine's policy), and the engine's
	// accepted orientation must match it. This is the load-bearing accept-invariance check.
	const struct xrt_pose &expected =
	    (twin_good && (!primary_good || c_twin < c_primary)) ? twin : primary;
	CHECK(quat_angle_deg(found.orientation, expected.orientation) < 1.0);

	// 4) Cold-start invariance: WITHOUT CS_FLAG_TRUST_PRIOR_ORIENT (no trusted prior, the ab-initio
	//    bootstrap), the prune is inert — prior_cost is 0, so its branch-4 guard can never hold and NO
	//    candidate is skipped. The full enumeration runs at cold start.
	struct correspondence_search *cs_cold = correspondence_search_new(&cam);
	correspondence_search_set_blobs(cs_cold, blobs.data(), (int)blobs.size());
	enum correspondence_search_flags cold_flags =
	    (enum correspondence_search_flags)(CS_FLAG_SHALLOW_SEARCH | CS_FLAG_HAVE_POSE_PRIOR);
	struct xrt_pose found_cold = prior;
	struct pose_metrics score_cold = {};
	correspondence_search_find_one_pose(cs_cold, smodel, cold_flags, &found_cold, &pos_thresh, &rot_thresh, &up,
	                                    SIGMA_TILT, SIGMA_YAW, HUBER_KNEE, COST_WEIGHT, &score_cold);
	CHECK(cs_cold->num_pose_checks_pruned == 0u);

	correspondence_search_free(cs_cold);
	correspondence_search_free(cs);
	t_constellation_search_model_free(smodel);
}
