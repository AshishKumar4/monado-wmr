/*!
 * @file
 * @brief Decoupled unit tests for per-LED multi-camera triangulation (multicam_triangulate_position).
 *        Synthetic geometry with KNOWN ground truth — no filter, no captured data — so a wrong solve
 *        fails rather than agreeing with itself. Validates the core recall primitive: a controller LED
 *        co-seen by >= 2 rigidly-mounted cameras triangulates to a 3D point from the rig baseline, and
 *        the per-LED points aggregate to the controller position WITHOUT a full constellation or any
 *        orientation solve — keeping the track alive on the sparse-LED frames a single-view PnP cannot
 *        solve. Also asserts the precision guards: it DECLINES (no false position) when only one camera
 *        sees the LEDs, and rejects an inconsistent cross-camera pairing.
 */
#include "catch_amalgamated.hpp"

#include "internal/multicam_triangulate.h"
#include "internal/l1_depth_verdict.h"
#include "internal/blobwatch.h"
#include "internal/camera_model.h"
#include "tracking/t_led_models.h"
#include "math/m_api.h"

#include <cmath>
#include <vector>

namespace {

constexpr int MODEL_ID = 2;
constexpr double FX = 400.0, CX = 320.0, CY = 240.0;

//! A zero-distortion pinhole camera (RADTAN_8 with all-zero coeffs == pinhole).
struct camera_model
make_pinhole()
{
	struct camera_model cam = {};
	cam.width = 640;
	cam.height = 480;
	cam.calib.fx = cam.calib.fy = (float)FX;
	cam.calib.cx = (float)CX;
	cam.calib.cy = (float)CY;
	cam.calib.model = T_DISTORTION_OPENCV_RADTAN_8;
	return cam;
}

//! A camera at world position @p pos looking down +Z (identity-ish), with a small yaw so the two test
//! cameras observe the target from genuinely different bearings (a real triangulation baseline). The pose
//! is camera->world (P_world_cam): orientation maps a camera-frame vector into world.
struct xrt_pose
make_cam_pose(const struct xrt_vec3 &pos, float yaw_rad)
{
	struct xrt_pose p = {};
	const struct xrt_vec3 up = {0.f, 1.f, 0.f};
	math_quat_from_angle_vector(yaw_rad, &up, &p.orientation);
	p.position = pos;
	return p;
}

//! Project a world point into a camera's pixels (pinhole). Returns false if behind the camera.
bool
project_world_point(const struct xrt_pose &P_world_cam, const struct xrt_vec3 &world_pt, float &out_x, float &out_y)
{
	struct xrt_pose P_cam_world;
	math_pose_invert(&P_world_cam, &P_cam_world);
	struct xrt_vec3 p_cam;
	math_pose_transform_point(&P_cam_world, &world_pt, &p_cam);
	if (p_cam.z < 1e-4f) {
		return false;
	}
	out_x = (float)(FX * (p_cam.x / p_cam.z) + CX);
	out_y = (float)(FX * (p_cam.y / p_cam.z) + CY);
	return true;
}

//! Place an LED model (a few points around the origin) and a true controller pose, then synthesise the
//! per-view blobs by projecting each LED into each camera. Each blob is labelled (MODEL_ID, led_index).
struct Scene
{
	std::vector<t_constellation_led> leds;
	t_constellation_led_model model;
	std::vector<std::vector<blob>> blobs; // [view][blob]
};

Scene
build_scene(const std::vector<struct xrt_vec3> &led_obj,
            const struct xrt_pose &P_world_obj,
            const std::vector<struct xrt_pose> &cam_poses)
{
	Scene s;
	s.leds.resize(led_obj.size());
	for (size_t i = 0; i < led_obj.size(); i++) {
		s.leds[i] = t_constellation_led{};
		s.leds[i].id = (uint8_t)i;
		s.leds[i].pos = led_obj[i];
	}
	s.model = t_constellation_led_model{};
	s.model.id = MODEL_ID;
	s.model.leds = s.leds.data();
	s.model.num_leds = (uint8_t)s.leds.size();

	s.blobs.resize(cam_poses.size());
	for (size_t v = 0; v < cam_poses.size(); v++) {
		for (size_t i = 0; i < led_obj.size(); i++) {
			struct xrt_vec3 world_pt;
			math_pose_transform_point(&P_world_obj, &led_obj[i], &world_pt);
			float px, py;
			if (!project_world_point(cam_poses[v], world_pt, px, py)) {
				continue;
			}
			blob b = {};
			b.x = px;
			b.y = py;
			b.led_id = LED_MAKE_ID(MODEL_ID, (uint16_t)i);
			b.pos_var_px2 = 0.25f;
			s.blobs[v].push_back(b);
		}
	}
	return s;
}

std::vector<multicam_tri_view>
make_views(Scene &s, const std::vector<struct xrt_pose> &cam_poses, struct camera_model &cam)
{
	std::vector<multicam_tri_view> views(cam_poses.size());
	for (size_t v = 0; v < cam_poses.size(); v++) {
		views[v].blobs = s.blobs[v].data();
		views[v].num_blobs = (int)s.blobs[v].size();
		views[v].calib = &cam;
		views[v].P_world_cam = cam_poses[v];
	}
	return views;
}

} // namespace

TEST_CASE("two cameras triangulate a sparse-LED controller position from the rig baseline")
{
	// A small LED patch; the controller is at a known world pose. Two cameras ~30 cm apart with a yaw
	// toward the target — a genuine stereo baseline. Each camera sees only TWO LEDs (below any PnP floor).
	const std::vector<struct xrt_vec3> led_obj = {
	    {-0.04f, 0.0f, 0.0f}, {0.04f, 0.0f, 0.0f}, {0.0f, 0.03f, 0.0f}, {0.0f, -0.03f, 0.0f}};
	struct xrt_pose P_world_obj = {};
	const struct xrt_vec3 up_axis = {0.f, 1.f, 0.f};
	math_quat_from_angle_vector(0.4f, &up_axis, &P_world_obj.orientation); // a real (yaw) controller attitude
	P_world_obj.position = {0.10f, -0.05f, 1.20f};

	const std::vector<struct xrt_pose> cam_poses = {
	    make_cam_pose({-0.15f, 0.0f, 0.0f}, 0.12f),
	    make_cam_pose({0.15f, 0.0f, 0.0f}, -0.12f),
	};

	Scene s = build_scene(led_obj, P_world_obj, cam_poses);
	struct camera_model cam = make_pinhole();
	std::vector<multicam_tri_view> views = make_views(s, cam_poses, cam);

	struct multicam_tri_result res = {};
	const bool ok = multicam_triangulate_position(views.data(), (int)views.size(), &s.model,
	                                               &P_world_obj.orientation, 0.0f, 2, &res);
	REQUIRE(ok);
	REQUIRE(res.num_views >= 2);
	REQUIRE(res.num_leds >= 2);

	// The recovered controller origin must match the true position to within a few mm (the lever arm is
	// subtracted with the EXACT orientation here, so only the linear-solve numerics remain).
	const double dx = res.position.x - P_world_obj.position.x;
	const double dy = res.position.y - P_world_obj.position.y;
	const double dz = res.position.z - P_world_obj.position.z;
	const double err = std::sqrt(dx * dx + dy * dy + dz * dz);
	REQUIRE(err < 0.01); // < 1 cm
}

TEST_CASE("declines when only ONE camera sees the LEDs (no false position)")
{
	const std::vector<struct xrt_vec3> led_obj = {
	    {-0.04f, 0.0f, 0.0f}, {0.04f, 0.0f, 0.0f}, {0.0f, 0.03f, 0.0f}};
	struct xrt_pose P_world_obj = {};
	P_world_obj.orientation.w = 1.f;
	P_world_obj.position = {0.0f, 0.0f, 1.0f};

	// Two camera views, but the second is pointed away so it sees nothing — only ONE camera observes.
	const std::vector<struct xrt_pose> cam_poses = {
	    make_cam_pose({-0.15f, 0.0f, 0.0f}, 0.1f),
	    make_cam_pose({100.0f, 0.0f, 0.0f}, 3.1415f), // far away + looking the other way -> no blobs
	};

	Scene s = build_scene(led_obj, P_world_obj, cam_poses);
	struct camera_model cam = make_pinhole();
	std::vector<multicam_tri_view> views = make_views(s, cam_poses, cam);

	struct multicam_tri_result res = {};
	const bool ok = multicam_triangulate_position(views.data(), (int)views.size(), &s.model,
	                                               &P_world_obj.orientation, 0.0f, 2, &res);
	REQUIRE_FALSE(ok); // a single observing camera cannot triangulate depth -> must decline
}

TEST_CASE("rejects an inconsistent cross-camera correspondence (mislabelled LED)")
{
	const std::vector<struct xrt_vec3> led_obj = {
	    {-0.04f, 0.0f, 0.0f}, {0.04f, 0.0f, 0.0f}, {0.0f, 0.03f, 0.0f}, {0.0f, -0.03f, 0.0f}};
	struct xrt_pose P_world_obj = {};
	P_world_obj.orientation.w = 1.f;
	P_world_obj.position = {0.05f, 0.0f, 1.10f};

	const std::vector<struct xrt_pose> cam_poses = {
	    make_cam_pose({-0.15f, 0.0f, 0.0f}, 0.12f),
	    make_cam_pose({0.15f, 0.0f, 0.0f}, -0.12f),
	};

	Scene s = build_scene(led_obj, P_world_obj, cam_poses);

	// Corrupt the second camera's blob for LED 0: move it far from its true projection, so the two rays
	// for LED 0 do NOT intersect (a wrong cross-camera pairing). That LED must be rejected by the per-ray
	// residual gate; the OTHER consistent LEDs still triangulate, so the solve can still succeed on them.
	REQUIRE(s.blobs[1].size() >= 1);
	for (auto &b : s.blobs[1]) {
		if (LED_LOCAL_ID(b.led_id) == 0) {
			b.x += 80.0f; // ~80 px off -> a decimetre-scale ray miss at this range
			b.y -= 60.0f;
		}
	}
	struct camera_model cam = make_pinhole();
	std::vector<multicam_tri_view> views = make_views(s, cam_poses, cam);

	struct multicam_tri_result res = {};
	const bool ok = multicam_triangulate_position(views.data(), (int)views.size(), &s.model,
	                                               &P_world_obj.orientation, 0.0f, 2, &res);
	// The three consistent LEDs still triangulate; assert the corrupted LED 0 did NOT poison the result.
	if (ok) {
		const double dx = res.position.x - P_world_obj.position.x;
		const double dy = res.position.y - P_world_obj.position.y;
		const double dz = res.position.z - P_world_obj.position.z;
		REQUIRE(std::sqrt(dx * dx + dy * dy + dz * dz) < 0.02); // robust average rejected the outlier
	}
}

//! Strip every blob's front-end label (set led_id INVALID) — the epipolar path must recover the same
//! correspondences from geometry + the prior-pose model gate, exactly the FOV-edge case it targets.
void
unlabel_all(Scene &s)
{
	for (auto &view : s.blobs) {
		for (auto &b : view) {
			b.led_id = LED_INVALID_ID;
		}
	}
}

TEST_CASE("epipolar triangulation recovers the position from UNLABELLED blobs (FOV-edge case)")
{
	const std::vector<struct xrt_vec3> led_obj = {
	    {-0.04f, 0.0f, 0.0f}, {0.04f, 0.0f, 0.0f}, {0.0f, 0.03f, 0.0f}, {0.0f, -0.03f, 0.0f}};
	struct xrt_pose P_world_obj = {};
	const struct xrt_vec3 up_axis = {0.f, 1.f, 0.f};
	math_quat_from_angle_vector(0.4f, &up_axis, &P_world_obj.orientation);
	P_world_obj.position = {0.10f, -0.05f, 1.20f};

	const std::vector<struct xrt_pose> cam_poses = {
	    make_cam_pose({-0.15f, 0.0f, 0.0f}, 0.12f),
	    make_cam_pose({0.15f, 0.0f, 0.0f}, -0.12f),
	};

	Scene s = build_scene(led_obj, P_world_obj, cam_poses);
	unlabel_all(s); // the label path would find NOTHING here
	struct camera_model cam = make_pinhole();
	std::vector<multicam_tri_view> views = make_views(s, cam_poses, cam);

	// Label path: declines (no co-labelled LED). Epipolar path: recovers the position from geometry.
	struct multicam_tri_result lab = {};
	REQUIRE_FALSE(multicam_triangulate_position(views.data(), (int)views.size(), &s.model,
	                                            &P_world_obj.orientation, 0.0f, 2, &lab));

	struct multicam_tri_result res = {};
	const bool ok = multicam_triangulate_epipolar_position(views.data(), (int)views.size(), &s.model,
	                                                        &P_world_obj, 0.0f, 1.2f, 0.06f, 2, &res);
	REQUIRE(ok);
	REQUIRE(res.num_views >= 2);
	REQUIRE(res.num_leds >= 2);
	const double dx = res.position.x - P_world_obj.position.x;
	const double dy = res.position.y - P_world_obj.position.y;
	const double dz = res.position.z - P_world_obj.position.z;
	REQUIRE(std::sqrt(dx * dx + dy * dy + dz * dz) < 0.02);
}

TEST_CASE("epipolar triangulation declines on clutter that does not form the model (no false position)")
{
	const std::vector<struct xrt_vec3> led_obj = {
	    {-0.04f, 0.0f, 0.0f}, {0.04f, 0.0f, 0.0f}, {0.0f, 0.03f, 0.0f}, {0.0f, -0.03f, 0.0f}};
	struct xrt_pose P_world_obj = {};
	P_world_obj.orientation.w = 1.f;
	P_world_obj.position = {0.10f, -0.05f, 1.20f};

	const std::vector<struct xrt_pose> cam_poses = {
	    make_cam_pose({-0.15f, 0.0f, 0.0f}, 0.12f),
	    make_cam_pose({0.15f, 0.0f, 0.0f}, -0.12f),
	};

	// Build the scene from the REAL controller, then unlabel it AND move the controller's true pose far from
	// the prior we hand the epipolar solver: the blobs no longer form the rigid model at the PRIOR body pose,
	// so every cross-view intersection fails the model gate -> decline (the clutter-rejection guard).
	struct xrt_pose P_world_obj_true = P_world_obj;
	P_world_obj_true.position = {0.9f, 0.6f, 1.6f}; // controller actually elsewhere
	Scene s = build_scene(led_obj, P_world_obj_true, cam_poses);
	unlabel_all(s);
	struct camera_model cam = make_pinhole();
	std::vector<multicam_tri_view> views = make_views(s, cam_poses, cam);

	// Prior says the controller is at P_world_obj (0.10,-0.05,1.20); the blobs triangulate ~ (0.9,0.6,1.6),
	// > model_gate from every prior-projected model LED -> no accepted correspondence -> no false position.
	struct multicam_tri_result res = {};
	const bool ok = multicam_triangulate_epipolar_position(views.data(), (int)views.size(), &s.model,
	                                                        &P_world_obj, 0.0f, 1.2f, 0.06f, 2, &res);
	REQUIRE_FALSE(ok);
}

TEST_CASE("epipolar triangulation ignores the OTHER controller's blobs (cross-device rejection)")
{
	// Two controllers in view; the epipolar solve for THIS device (anchored at its own prior) must recover
	// only its own position and not be pulled by the second controller's blobs sharing the frames.
	const std::vector<struct xrt_vec3> led_obj = {
	    {-0.04f, 0.0f, 0.0f}, {0.04f, 0.0f, 0.0f}, {0.0f, 0.03f, 0.0f}, {0.0f, -0.03f, 0.0f}};
	struct xrt_pose P_a = {};
	P_a.orientation.w = 1.f;
	P_a.position = {0.10f, -0.05f, 1.20f}; // device A (the one we solve)
	struct xrt_pose P_b = {};
	P_b.orientation.w = 1.f;
	P_b.position = {0.45f, 0.10f, 1.35f};  // device B, a different controller elsewhere in the frame

	const std::vector<struct xrt_pose> cam_poses = {
	    make_cam_pose({-0.15f, 0.0f, 0.0f}, 0.12f),
	    make_cam_pose({0.15f, 0.0f, 0.0f}, -0.12f),
	};

	// Synthesize both controllers' blobs into the same views; unlabel everything (epipolar uses no labels).
	Scene sa = build_scene(led_obj, P_a, cam_poses);
	Scene sb = build_scene(led_obj, P_b, cam_poses);
	for (size_t v = 0; v < sa.blobs.size(); v++) {
		for (auto &b : sb.blobs[v]) {
			sa.blobs[v].push_back(b);
		}
	}
	unlabel_all(sa);
	struct camera_model cam = make_pinhole();
	std::vector<multicam_tri_view> views = make_views(sa, cam_poses, cam);

	// Solve for device A at its prior: the recovered position must be A's, not dragged toward B (~37 cm away).
	struct multicam_tri_result res = {};
	const bool ok = multicam_triangulate_epipolar_position(views.data(), (int)views.size(), &sa.model, &P_a,
	                                                        0.0f, 1.2f, 0.06f, 2, &res);
	REQUIRE(ok);
	const double dA = std::sqrt((res.position.x - P_a.position.x) * (res.position.x - P_a.position.x) +
	                            (res.position.y - P_a.position.y) * (res.position.y - P_a.position.y) +
	                            (res.position.z - P_a.position.z) * (res.position.z - P_a.position.z));
	REQUIRE(dA < 0.02); // locked onto device A, the second controller did not corrupt it
}

// ----------------------------------------------------------------------------------------------------
// L1 verdict-ladder invariants (design-L1-multicam-depth.md §2.6). These assert the AGREE/REFINE/DISPUTE
// decision + the recall-monotone construction on the pure l1_depth_verdict() core, at the SAME operating
// point the live tracker wires in (so the test exercises the production gate, not a divergent copy).
namespace {

//! The production L1 operating point (mirrors the ASSOC_L1_* constants in t_constellation_tracking.c).
const struct l1_depth_params kL1Params = {
    /* dispute_sigma */ 3.0,
    /* min_dispute_leds */ 2,
    /* refine_per_cam_leds */ 4,
    /* refine_max_std_m */ 0.04, // == ASSOC_POSITION_ONLY_MIN_STD_M
};

} // namespace

TEST_CASE("L1 no-2nd-view -> AGREE (recall-monotone: an absent 2nd view can never change the commit)")
{
	// The committing camera could be a metre off in depth, but NO other camera sees >=2 device LEDs.
	// The ladder MUST treat the single-view frame as untouchable — this is the dev2 single-cam-dwell guard.
	struct l1_depth_evidence e = {};
	e.disp_m = 1.0;               // a huge (hypothetical) depth disagreement
	e.triangulated_std_m = 0.01;  // and a (hypothetical) confident triangulation
	e.committed_std_m = 0.06;
	e.best_second_view_leds = 0;  // ... but NO 2nd view sees the device
	e.n_cams_ge_refine_leds = 1;  // only the committing camera
	e.tri_succeeded = true;       // even if some stray geometry "succeeded", the census gate fires FIRST
	REQUIRE(l1_depth_verdict(&e, &kL1Params) == L1V_AGREE);

	// And a 2nd view with a single LED (pure bearing, no depth constraint) is still below the dispute bar.
	e.best_second_view_leds = 1;
	REQUIRE(l1_depth_verdict(&e, &kL1Params) == L1V_AGREE);
}

TEST_CASE("L1 monotone invariant: forcing tri_succeeded=false ALWAYS yields AGREE")
{
	// The triangulation declining (near-parallel baseline, <2 co-seen LEDs, consensus collapse) is the
	// 'no usable 2-view geometry' case: the commit is left alone regardless of the 2nd-view LED census.
	for (int second_leds = 0; second_leds <= 6; second_leds++) {
		for (int refine_cams = 0; refine_cams <= 3; refine_cams++) {
			struct l1_depth_evidence e = {};
			e.disp_m = 0.8; // would be a gross dispute IF the geometry were usable
			e.triangulated_std_m = 0.0;
			e.committed_std_m = 0.06;
			e.best_second_view_leds = second_leds;
			e.n_cams_ge_refine_leds = refine_cams;
			e.tri_succeeded = false; // declined
			REQUIRE(l1_depth_verdict(&e, &kL1Params) == L1V_AGREE);
		}
	}
}

TEST_CASE("L1 confirming 2nd view -> AGREE (within the shared-covariance gate, the commit is untouched)")
{
	// A real 2nd view, a small depth difference well inside 3 sigma of the combined std -> the healthy
	// multi-cam frame, a no-op. Here gate = 3 * sqrt(0.02^2 + 0.06^2) ~= 0.19 m; disp 0.05 m is inside.
	struct l1_depth_evidence e = {};
	e.disp_m = 0.05;
	e.triangulated_std_m = 0.02;
	e.committed_std_m = 0.06;
	e.best_second_view_leds = 5;
	e.n_cams_ge_refine_leds = 2; // even a STRONG 2nd view that AGREES is left untouched (no needless refine)
	e.tri_succeeded = true;
	REQUIRE(l1_depth_verdict(&e, &kL1Params) == L1V_AGREE);
}

TEST_CASE("L1 strong disputing 2nd view -> REFINE (>=2 cams each >=4 LEDs, confident triangulation)")
{
	// Committed depth ~1 m off; two cameras each carry >=4 device LEDs and the triangulation is sub-cm.
	// gate = 3 * sqrt(0.01^2 + 0.06^2) ~= 0.18 m; disp 1.0 m is far outside -> a genuine dispute, strong.
	struct l1_depth_evidence e = {};
	e.disp_m = 1.0;
	e.triangulated_std_m = 0.01;
	e.committed_std_m = 0.06;
	e.best_second_view_leds = 4;
	e.n_cams_ge_refine_leds = 2;
	e.tri_succeeded = true;
	REQUIRE(l1_depth_verdict(&e, &kL1Params) == L1V_REFINE);
}

TEST_CASE("L1 weak disputing 2nd view -> DISPUTE (2-3 LEDs: reject the commit, do not re-solve)")
{
	// The depth dispute is real (disp 1.0 m, far outside the gate) but the 2nd camera carries only 3 LEDs,
	// so fewer than two cameras reach the >=4 REFINE bar -> reject only, never re-solve a weak position.
	struct l1_depth_evidence e = {};
	e.disp_m = 1.0;
	e.triangulated_std_m = 0.01;
	e.committed_std_m = 0.06;
	e.best_second_view_leds = 3;
	e.n_cams_ge_refine_leds = 1; // only the committing camera reaches the per-cam REFINE LED bar
	e.tri_succeeded = true;
	REQUIRE(l1_depth_verdict(&e, &kL1Params) == L1V_DISPUTE);
}

TEST_CASE("L1 disputing but unconfident triangulation -> DISPUTE not REFINE (std above the floor)")
{
	// Two strong cameras dispute the depth, but the triangulation's own 1 sigma is above the REFINE floor
	// (0.04 m): folding it would inject a position no better than the disputed commit -> reject, don't fold.
	struct l1_depth_evidence e = {};
	e.disp_m = 1.0;
	e.triangulated_std_m = 0.10; // > refine_max_std_m
	e.committed_std_m = 0.06;
	e.best_second_view_leds = 5;
	e.n_cams_ge_refine_leds = 2;
	e.tri_succeeded = true;
	// gate = 3 * sqrt(0.10^2 + 0.06^2) ~= 0.35 m; disp 1.0 m is still outside -> a dispute, but too soft.
	REQUIRE(l1_depth_verdict(&e, &kL1Params) == L1V_DISPUTE);
}

TEST_CASE("L1 monotone via real geometry: a single-observing-camera scene declines -> AGREE")
{
	// Tie the math to the ladder: only ONE camera observes the device (the 2nd is pointed away). The parked
	// triangulator declines (no baseline), and the verdict on that real decline is AGREE — the commit a
	// single camera produced is never disputed by a 2nd view that does not exist. (Mirrors the decline test.)
	const std::vector<struct xrt_vec3> led_obj = {
	    {-0.04f, 0.0f, 0.0f}, {0.04f, 0.0f, 0.0f}, {0.0f, 0.03f, 0.0f}};
	struct xrt_pose P_world_obj = {};
	P_world_obj.orientation.w = 1.f;
	P_world_obj.position = {0.0f, 0.0f, 1.0f};
	const std::vector<struct xrt_pose> cam_poses = {
	    make_cam_pose({-0.15f, 0.0f, 0.0f}, 0.1f),
	    make_cam_pose({100.0f, 0.0f, 0.0f}, 3.1415f),
	};
	Scene s = build_scene(led_obj, P_world_obj, cam_poses);
	struct camera_model cam = make_pinhole();
	std::vector<multicam_tri_view> views = make_views(s, cam_poses, cam);

	struct multicam_tri_result res = {};
	const bool tri_ok = multicam_triangulate_position(views.data(), (int)views.size(), &s.model,
	                                                   &P_world_obj.orientation, 0.0f, 2, &res);
	REQUIRE_FALSE(tri_ok); // single observing camera -> no triangulation

	// The committing camera sees 3 LEDs; the 2nd camera sees 0. Feed the real census + the real decline.
	struct l1_depth_evidence e = {};
	e.disp_m = m_vec3_len(m_vec3_sub(res.position, P_world_obj.position));
	e.triangulated_std_m = (double)res.position_std_m;
	e.committed_std_m = 0.06;
	e.best_second_view_leds = 0; // only one camera observes
	e.n_cams_ge_refine_leds = 1;
	e.tri_succeeded = tri_ok;
	REQUIRE(l1_depth_verdict(&e, &kL1Params) == L1V_AGREE);
}

TEST_CASE("L1 keystone: prior-seeded epipolar attach turns a 0-2nd-view AGREE into a real disputing census")
{
	// The detection-keystone correspondence fix. A wrong-depth single-cam commit cannot reproject-label its
	// own 2nd camera, so production leaves the 2nd cam's (recovered) blobs UNLABELLED -> the label-based
	// triangulation declines, the L1 census sees best_second_view_leds == 0, and the verdict is AGREE on
	// every frame (the literal 0 -> 0). This test reproduces that starved state and proves the fix: the
	// prior-seeded epipolar attach (seeded from the gravity/IMU PRIOR, not the wrong-depth commit) recovers
	// a 2nd-cam correspondence from the raw blobs, the re-census reaches the dispute bar, and L1 ENGAGES.
	const std::vector<struct xrt_vec3> led_obj = {
	    {-0.04f, 0.0f, 0.0f}, {0.04f, 0.0f, 0.0f}, {0.0f, 0.03f, 0.0f}, {0.0f, -0.03f, 0.0f}};
	struct xrt_pose P_world_obj = {};
	P_world_obj.orientation.w = 1.f;
	P_world_obj.position = {0.05f, -0.02f, 1.10f}; // TRUE controller pose (and the gravity/IMU prior pose)
	const std::vector<struct xrt_pose> cam_poses = {
	    make_cam_pose({-0.15f, 0.0f, 0.0f}, 0.12f),
	    make_cam_pose({0.15f, 0.0f, 0.0f}, -0.12f),
	};
	Scene s = build_scene(led_obj, P_world_obj, cam_poses);
	struct camera_model cam = make_pinhole();
	std::vector<multicam_tri_view> views = make_views(s, cam_poses, cam);

	// STARVED STATE: unlabel ALL blobs -> the label-based path cannot collect a single cross-view LED, exactly
	// as the wrong-depth single-cam commit leaves matched_blobs[] single-cam-only. The label triangulation
	// declines and the only census the legacy L1 could build is best_second_view_leds == 0 -> AGREE-by-absence.
	unlabel_all(s);
	views = make_views(s, cam_poses, cam);
	struct multicam_tri_result lab = {};
	REQUIRE_FALSE(multicam_triangulate_position(views.data(), (int)views.size(), &s.model,
	                                            &P_world_obj.orientation, 0.0f, 2, &lab));

	// THE FIX: prior-seeded epipolar attach over the raw 2nd-cam blobs (reach + model gate as wired in L1).
	struct multicam_tri_result epi = {};
	const bool epi_ok = multicam_triangulate_epipolar_position(views.data(), (int)views.size(), &s.model,
	                                                            &P_world_obj, 0.0f, /*reach_m=*/0.30f,
	                                                            /*model_gate_m=*/0.06f, 2, &epi);
	REQUIRE(epi_ok);
	REQUIRE(epi.num_views >= 2);

	// Re-census exactly as device_multicam_depth_check does from epi.blob_refs[]: per-view inlier LED counts.
	int view_leds[8] = {0};
	for (int i = 0; i < epi.num_blob_refs; i++) {
		const int v = epi.blob_refs[i].view_id;
		if (v >= 0 && v < 8) {
			view_leds[v]++;
		}
	}
	const int committed_view = 0;
	int best_second_view_leds = 0;
	for (int v = 0; v < (int)cam_poses.size(); v++) {
		if (v != committed_view && view_leds[v] > best_second_view_leds) {
			best_second_view_leds = view_leds[v];
		}
	}
	// The 2nd camera now carries >= the dispute bar of device LEDs from the epipolar attach (it was 0 before).
	REQUIRE(best_second_view_leds >= kL1Params.min_dispute_leds);

	// With a present, confident 2nd view, a GROSS wrong-depth commit is no longer untouchable: feed the real
	// epipolar triangulation + a wrong-depth committed position and confirm L1 leaves AGREE-by-absence.
	struct xrt_vec3 wrong_depth = P_world_obj.position;
	wrong_depth.z += 0.40f; // 40 cm depth error, the single-cam weak direction
	struct l1_depth_evidence e = {};
	e.disp_m = m_vec3_len(m_vec3_sub(epi.position, wrong_depth));
	e.triangulated_std_m = (double)epi.position_std_m;
	e.committed_std_m = 0.06;
	e.best_second_view_leds = best_second_view_leds;
	e.n_cams_ge_refine_leds = 0; // 2nd cam supplies 2-3 LEDs: enough to DISPUTE, below the REFINE per-cam bar
	for (int v = 0; v < (int)cam_poses.size(); v++) {
		if (view_leds[v] >= kL1Params.refine_per_cam_leds) {
			e.n_cams_ge_refine_leds++;
		}
	}
	e.tri_succeeded = epi_ok;
	const enum l1_verdict verdict = l1_depth_verdict(&e, &kL1Params);
	REQUIRE(verdict != L1V_AGREE); // L1 ENGAGES (REFINE or DISPUTE) — the 0 -> 0 is closed

	// Monotonicity is preserved: the SAME confident epipolar triangulation that AGREES with a CORRECT commit
	// still AGREES (the fix only disputes a genuinely wrong depth, never a correct one).
	e.disp_m = m_vec3_len(m_vec3_sub(epi.position, P_world_obj.position));
	REQUIRE(l1_depth_verdict(&e, &kL1Params) == L1V_AGREE);
}
