/*!
 * @file
 * @brief  Per-LED multi-camera triangulation of a controller position (no full constellation / pose).
 * @ingroup constellation
 *
 * A single camera that sees fewer than four cleanly-PnP-able LEDs of a controller yields NO pose: the
 * full-constellation PnP needs >= 4 LEDs in one view, and the joint (generalised) PnP needs >= 3 LEDs per
 * view in >= 2 views. The dominant recall floor is exactly the sparse-LED regime below those thresholds
 * (oblique / far / frame-edge / motion-blur; the far-side LEDs face away), where the controller still
 * shows one or two LEDs to each of several cameras.
 *
 * The four head cameras are rigidly mounted with validated extrinsics — a known multi-baseline rig. When a
 * SINGLE controller LED is co-identified in >= 2 cameras (the front-end already labels each blob with its
 * (device, led_id); blobs across cameras sharing the same label are the same physical LED), its 3D
 * position triangulates directly from the bearing rays of those cameras — no constellation, no orientation.
 *
 * Each triangulated LED world point is mapped to a controller-origin estimate by subtracting the known
 * model lever arm rotated by the (gravity-anchored, tilt-driftless) prior orientation, then the per-LED
 * estimates are robustly averaged into ONE controller position with a covariance. The result feeds the
 * ESKF as a position-only measurement (decoupled from orientation), keeping the track alive through the
 * sparse-LED stretches that produce nothing today.
 */
#pragma once

#include "xrt/xrt_defines.h"

#include "tracking/t_led_models.h"

#include "blobwatch.h"
#include "camera_model.h"

#ifdef __cplusplus
extern "C" {
#endif

//! At most this many cameras can contribute a ray to one LED's triangulation (one per view).
#define MULTICAM_TRI_MAX_VIEWS XRT_TRACKING_MAX_SLAM_CAMS

/*!
 * One camera's contribution: its blobs (already labelled for the device being triangulated, exactly as
 * the single-camera PnP selects its correspondences), intrinsics/distortion, and the camera-to-CV-world
 * extrinsic @p P_world_cam (the camera pose in the OpenCV-convention world frame the matcher operates in).
 */
struct multicam_tri_view
{
	struct blob *blobs;
	int num_blobs;
	struct camera_model *calib;
	struct xrt_pose P_world_cam; //!< camera pose in the OpenCV-world frame (= sample view->P_world_cam)
};

/*!
 * One contributing blob in a successful triangulation, so the caller can register the cross-view evidence
 * with the association hypothesis (distinct-view count + per-device blob ownership).
 */
struct multicam_tri_blob_ref
{
	int view_id; //!< index into the @p views array passed to multicam_triangulate_position
	int blob_id; //!< index into that view's blobs[]
	int led_id;  //!< LED_LOCAL_ID of the triangulated LED
};

/*!
 * Result of triangulating a device's co-seen LEDs into a single controller position.
 */
struct multicam_tri_result
{
	struct xrt_vec3 position;      //!< controller model-origin position, OpenCV-world frame (m)
	float position_std_m;          //!< isotropic 1-sigma of the aggregated position (m)
	int num_leds;                  //!< distinct LEDs that triangulated and survived the robust average
	int num_views;                 //!< distinct cameras that contributed at least one inlier ray
	int num_blob_refs;             //!< number of valid entries in blob_refs
	struct multicam_tri_blob_ref blob_refs[MULTICAM_TRI_MAX_VIEWS * 64];
};

/*!
 * Triangulate the device's position from LEDs co-identified across @p num_views cameras.
 *
 * @p prior_orientation is the controller's CURRENT estimated orientation in the OpenCV-world frame (the
 * gravity-anchored prior). It is used ONLY to subtract each LED's known model lever arm so distinct LEDs
 * reference the same rigid-body point; tilt is driftless and the residual yaw error on a <= ~5 cm lever is
 * small and folded into the reported covariance. @p prior_yaw_sigma_rad is the prior orientation's 1-sigma
 * yaw uncertainty (rad); the lever-arm subtraction injects a position error ~ |lever| * yaw_sigma that the
 * inter-LED spread alone underestimates when the visible LEDs share a lever direction, so it is added in
 * quadrature to the reported position std (an honest, yaw-aware measurement noise for the ESKF fold).
 * @p leds_model supplies the per-LED model positions and the device id selecting which blobs to use.
 *
 * Returns true and fills @p out_result when at least @p min_leds LEDs are each co-seen by >= 2 cameras,
 * triangulate to a low per-ray residual, and agree with each other (the robust average rejects outliers and
 * requires >= 2 surviving inlier LEDs across >= 2 cameras). Returns false when there is not enough
 * cross-camera agreement — the caller then adds no triangulation hypothesis (no false position). The solve
 * is purely geometric and never commits a pose; the caller decides how to fold the position.
 */
bool
multicam_triangulate_position(const struct multicam_tri_view *views,
                              int num_views,
                              const struct t_constellation_led_model *leds_model,
                              const struct xrt_quat *prior_orientation,
                              float prior_yaw_sigma_rad,
                              int min_leds,
                              struct multicam_tri_result *out_result);

/*!
 * Prior-free EPIPOLAR triangulation: recover the device position from blobs matched across cameras by
 * epipolar geometry alone, NOT by a shared front-end label. The label-based @ref multicam_triangulate_position
 * only fires when the same led_id was independently labelled in >= 2 views; at a camera's FOV edge (few
 * labelled LEDs) that misses, even though the same physical LEDs are imaged. This pass instead pairs every
 * cross-view blob whose bearing rays nearly intersect (the epipolar constraint from the known rig
 * extrinsics), giving each pair a 3D point with NO prior label.
 *
 * Precision is held by the SAME multi-LED consensus the label path uses, plus two extra guards that make
 * the prior-free matching safe against clutter: each candidate point must (1) lie within @p reach_m of the
 * IMU-predicted prior position @p P_world_obj_prior (an out-of-reach intersection is a wrong pairing /
 * clutter), and (2) sit within @p model_gate_m of one of the model LEDs PROJECTED by the prior pose — i.e.
 * the recovered constellation must be geometrically consistent with the rigid controller at the predicted
 * body pose. A point that matches no model LED, or matches one already claimed by a closer point, is
 * dropped. Surviving points inherit the led_id of their matched model LED, are merged per-LED across views,
 * re-triangulated, and aggregated by the shared robust consensus (>= 2 mutually-agreeing LEDs, >= 2
 * cameras, residual + parallax gates) — so this NEVER commits a position the rig geometry does not support.
 *
 * @p P_world_obj_prior is the full predicted device pose (orientation subtracts the lever arm AND projects
 * the model for the gate; position bounds the reach). Returns true and fills @p out_result on consensus.
 */
bool
multicam_triangulate_epipolar_position(const struct multicam_tri_view *views,
                                       int num_views,
                                       const struct t_constellation_led_model *leds_model,
                                       const struct xrt_pose *P_world_obj_prior,
                                       float prior_yaw_sigma_rad,
                                       float reach_m,
                                       float model_gate_m,
                                       int min_leds,
                                       struct multicam_tri_result *out_result);

/*!
 * Triangulate from one labelled primary view plus unlabelled secondary-view blobs. This is the depth-check
 * primitive for a single-camera pose lock: the committing camera already identified local LED ids, while a
 * co-visible camera may have raw blobs that the wrong-depth pose failed to label. The solve pairs each
 * labelled primary LED ray with raw rays from other views, then keeps only cross-LED origin estimates that
 * form a rigid controller under @p prior_orientation. It does not gate on prior position.
 */
bool
multicam_triangulate_primary_label_epipolar_position(const struct multicam_tri_view *views,
                                                     int num_views,
                                                     int primary_view,
                                                     const struct t_constellation_led_model *leds_model,
                                                     const struct xrt_quat *prior_orientation,
                                                     float prior_yaw_sigma_rad,
                                                     int min_leds,
                                                     struct multicam_tri_result *out_result);

#ifdef __cplusplus
}
#endif
