// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Multi-camera joint (generalised / non-central) PnP over pooled LED bearing rays.
 * @ingroup constellation
 *
 * A single camera that sees only a near-planar LED subset of a controller admits the intrinsic
 * front/back PnP mirror two-fold: two poses reproject the same blobs almost identically, so a
 * single-view solve cannot tell them apart and the front-end can only re-rank them with the
 * orientation prior. When two or more rigidly-mounted cameras co-see the controller, their bearing
 * rays — combined through the known rig extrinsics — observe the same LEDs from genuinely different
 * directions. Solving ONE generalised PnP over all the pooled rays at once constrains the pose
 * jointly and the mirror ambiguity dissolves by construction (the second mode no longer fits both
 * cameras), pulling the solve above the flip-prone few-inlier floor.
 *
 * The solve runs in the shared rig (IMU) frame: each labelled blob contributes a ray (its camera
 * centre + undistorted bearing, mapped into the IMU frame by that camera's extrinsic), and a
 * candidate object pose P_imu_obj places each LED on its ray. A Levenberg-Marquardt minimisation on
 * the SE(3) tangent of P_imu_obj, seeded by the fusion prior, drives the pooled point-to-ray
 * residuals down. Robustified per-ray (Huber) so a single mislabel cannot capture the solve.
 */
#pragma once

#include "xrt/xrt_defines.h"

#include "tracking/t_led_models.h"

#include "blobwatch.h"
#include "camera_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define JOINT_PNP_MAX_VIEWS XRT_TRACKING_MAX_SLAM_CAMS

/* The per-LED bitmask that deduplicates rays within a view is a uint64_t, so at most 64 distinct LEDs
 * (== the LED_LOCAL_ID 8-bit id space's working subset) can contribute a ray per camera. */
#define JOINT_PNP_MAX_LEDS_PER_VIEW 64

/*!
 * One camera's contribution to a joint solve: its labelled blobs, intrinsics/distortion, and the
 * camera-to-rig extrinsic P_imu_cam (camera pose in the shared IMU/rig frame). Only blobs already
 * labelled for the device being solved are used (LED_OBJECT_ID == device id), exactly as the
 * single-camera RANSAC-PnP selects its correspondences.
 */
struct joint_pnp_view
{
	struct blob *blobs;
	int num_blobs;
	struct camera_model *calib;
	struct xrt_pose P_imu_cam;
};

/*!
 * Solve the device pose jointly over @p num_views camera views (each with its own labelled blobs and
 * extrinsic), seeded by @p pose. @p pose is in/out: on input the prior in the IMU/rig frame
 * (P_imu_obj), on success the refined joint solution in the same frame. Returns true and sets
 * @p num_rays_out / @p num_inliers when at least two views each contribute LEDs and the joint solve
 * converges to a consistent (low pooled residual) pose; false when there is not enough cross-camera
 * information to beat the single-camera path (the caller then falls back to it — no regression).
 *
 * @p leds_model is the device LED model (LED positions in the object/model frame, as the
 * single-camera PnP uses them). The same LED indexing (LED_LOCAL_ID) maps a labelled blob to its 3D
 * point, deduplicated per LED across all views (one ray per LED per camera).
 */
bool
joint_pnp_solve(struct xrt_pose *pose,
                const struct joint_pnp_view *views,
                int num_views,
                struct t_constellation_led_model *leds_model,
                int *num_rays_out,
                int *num_inliers);

#ifdef __cplusplus
}
#endif
