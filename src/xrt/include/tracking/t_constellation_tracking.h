// Copyright 2023 Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Implementation of LED constellation tracking logic
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup constellation
 */
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#include "os/os_threading.h"
#include "tracking/t_tracking.h"
#include "tracking/t_led_models.h"
#include "util/u_sink.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_frame.h"
#include "xrt/xrt_tracking.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * @defgroup constellation LED constellation tracking
 * @ingroup tracking
 *
 * @brief Tracker for devices with LED constellations
 */

/*!
 * @dir tracking/constellation
 *
 * @brief @ref constellation tracking files.
 */

struct t_constellation_tracker;
struct t_constellation_tracked_device_connection;

struct t_constellation_camera
{
	//!< IMU to camera pose
	struct xrt_pose P_imu_cam;
	//! ROI in the full frame mosaic
	struct xrt_rect roi;
	//! Intrinsics and distortion parameters
	struct t_camera_calibration calibration;
	//! Minimum blob brightness threshold
	uint8_t min_threshold;
	//! Minimum blob brightness threshold for pixel inclusion
	uint8_t blob_min_threshold;
	//! Threshold at which a group of pixels become a detected blob
	uint8_t blob_detect_threshold;
	//! The index into the slam tracking camera array this camera represents
	size_t slam_tracking_index;
};

struct t_constellation_camera_group
{
	int cam_count; //!< Number of cameras
	struct t_constellation_camera cams[XRT_TRACKING_MAX_SLAM_CAMS];
};

int
t_constellation_tracker_create(struct xrt_frame_context *xfctx,
                               struct xrt_device *hmd_xdev,
                               struct t_constellation_camera_group *cams,
                               struct t_constellation_tracker **out_tracker,
                               struct xrt_frame_sink **out_sink,
                               struct xrt_device_masks_sink *controller_mask_sink);

/*!
 * Serialise a camera group (per-camera intrinsics, distortion model + params, IMU->camera extrinsic,
 * mosaic ROI and blob thresholds) to JSON. Lets the exact calibration the tracker is built from be
 * persisted for OFFLINE replay of the full controller VIO against recorded raw frames — the per-unit
 * camera calibration otherwise lives only in headset flash. Self-describing ("g2-constellation-cameras"
 * v1). No-op on NULL args.
 */
void
t_constellation_camera_group_dump_json(const struct t_constellation_camera_group *cams, FILE *f);

/*!
 * One matched constellation LED for the tightly-coupled per-LED fusion feed: the observed blob
 * centroid (UNDISTORTED, normalized image coords — i.e. already through t_camera_models_undistort)
 * paired with the LED's 3D model position in the device's OpenXR object frame (the OpenXR<->OpenCV
 * YZ flip vs the constellation's OpenCV frame is pre-applied by the tracker).
 */
struct t_constellation_led_obs
{
	struct xrt_vec2 obs_px;  //!< undistorted blob centroid, in camera PIXELS (distortion removed)
	struct xrt_vec3 led_obj; //!< LED position in the OpenXR object frame (m)
};

//! Camera pinhole intrinsics for one view, so the fusion can reproject in physical pixels (the per-LED
//! measurement noise/gate are then focal-independent). Matches @ref t_constellation_led_obs::obs_px,
//! which is the undistorted pixel position under exactly these intrinsics.
struct t_constellation_cam_calib
{
	float fx, fy, cx, cy;
};

struct t_constellation_tracked_device_callbacks
{
	bool (*get_led_model)(struct xrt_device *xdev, struct t_constellation_led_model *led_model);
	void (*notify_frame_received)(struct xrt_device *xdev, uint64_t frame_mono_ns, uint64_t frame_sequence);
	void (*push_observed_pose)(struct xrt_device *xdev, timepoint_ns frame_mono_ns, const struct xrt_pose *pose);
	void (*push_brightness_update)(struct xrt_device *xdev, uint8_t average_brightness);
	//! Per-view matched LED correspondences for the tightly-coupled (ESKF) fusion feed. @p
	//! P_xrworld_cam maps the device's OpenXR world frame to this camera (OpenCV camera frame),
	//! consistent with @ref t_constellation_led_obs::led_obj; @p cam_calib are the view's pinhole
	//! intrinsics matching @ref t_constellation_led_obs::obs_px. Optional — may be left NULL.
	void (*push_observed_leds)(struct xrt_device *xdev, timepoint_ns frame_mono_ns,
	                           const struct xrt_pose *P_xrworld_cam,
	                           const struct t_constellation_cam_calib *cam_calib,
	                           const struct t_constellation_led_obs *leds, size_t led_count);
	//! Current 1-sigma uncertainty of the device's predicted (prior) pose: scalar position std (m) and
	//! orientation std (rad) from the fusion covariance. Lets the matcher size its prior-consistency
	//! gate by the filter's live confidence (tight when tracked, wide after a dropout) instead of a
	//! fixed tolerance. Optional — may be NULL, or return false until the fusion is tracking; the
	//! matcher then falls back to its fixed default bounds.
	bool (*get_pose_uncertainty)(struct xrt_device *xdev, double *position_std, double *orientation_std);
};

struct t_constellation_tracked_device_connection *
t_constellation_tracker_add_device(struct t_constellation_tracker *ct,
                                   struct xrt_device *xdev,
                                   struct t_constellation_tracked_device_callbacks *cb);
void
t_constellation_tracked_device_connection_disconnect(struct t_constellation_tracked_device_connection *ctdc);

#ifdef __cplusplus
}
#endif
