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

#include "os/os_threading.h"
#include "tracking/t_tracking.h"
#include "tracking/t_led_models.h"
#include "util/u_sink.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_frame.h"

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
                               struct xrt_frame_sink **out_sink);

struct t_constellation_tracked_device_callbacks
{
	bool (*get_led_model)(struct xrt_device *xdev, struct t_constellation_led_model *led_model);
	void (*notify_frame_received)(struct xrt_device *xdev, uint64_t frame_mono_ns, uint64_t frame_sequence);
	void (*push_observed_pose)(struct xrt_device *xdev, timepoint_ns frame_mono_ns, const struct xrt_pose *pose);
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
