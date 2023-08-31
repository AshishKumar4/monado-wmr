// Copyright 2023, Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Constellation tracking details for 1 exposure sample
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup drv_wmr
 */

#pragma once

#include "xrt/xrt_defines.h"
#include "xrt/xrt_frame.h"

#include "blobwatch.h"
#include "pose_metrics.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CONSTELLATION_MAX_DEVICES 3
#define CONSTELLATION_MAX_CAMERAS 5

/* Information about one device being tracked in this sample */
struct tracking_sample_device_state
{
	/* Index into the devices array for this state info */
	int dev_index;

	struct constellation_led_model *led_model;

	/* Predicted device pose and error bounds from fusion, in world space */
	struct xrt_pose prior_pose;
	struct xrt_vec3 prior_pos_error;
	struct xrt_vec3 prior_rot_error;
	float gravity_error_rad; /* Gravity vector uncertainty in radians 0..M_PI */

	/* Last observed pose, in world space */
	bool have_last_seen_pose;
	struct xrt_pose last_seen_pose;

	bool found_device_pose;     /* Set to true when the device was found in this sample */
	int found_pose_view_id;     /* Set to the camera ID where the device was found */
	struct xrt_pose final_pose; /* Global pose that was detected */

	struct pose_metrics score;
	struct pose_metrics_blob_match_info blob_match_info;
};

/* Information about 1 camera frame in this sample */
struct tracking_sample_frame
{
	/* Video frame data we are analysing */
	struct xrt_frame *vframe;

	/* The pose from which this view is observed */
	struct xrt_pose pose;
	/* Inverse of the pose from which this view is observed */
	struct xrt_pose inv_pose;
	/* Gravity vector as observed from this camera */
	struct xrt_vec3 cam_gravity_vector;

	/* blobs observation and the owning blobwatch */
	blobwatch *bw;
	blobservation *bwobs;
};

struct constellation_tracking_sample
{
	uint64_t timestamp; // Exposure timestamp

	/* Device poses at capture time */
	struct tracking_sample_device_state devices[CONSTELLATION_MAX_DEVICES];
	uint8_t n_devices;

	struct tracking_sample_frame views[CONSTELLATION_MAX_CAMERAS];
	uint8_t n_views;

	bool need_long_analysis;

	bool long_analysis_found_new_blobs;
};

struct constellation_tracking_sample *
constellation_tracking_sample_new();
void
constellation_tracking_sample_free(struct constellation_tracking_sample *sample);

#ifdef __cplusplus
}
#endif
