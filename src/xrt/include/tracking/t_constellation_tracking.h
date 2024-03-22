// Copyright 2023 Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Implementation of WMR controller tracking logic
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup drv_wmr
 */
#pragma once

#include <stdint.h>

#include "os/os_threading.h"
#include "tracking/t_tracking.h"
#include "tracking/t_led_models.h"
#include "util/u_sink.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_frame.h"

#include "../../drivers/wmr/wmr_config.h"

#ifdef __cplusplus
extern "C" {
#endif

struct t_constellation_tracker;
struct t_constellation_tracked_device_connection;

int
t_constellation_tracker_create(struct xrt_frame_context *xfctx,
                               struct xrt_device *hmd_xdev,
                               struct wmr_hmd_config *hmd_cfg,
                               struct t_slam_calibration *slam_calib,
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
