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
#include "util/u_sink.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_frame.h"

#include "wmr_config.h"
#include "wmr_controller_base.h"

#ifdef __cplusplus
extern "C" {
#endif

struct wmr_controller_tracker;
struct wmr_controller_tracker_connection;

void
wmr_controller_tracker_connection_disconnect(struct wmr_controller_tracker_connection *wctc);

int
wmr_controller_tracker_create(struct xrt_frame_context *xfctx,
                              struct xrt_device *hmd_xdev,
                              struct wmr_hmd_config *hmd_cfg,
                              struct t_slam_calibration *slam_calib,
                              struct wmr_controller_tracker **out_tracker,
                              struct xrt_frame_sink **out_sink);

struct wmr_controller_tracker_connection *
wmr_controller_tracker_add_controller(struct wmr_controller_tracker *tracker, struct wmr_controller_base *wcb);

#ifdef __cplusplus
}
#endif
