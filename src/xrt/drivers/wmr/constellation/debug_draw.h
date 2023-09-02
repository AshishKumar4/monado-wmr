// Copyright 2020-2023, Jan Schmidt <thaytan@noraisin.net>
// SPDX-License-Identifier: BSL-1.0
/*
 * Ported from OpenHMD - Free and Open Source API and drivers for immersive technology.
 */
/*!
 * @file
 * @brief  Debug visualisation for constellation tracking
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup drv_wmr
 */
#pragma once

#include "xrt/xrt_defines.h"
#include "xrt/xrt_frame.h"
#include "sample.h"

#ifdef __cplusplus
extern "C" {
#endif

enum debug_draw_flag
{
	DEBUG_DRAW_FLAG_NONE = 0,
	DEBUG_DRAW_FLAG_BLOB_TINT = 1,
	DEBUG_DRAW_FLAG_BLOB_CIRCLE = 2,
	DEBUG_DRAW_FLAG_BLOB_IDS = 4,
	DEBUG_DRAW_FLAG_BLOB_UNIQUE_IDS = 8,
	DEBUG_DRAW_FLAG_LEDS = 16,
	DEBUG_DRAW_FLAG_POSE_BOUNDS = 32,
	DEBUG_DRAW_FLAG_NORMALISE = 64,
	DEBUG_DRAW_FLAG_ALL = 0xff,
};


void
debug_draw_blobs_leds(struct xrt_frame *rgb_out,
                      struct xrt_frame *gray_in,
                      enum debug_draw_flag flags,
                      struct tracking_sample_frame *view,
                      struct camera_model *calib,
                      struct tracking_sample_device_state *devices,
                      uint8_t n_devices);

#ifdef __cplusplus
}
#endif
