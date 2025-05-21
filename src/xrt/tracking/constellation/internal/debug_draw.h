// Copyright 2020-2023, Jan Schmidt <thaytan@noraisin.net>
// SPDX-License-Identifier: BSL-1.0
/*
 * Ported from OpenHMD - Free and Open Source API and drivers for immersive technology.
 */
/*!
 * @file
 * @brief  Debug visualisation for constellation tracking
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup constellation
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
	DEBUG_DRAW_FLAG_BLOB_TINT = 1 << 0,
	DEBUG_DRAW_FLAG_BLOB_CIRCLE = 1 << 1,
	DEBUG_DRAW_FLAG_BLOB_IDS = 1 << 2,
	DEBUG_DRAW_FLAG_BLOB_UNIQUE_IDS = 1 << 3,
	DEBUG_DRAW_FLAG_LEDS = 1 << 4,
	DEBUG_DRAW_FLAG_PRIOR_LEDS = 1 << 5,
	DEBUG_DRAW_FLAG_LAST_SEEN_LEDS = 1 << 6,
	DEBUG_DRAW_FLAG_POSE_BOUNDS = 1 << 7,
	DEBUG_DRAW_FLAG_NORMALISE = 1 << 8,
	DEBUG_DRAW_FLAG_DEVICE_BOUNDS = 1 << 9,
	DEBUG_DRAW_FLAG_ALL = 0xfff,
};


void
debug_draw_blobs_leds(struct xrt_frame *rgb_out,
                      struct xrt_frame *gray_in,
                      enum debug_draw_flag flags,
                      struct tracking_sample_frame *view,
                      int view_id,
                      struct camera_model *calib,
                      struct tracking_sample_device_state *devices,
                      uint8_t n_devices);

#ifdef __cplusplus
}
#endif
