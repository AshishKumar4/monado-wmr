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

#include "xrt/xrt_frame.h"
#include "blobwatch.h"

#ifdef __cplusplus
extern "C" {
#endif

void
debug_draw_blobs(struct xrt_frame *rgb_out, struct blobservation *bwobs, struct xrt_frame *gray_in);

#ifdef __cplusplus
}
#endif
