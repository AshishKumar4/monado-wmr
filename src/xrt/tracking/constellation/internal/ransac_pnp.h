/*
 * Pose estimation using OpenCV
 * Copyright 2015 Philipp Zabel
 * Copyright 2020-2023, Jan Schmidt <thaytan@noraisin.net>
 * SPDX-License-Identifier: BSL-1.0
 */
/*
 * Ported from OpenHMD - Free and Open Source API and drivers for immersive technology.
 */
/*!
 * @file
 * @brief  RANSAC PnP pose refinement
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup drv_wmr
 */
#pragma once

#include "xrt/xrt_config_have.h"
#include "xrt/xrt_defines.h"

#include "tracking/t_led_models.h"

#include "blobwatch.h"
#include "camera_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef XRT_HAVE_OPENCV
bool
ransac_pnp_pose(struct xrt_pose *pose,
                struct blob *blobs,
                int num_blobs,
                struct t_constellation_led_model *leds_model,
                struct camera_model *calib,
                int *num_leds_out,
                int *num_inliers);

#else
static inline bool
ransac_pnp_pose(struct xrt_pose *pose,
                struct blob *blobs,
                int num_blobs,
                struct t_constellation_led_model *leds_model,
                struct camera_model *calib,
                int *num_leds_out,
                int *num_inliers)
{
	(void)pose;
	(void)blobs;
	(void)num_blobs;
	(void)leds_model;
	(void)calib;
	(void)num_leds_out;
	(void)num_inliers;
	return false;
}
#endif /* HAVE_OPENCV */

#ifdef __cplusplus
}
#endif
