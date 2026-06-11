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
 * @ingroup constellation
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
/*!
 * RANSAC + LM PnP solve over the identified-LED blobs. When the inlier LED set is near-planar (the few-LED,
 * edge-on geometry that causes the PnP mirror two-fold ambiguity), also returns the second (mirror-twin)
 * pose so the caller can pick the one consistent with the IMU/fusion prior instead of silently committing
 * one. @p twin receives the analytic twin and @p has_twin is set true only when a distinct second solution
 * exists; otherwise @p has_twin is false and @p twin is untouched. @p pose is always the primary
 * (RANSAC + LM) solution on success.
 */
bool
ransac_pnp_pose_with_twin(struct xrt_pose *pose,
                          struct blob *blobs,
                          int num_blobs,
                          struct t_constellation_led_model *leds_model,
                          struct camera_model *calib,
                          int *num_leds_out,
                          int *num_inliers,
                          struct xrt_pose *twin,
                          bool *has_twin);

bool
ransac_pnp_tilt_clamp(const struct xrt_pose *pose,
                      const struct xrt_vec3 *obj_pts,
                      const struct xrt_vec2 *img_pts,
                      int n,
                      struct camera_model *calib,
                      const struct xrt_pose *prior_cam,
                      const struct xrt_vec3 *up_cam,
                      bool require_coplanar,
                      struct xrt_pose *out,
                      struct xrt_pose *yaw_twin,
                      bool *has_yaw_twin);

int
pnp_solve_p3p(struct blob *blobs,
              int num_blobs,
              struct t_constellation_led_model *leds_model,
              struct camera_model *calib,
              struct xrt_pose *out_poses,
              int max_out);

#else
static inline bool
ransac_pnp_pose_with_twin(struct xrt_pose *pose,
                          struct blob *blobs,
                          int num_blobs,
                          struct t_constellation_led_model *leds_model,
                          struct camera_model *calib,
                          int *num_leds_out,
                          int *num_inliers,
                          struct xrt_pose *twin,
                          bool *has_twin)
{
	(void)pose;
	(void)blobs;
	(void)num_blobs;
	(void)leds_model;
	(void)calib;
	(void)num_leds_out;
	(void)num_inliers;
	(void)twin;
	if (has_twin != NULL) {
		*has_twin = false;
	}
	return false;
}

static inline bool
ransac_pnp_tilt_clamp(const struct xrt_pose *pose,
                      const struct xrt_vec3 *obj_pts,
                      const struct xrt_vec2 *img_pts,
                      int n,
                      struct camera_model *calib,
                      const struct xrt_pose *prior_cam,
                      const struct xrt_vec3 *up_cam,
                      bool require_coplanar,
                      struct xrt_pose *out,
                      struct xrt_pose *yaw_twin,
                      bool *has_yaw_twin)
{
	(void)pose;
	(void)obj_pts;
	(void)img_pts;
	(void)n;
	(void)calib;
	(void)prior_cam;
	(void)up_cam;
	(void)require_coplanar;
	(void)out;
	(void)yaw_twin;
	if (has_yaw_twin != NULL) {
		*has_yaw_twin = false;
	}
	return false;
}

static inline int
pnp_solve_p3p(struct blob *blobs,
              int num_blobs,
              struct t_constellation_led_model *leds_model,
              struct camera_model *calib,
              struct xrt_pose *out_poses,
              int max_out)
{
	(void)blobs;
	(void)num_blobs;
	(void)leds_model;
	(void)calib;
	(void)out_poses;
	(void)max_out;
	return 0;
}
#endif /* HAVE_OPENCV */

#ifdef __cplusplus
}
#endif
