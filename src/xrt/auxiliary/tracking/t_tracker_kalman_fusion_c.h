// Copyright 2024, Joel Valenciano
// Copyright 2025, Beyley Cardellio
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  C interface to generalized kalman filter.
 *
 * @author Joel Valenciano <joelv1907@gmail.com>
 * @author Beyley Cardellio <ep1cm1n10n123@gmail.com>
 * @ingroup aux_tracking
 */

#include "xrt/xrt_tracking.h"

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct KalmanFusionInterfaceWrapper;

//! One matched constellation LED for the tightly-coupled per-LED fusion update. C-accessible mirror
//! of the C++ LEDObservation (the C++ header aliases this exact struct, so there is one definition).
struct kalman_led_observation
{
	struct xrt_vec2 observed_px; //!< measured blob centroid (undistorted, normalized image coords)
	struct xrt_vec3 led_obj;     //!< LED position in the object frame (caller applies any frame flip)
};

//! Pinhole camera + world->camera extrinsic for one constellation view. For normalized observations
//! pass fx=fy=1, cx=cy=0; the extrinsic must map the FILTER's world frame to the camera frame (the
//! caller folds the OpenXR<->OpenCV flip into this extrinsic and into led_obj).
struct kalman_led_camera_view
{
	float fx, fy, cx, cy;
	struct xrt_quat cam_world_orient; //!< world->camera rotation
	struct xrt_vec3 cam_world_pos;    //!< world->camera translation
};

struct KalmanFusionInterfaceWrapper *
kalman_fusion_create(void);

void
kalman_fusion_add_ui(struct KalmanFusionInterfaceWrapper *wrapper, void *root, const char *device_name);

void
kalman_fusion_destroy(struct KalmanFusionInterfaceWrapper *wrapper);

void
kalman_fusion_process_imu_data(struct KalmanFusionInterfaceWrapper *wrapper,
                               const struct xrt_imu_sample *sample,
                               const struct xrt_vec3 *accel_variance_optional,
                               const struct xrt_vec3 *gyro_variance_optional);

void
kalman_fusion_process_pose(struct KalmanFusionInterfaceWrapper *wrapper,
                           const struct xrt_pose_sample *sample,
                           const struct xrt_vec3 *position_variance_optional,
                           const struct xrt_vec3 *orientation_variance_optional,
                           float residual_limit);

//! Tightly-coupled per-LED optical update (the ESKF feed). @p feed=false runs a read-only diagnostic
//! (computes the per-LED reprojection RMS vs the current pose) instead of folding — used to verify
//! the frame/undistort transforms live before trusting the feed. feed=true returns the number of LEDs
//! folded this frame (>=0; -1 if the filter is not yet tracking); feed=false returns the diagnostic
//! reprojection RMS in NORMALIZED image units (or -1 if no pose).
float
kalman_fusion_process_led_observations(struct KalmanFusionInterfaceWrapper *wrapper,
                                       timepoint_ns timestamp_ns,
                                       const struct kalman_led_observation *obs,
                                       size_t obs_count,
                                       const struct kalman_led_camera_view *view,
                                       const struct xrt_vec2 *pixel_variance_optional,
                                       float max_innov_px,
                                       bool feed);

void
kalman_fusion_get_prediction(struct KalmanFusionInterfaceWrapper *wrapper,
                             const timepoint_ns timestamp_ns,
                             struct xrt_space_relation *out_relation);

//! Current 1-sigma uncertainty of the predicted pose (scalar position std in m, orientation std in
//! rad) from the filter covariance — lets a consumer size a prior-consistency gate by the filter's
//! live confidence (tight when tracked, wide after a dropout). Returns false until tracking.
bool
kalman_fusion_get_pose_uncertainty(struct KalmanFusionInterfaceWrapper *wrapper,
                                   double *position_std,
                                   double *orientation_std);

//! Cross-session IMU calibration cache (persisted per controller by the driver). set_* seeds a prior
//! before tracking; get_* reads the current converged gyro/accel bias + accel scale and returns true
//! only once a calibrated stance has occurred (estimate trustworthy to persist).
void
kalman_fusion_set_imu_calibration(struct KalmanFusionInterfaceWrapper *wrapper,
                                  const double gyro_bias[3],
                                  const double accel_bias[3],
                                  double accel_scale);
bool
kalman_fusion_get_imu_calibration(struct KalmanFusionInterfaceWrapper *wrapper,
                                  double gyro_bias[3],
                                  double accel_bias[3],
                                  double *accel_scale);

#ifdef __cplusplus
}
#endif // __cplusplus
