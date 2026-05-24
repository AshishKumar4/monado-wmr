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

#include "t_tracker_kalman_fusion.hpp"
#include "xrt/xrt_tracking.h"

#include <cstdlib>
#include <vector>

using xrt::auxiliary::tracking::KalmanFusionInterface;

struct KalmanFusionInterfaceWrapper
{
	std::unique_ptr<KalmanFusionInterface> fusion;

	KalmanFusionInterfaceWrapper() : fusion(KalmanFusionInterface::create()) {}

	~KalmanFusionInterfaceWrapper() {}
};

extern "C" {

struct KalmanFusionInterfaceWrapper *
kalman_fusion_create(void)
{
	return new KalmanFusionInterfaceWrapper;
}

void
kalman_fusion_destroy(KalmanFusionInterfaceWrapper *wrapper)
{
	delete wrapper;
}

void
kalman_fusion_add_ui(struct KalmanFusionInterfaceWrapper *wrapper, void *root, const char *device_name)
{
	wrapper->fusion->add_ui(root, device_name);
}

void
kalman_fusion_process_imu_data(KalmanFusionInterfaceWrapper *wrapper,
                               const struct xrt_imu_sample *sample,
                               const struct xrt_vec3 *accel_variance_optional,
                               const struct xrt_vec3 *gyro_variance_optional)
{
	wrapper->fusion->process_imu_data(sample, accel_variance_optional, gyro_variance_optional);
}

void
kalman_fusion_process_pose(KalmanFusionInterfaceWrapper *wrapper,
                           const struct xrt_pose_sample *sample,
                           const struct xrt_vec3 *position_variance_optional,
                           const struct xrt_vec3 *orientation_variance_optional,
                           const float residual_limit)
{
	wrapper->fusion->process_pose(sample, position_variance_optional, orientation_variance_optional,
	                              residual_limit);
}

float
kalman_fusion_process_led_observations(struct KalmanFusionInterfaceWrapper *wrapper,
                                       timepoint_ns timestamp_ns,
                                       const struct kalman_led_observation *obs,
                                       size_t obs_count,
                                       const struct kalman_led_camera_view *view,
                                       const struct xrt_vec2 *pixel_variance_optional,
                                       float max_innov_px,
                                       bool feed)
{
	if (wrapper == nullptr || view == nullptr || (obs == nullptr && obs_count > 0)) {
		return -1.0f;
	}
	// LEDObservation is an alias of kalman_led_observation, so the C array constructs the vector
	// directly (no per-element conversion).
	std::vector<kalman_led_observation> v(obs, obs + obs_count);
	return wrapper->fusion->process_led_observations(timestamp_ns, v, *view, pixel_variance_optional,
	                                                 max_innov_px, feed);
}

void
kalman_fusion_get_prediction(struct KalmanFusionInterfaceWrapper *wrapper,
                             const timepoint_ns timestamp_ns,
                             struct xrt_space_relation *out_relation)
{
	wrapper->fusion->get_prediction(timestamp_ns, out_relation);
}

bool
kalman_fusion_get_pose_uncertainty(struct KalmanFusionInterfaceWrapper *wrapper,
                                   double *position_std,
                                   double *orientation_std)
{
	if (wrapper == nullptr) {
		return false;
	}
	return wrapper->fusion->get_pose_uncertainty(position_std, orientation_std);
}

void
kalman_fusion_set_imu_calibration(struct KalmanFusionInterfaceWrapper *wrapper,
                                  const double gyro_bias[3],
                                  const double accel_bias[3],
                                  double accel_scale)
{
	wrapper->fusion->set_imu_calibration(gyro_bias, accel_bias, accel_scale);
}

bool
kalman_fusion_get_imu_calibration(struct KalmanFusionInterfaceWrapper *wrapper,
                                  double gyro_bias[3],
                                  double accel_bias[3],
                                  double *accel_scale)
{
	return wrapper->fusion->get_imu_calibration(gyro_bias, accel_bias, accel_scale);
}
}
