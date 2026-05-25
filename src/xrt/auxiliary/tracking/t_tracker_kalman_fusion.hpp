// Copyright 2019, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  PS Move tracker code.
 * @author Pete Black <pblack@collabora.com>
 * @author Jakob Bornecrantz <jakob@collabora.com>
 * @author Rylie Pavlik <rylie.pavlik@collabora.com>
 * @ingroup aux_tracking
 */

#pragma once

#ifndef __cplusplus
#error "This header is C++-only."
#endif

#include "xrt/xrt_defines.h"
#include "xrt/xrt_tracking.h"

#include "util/u_time.h"

#include <memory>
#include <vector>

// The per-LED observation + camera-view structs are defined once as C structs (so the C driver code
// can build them) and aliased here for the C++ filter — single definition, no duplication.
#include "t_tracker_kalman_fusion_c.h"


namespace xrt::auxiliary::tracking {

//! @see kalman_led_observation (C definition). Pixel is undistorted-normalized; led_obj is in the
//! object frame with any OpenXR<->OpenCV flip pre-applied by the caller.
using LEDObservation = ::kalman_led_observation;
//! @see kalman_led_camera_view. fx=fy=1/cx=cy=0 for normalized obs; extrinsic maps the filter's
//! world frame to the camera frame (caller folds in the YZ flip).
using LEDCameraView = ::kalman_led_camera_view;

class KalmanFusionInterface
{
public:
	static std::unique_ptr<KalmanFusionInterface>
	create();
	virtual ~KalmanFusionInterface() = default;

	virtual void
	add_ui(void *root, const char *device_name) = 0;

	/*!
	 * @brief If you've lost sight of the position tracking and won't even
	 * enter another function in this class.
	 */
	virtual void
	clear_position_tracked_flag() = 0;

	virtual void
	process_imu_data(const struct xrt_imu_sample *sample,
	                 const struct xrt_vec3 *accel_variance_optional,
	                 const struct xrt_vec3 *gyro_variance_optional) = 0;
	virtual void
	process_pose(const struct xrt_pose_sample *sample,
	             const struct xrt_vec3 *position_variance_optional,
	             const struct xrt_vec3 *orientation_variance_optional,
	             const float residual_limit,
	             const struct xrt_pose *hmd_world_pose) = 0;

	/*!
	 * Tightly-coupled optical update: fold each matched LED's reprojection
	 * into the filter directly, instead of solving one PnP pose and feeding
	 * that. Works with as few as a single LED, so frames that have too few
	 * matched LEDs for a PnP pose still inform the filter. Each LED is gated
	 * individually (a per-LED pixel innovation threshold) so a mislabelled
	 * correspondence cannot drag the pose.
	 *
	 * @param timestamp_ns    capture time of this view.
	 * @param obs             matched (blob pixel, LED object-frame point) pairs.
	 * @param view            pinhole intrinsics + world->camera extrinsic.
	 * @param pixel_variance  per-axis pixel measurement variance (px^2).
	 * @param max_innov_px    per-LED robust gate: skip an LED whose pixel
	 *                        innovation magnitude exceeds this (mislabel guard).
	 * @param feed            true: fold the observations into the filter (tightly-coupled
	 *                        update). false: DIAGNOSTIC only — compute the per-LED reprojection
	 *                        residual against the current pose without touching the filter
	 *                        (read-only; used to verify the frame/undistort transforms before
	 *                        enabling the feed).
	 * @param hmd_world_pose  live HMD pose in the controller's world frame (or null). Used to gate
	 *                        adoption on the controller-to-HMD distance (arm reach, room-roam-invariant)
	 *                        and to capture the body-lock offset when a fold constrains position.
	 * @return feed=true: the number of LEDs actually folded this frame (after the per-LED gate; 0 =
	 *         all gated out as mislabels), or -1 if the filter is not yet tracking (awaiting a
	 *         process_pose bootstrap). feed=false (diagnostic): the RMS per-LED reprojection residual
	 *         in NORMALIZED image units (~residual_px / focal; < ~0.01 ⇒ frame transform correct), or
	 *         -1 if no pose was available.
	 */
	virtual float
	process_led_observations(const timepoint_ns timestamp_ns,
	                         const std::vector<LEDObservation> &obs,
	                         const LEDCameraView &view,
	                         const struct xrt_vec2 *pixel_variance,
	                         const float max_innov_px,
	                         const bool feed,
	                         const struct xrt_pose *hmd_world_pose) = 0;

	/*!
	 * @param hmd_world_pose live HMD pose in the controller's world frame (or null). When position is no
	 * longer optically observable, the reported controller rides this live HMD pose at its last
	 * controller-to-HMD offset (body-lock), instead of dead-reckoning away in world frame. Null falls
	 * back to the world-frame hold at the last optically-observed position.
	 */
	virtual void
	get_prediction(const timepoint_ns when_ns,
	               struct xrt_space_relation *out_relation,
	               const struct xrt_pose *hmd_world_pose) = 0;

	/*!
	 * Predict one LED's image point and its 2x2 innovation covariance S = H P H^T + R for the
	 * covariance-driven associator (the per-LED gate ellipse). Uses the EXACT same H/projection as the
	 * tightly-coupled fold (single source of truth), so the gate the front-end sizes matches the filter's
	 * own measurement model. @p out_zhat receives the predicted pixel; @p out_S receives S row-major
	 * (S00,S01,S10,S11). Returns false (and writes nothing) until the filter is tracking.
	 */
	virtual bool
	predict_led_gate(const LEDObservation &obs, const LEDCameraView &view, float out_zhat[2], float out_S[4])
	{
		(void)obs;
		(void)view;
		(void)out_zhat;
		(void)out_S;
		return false;
	}

	/*!
	 * Diagnostics / tests: the raw per-LED reprojection Jacobian at the current state — @p out_H row-major
	 * 2x6 [d/dp(3), d/dtheta_world(3)] from the SAME led_project_jacobian the fold/gate use, plus the
	 * predicted pixel @p out_zhat. Lets a test pin the H convention directly against a finite-difference
	 * (the global-vs-local orientation-Jacobian regression), which S = H P H^T + R cannot isolate when
	 * P_theta is isotropic. Off the hot path. Returns false if not tracking.
	 */
	virtual bool
	debug_predict_led_jacobian(const LEDObservation &obs, const LEDCameraView &view, double out_zhat[2],
	                           double out_H[12])
	{
		(void)obs;
		(void)view;
		(void)out_zhat;
		(void)out_H;
		return false;
	}

	/*!
	 * Diagnostics / tests: filter consistency. Writes the mean per-DOF NIS over the last folds (a
	 * correctly-tuned filter averages ~1) and the most recent fold's per-DOF NIS, and returns the number
	 * of folds in the window (0 if none yet). Either out pointer may be null.
	 */
	virtual int
	debug_get_nis_stats(double *mean_per_dof, double *last_per_dof)
	{
		(void)mean_per_dof;
		(void)last_per_dof;
		return 0;
	}

	/*!
	 * Current 1-sigma uncertainty of the predicted pose: a scalar position std (m) and orientation
	 * std (rad) from the filter covariance (the worst-direction / largest-eigenvalue std, applied
	 * isotropically — frame-conservative, since the covariance is world-frame while a consumer's gate
	 * may be in another frame, and variance in any direction is bounded by the largest eigenvalue).
	 * Lets the constellation matcher size its prior-consistency gate by the filter's ACTUAL confidence
	 * instead of a fixed tolerance: tight when well-tracked, wide right after an optical dropout (when
	 * the filter inflates P). Wait-free (reads the published snapshot). Returns false if not tracking.
	 */
	virtual bool
	get_pose_uncertainty(double *position_std, double *orientation_std)
	{
		(void)position_std;
		(void)orientation_std;
		return false;
	}

	/*!
	 * Diagnostics / tests: copy the current estimate's 3x3 position covariance
	 * (row-major, m^2). Off the hot path. Returns false if unavailable or not yet
	 * tracking. Used by the filter-consistency (NEES) test and live debugging.
	 */
	virtual bool
	debug_get_position_covariance(double cov_row_major[9])
	{
		(void)cov_row_major;
		return false;
	}

	/*!
	 * Diagnostics / tests: copy the 6x6 pose covariance (row-major) ordered [position(3), global
	 * orientation(3)] — the [EP,ET] sub-blocks of the 15x15 P. Lets a test reconstruct the per-LED
	 * innovation covariance S = H P H^T + R from the same blocks predict_led_gate uses. Off the hot
	 * path. Returns false if not tracking.
	 */
	virtual bool
	debug_get_pose_covariance(double cov6_row_major[36])
	{
		(void)cov6_row_major;
		return false;
	}

	//! Diagnostics / tests: the current online accel-scale correction (nominal 1.0). 1.0 if unsupported.
	virtual double
	debug_get_accel_scale()
	{
		return 1.0;
	}

	//! Diagnostics / tests: the fitted full accel calibration matrix (row-major, per-axis scale +
	//! misalignment). Returns false until the ellipsoid has been fitted from enough rest orientations.
	virtual bool
	debug_get_accel_calibration(double T_row_major[9])
	{
		(void)T_row_major;
		return false;
	}

	/*!
	 * Cross-session IMU calibration cache. The driver persists the converged per-controller gyro/accel
	 * bias + accel scale (keyed by serial) and seeds a new session with them, so the first second is
	 * accurate before any stance and a large physical bias survives. set_* applies a prior (before
	 * tracking); get_* reads the current converged values to persist (returns false until a calibrated
	 * stance has occurred, i.e. the estimate is trustworthy).
	 */
	virtual void
	set_imu_calibration(const double gyro_bias[3], const double accel_bias[3], double accel_scale)
	{
		(void)gyro_bias;
		(void)accel_bias;
		(void)accel_scale;
	}
	virtual bool
	get_imu_calibration(double gyro_bias[3], double accel_bias[3], double *accel_scale)
	{
		(void)gyro_bias;
		(void)accel_bias;
		(void)accel_scale;
		return false;
	}

	/*!
	 * Optically-derived IMU INTRINSICS (computed offline by imu_calib_from_optical.py and persisted per
	 * controller serial alongside the bias cache). Two 3x3 corrections applied in the IMU integration
	 * BEFORE bias/integration drift the state:
	 *  - @p gyro_correction (M_g, row-major): corrected body rate = M_g * (gyro_meas - bg). Captures gyro
	 *    scale (its singular values) AND the gyro->device misalignment (its orthogonal part) in one matrix
	 *    — the exact M the offline tool fits from phi_optical = M * gyro_integral (single source of truth).
	 *  - @p accel_correction (T_a, row-major): corrected specific force = T_a * (accel_meas - ba) — the
	 *    accel ellipsoid (per-axis scale + cross-axis). Seeds m_accel_T so the offline ellipsoid is applied
	 *    immediately, rather than waiting for the rarely-occurring online rest-orientation spread.
	 * Either may be identity (then that channel is uncorrected — no regression). set_* applies a prior
	 * before tracking; get_* reads the currently-applied intrinsics to persist. get_* returns false if
	 * neither channel is a real correction (nothing worth persisting). The bias/scale path is unchanged.
	 */
	virtual void
	set_imu_intrinsics(const double gyro_correction[9], const double accel_correction[9])
	{
		(void)gyro_correction;
		(void)accel_correction;
	}
	virtual bool
	get_imu_intrinsics(double gyro_correction[9], double accel_correction[9])
	{
		(void)gyro_correction;
		(void)accel_correction;
		return false;
	}
};
} // namespace xrt::auxiliary::tracking
