// Copyright 2019, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  C++ sensor fusion/filtering code that uses flexkalman
 * @author Rylie Pavlik <rylie.pavlik@collabora.com>
 * @ingroup aux_tracking
 */

#pragma once

#ifndef __cplusplus
#error "This header is C++-only."
#endif

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "flexkalman/AugmentedProcessModel.h"
#include "flexkalman/AugmentedState.h"
#include "flexkalman/BaseTypes.h"
#include "flexkalman/PoseState.h"


namespace xrt::auxiliary::tracking {

namespace types = flexkalman::types;
using flexkalman::types::Vector;

//! For things like accelerometers, which on some level measure the local vector
//! of a world direction.
template <typename State>
class WorldDirectionMeasurement : public flexkalman::MeasurementBase<WorldDirectionMeasurement<State>>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	static constexpr size_t Dimension = 3;
	using MeasurementVector = types::Vector<Dimension>;
	using MeasurementSquareMatrix = types::SquareMatrix<Dimension>;
	WorldDirectionMeasurement(types::Vector<3> const &direction,
	                          types::Vector<3> const &reference,
	                          types::Vector<3> const &variance)
	    : direction_(direction.normalized()), reference_(reference.normalized()), covariance_(variance.asDiagonal())
	{}

	MeasurementSquareMatrix const &
	getCovariance(State const & /*s*/)
	{
		return covariance_;
	}

	types::Vector<3>
	predictMeasurement(State const &s) const
	{
		return s.getCombinedQuaternion() * reference_;
	}

	MeasurementVector
	getResidual(MeasurementVector const &predictedMeasurement, State const &s) const
	{
		return predictedMeasurement - reference_;
	}

	MeasurementVector
	getResidual(State const &s) const
	{
		return getResidual(predictMeasurement(s), s);
	}

private:
	types::Vector<3> direction_;
	types::Vector<3> reference_;
	MeasurementSquareMatrix covariance_;
};
#if 0
//! For things like accelerometers, which on some level measure the local vector
//! of a world direction.
class LinAccelWithGravityMeasurement
    : public flexkalman::MeasurementBase<LinAccelWithGravityMeasurement>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	static constexpr size_t Dimension = 3;
	using MeasurementVector = types::Vector<Dimension>;
	using MeasurementSquareMatrix = types::SquareMatrix<Dimension>;
	LinAccelWithGravityMeasurement(types::Vector<3> const &direction,
	                          types::Vector<3> const &reference,
	                          types::Vector<3> const &variance)
	    : direction_(direction), reference_(reference),
	      covariance_(variance.asDiagonal())
	{}

	// template <typename State>
	MeasurementSquareMatrix const &
	getCovariance(State const & /*s*/)
	{
		return covariance_;
	}

	// template <typename State>
	types::Vector<3>
	predictMeasurement(State const &s) const
	{
		return reference_;
	}

	// template <typename State>
	MeasurementVector
	getResidual(MeasurementVector const &predictedMeasurement,
	            State const &s) const
	{
		s.getQuaternion().conjugate() *
		        predictedMeasurement return predictedMeasurement -
		    reference_.normalized();
	}

	template <typename State>
	MeasurementVector
	getResidual(State const &s) const
	{
		MeasurementVector residual =
		    direction_ - reference_ * s.getQuaternion();
		return getResidual(predictMeasurement(s), s);
	}

private:
	types::Vector<3> direction_;
	types::Vector<3> reference_;
	MeasurementSquareMatrix covariance_;
};
#endif

class BiasedGyroMeasurement : public flexkalman::MeasurementBase<BiasedGyroMeasurement>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	static constexpr size_t Dimension = 3;
	using MeasurementVector = types::Vector<Dimension>;
	using MeasurementSquareMatrix = types::SquareMatrix<Dimension>;
	BiasedGyroMeasurement(types::Vector<3> const &angVel, types::Vector<3> const &variance)
	    : angVel_(angVel), covariance_(variance.asDiagonal())
	{}

	template <typename State>
	MeasurementSquareMatrix const &
	getCovariance(State const & /*s*/)
	{
		return covariance_;
	}

	template <typename State>
	MeasurementVector
	predictMeasurement(State const &s) const
	{
		return (s.a().angularVelocity() - s.b().gyroBias());
	}

	template <typename State>
	MeasurementVector
	getResidual(MeasurementVector const &predictedMeasurement, State const &s) const
	{
		return angVel_ - predictedMeasurement;
	}

	template <typename State>
	MeasurementVector
	getResidual(State const &s) const
	{
		return getResidual(predictMeasurement(s), s);
	}

private:
	types::Vector<3> angVel_;
	MeasurementSquareMatrix covariance_;
};
/*!
 * For PS Move-like things, where there's a directly-computed absolute position
 * that is not at the tracked body's origin.
 */
class AbsolutePositionLeverArmMeasurement : public flexkalman::MeasurementBase<AbsolutePositionLeverArmMeasurement>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	using State = flexkalman::pose_externalized_rotation::State;
	static constexpr size_t Dimension = 3;
	using MeasurementVector = types::Vector<Dimension>;
	using MeasurementSquareMatrix = types::SquareMatrix<Dimension>;

	/*!
	 * @todo the point we get from the camera isn't the center of the ball,
	 * but the center of the visible surface of the ball - a closer
	 * approximation would be translation along the vector to the center of
	 * projection....
	 */
	AbsolutePositionLeverArmMeasurement(MeasurementVector const &measurement,
	                                    MeasurementVector const &knownLocationInBodySpace,
	                                    MeasurementVector const &variance)
	    : measurement_(measurement), knownLocationInBodySpace_(knownLocationInBodySpace),
	      covariance_(variance.asDiagonal())
	{}

	MeasurementSquareMatrix const &
	getCovariance(State const & /*s*/)
	{
		return covariance_;
	}

	types::Vector<3>
	predictMeasurement(State const &s) const
	{
		return s.getIsometry() * knownLocationInBodySpace_;
	}

	MeasurementVector
	getResidual(MeasurementVector const &predictedMeasurement, State const & /*s*/) const
	{
		return measurement_ - predictedMeasurement;
	}

	MeasurementVector
	getResidual(State const &s) const
	{
		return getResidual(predictMeasurement(s), s);
	}

private:
	MeasurementVector measurement_;
	MeasurementVector knownLocationInBodySpace_;
	MeasurementSquareMatrix covariance_;
};

/*!
 * Tightly-coupled per-LED reprojection measurement: one observed 2D blob pixel
 * for a single constellation LED whose 3D position in the controller's object
 * frame is known. predictMeasurement projects that LED through the current
 * filter pose, a fixed world->camera extrinsic, and a pinhole camera, and the
 * residual is the pixel innovation z - h(x). Unlike the all-or-nothing PnP pose
 * path, a single LED is enough to apply a correction, so frames with too few
 * matched LEDs to solve a pose still inform the filter.
 *
 * Orientation observability: predictMeasurement builds the object->world
 * transform from position() + getCombinedQuaternion(), NOT getIsometry().
 * getIsometry() uses the externalized base quaternion getQuaternion(), which
 * the UKF sigma points do NOT perturb (they perturb incrementalOrientation(),
 * indices 3..5). Projecting through getCombinedQuaternion() lets the sigma-point
 * spread in orientation reach the predicted pixel, so a reprojection genuinely
 * observes orientation/yaw — the whole point of going tightly coupled.
 *
 * Camera model: a plain pinhole (fx, fy, cx, cy), no distortion. That is fine
 * for this benchmark, which generates pixels from the same pinhole geometry.
 * NOTE: production must project with the real t_camera_models / radtan8|kb4
 * distortion (see t_camera_models_project, pose_metrics.c:137-151) so the
 * predicted pixel matches the constellation blob detector's distorted pixels.
 */
class LEDReprojectionMeasurement : public flexkalman::MeasurementBase<LEDReprojectionMeasurement>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	using State = flexkalman::pose_externalized_rotation::State;
	static constexpr size_t Dimension = 2;
	using MeasurementVector = types::Vector<Dimension>;
	using MeasurementSquareMatrix = types::SquareMatrix<Dimension>;

	//! @param observed_px  measured blob pixel (z), camera image coordinates.
	//! @param led_obj      LED position in the controller object/model frame (m).
	//! @param T_cam_world  world->camera transform for this view (OpenCV frame:
	//!                     +X right, +Y down, +Z forward into the scene).
	//! @param fx,fy,cx,cy  pinhole intrinsics (px).
	//! @param px_variance  per-axis pixel measurement variance (px^2), R diagonal.
	LEDReprojectionMeasurement(Eigen::Vector2d const &observed_px,
	                           Eigen::Vector3d const &led_obj,
	                           Eigen::Isometry3d const &T_cam_world,
	                           double fx,
	                           double fy,
	                           double cx,
	                           double cy,
	                           Eigen::Vector2d const &px_variance)
	    : z_(observed_px), led_obj_(led_obj), T_cam_world_(T_cam_world), fx_(fx), fy_(fy), cx_(cx), cy_(cy),
	      covariance_(px_variance.asDiagonal())
	{}

	MeasurementSquareMatrix const &
	getCovariance(State const & /*s*/)
	{
		return covariance_;
	}

	MeasurementVector
	predictMeasurement(State const &s) const
	{
		// object->world using position + the COMBINED quaternion so the
		// sigma-point orientation spread is observable (see class doc).
		const Eigen::Vector3d p_world = s.position() + s.getCombinedQuaternion() * led_obj_;
		// world->camera (OpenCV frame).
		const Eigen::Vector3d p_cam = T_cam_world_ * p_world;
		// Pinhole projection. Guard against a point at/behind the camera so a
		// degenerate sigma point cannot produce a non-finite measurement; a
		// tiny positive epsilon keeps the projection finite and the resulting
		// huge residual is harmlessly down-weighted by the UKF / gate.
		const double z = (p_cam.z() > 1e-6) ? p_cam.z() : 1e-6;
		MeasurementVector px;
		px[0] = fx_ * (p_cam.x() / z) + cx_;
		px[1] = fy_ * (p_cam.y() / z) + cy_;
		return px;
	}

	MeasurementVector
	getResidual(MeasurementVector const &predictedMeasurement, State const & /*s*/) const
	{
		return z_ - predictedMeasurement; // pixel innovation
	}

	MeasurementVector
	getResidual(State const &s) const
	{
		return getResidual(predictMeasurement(s), s);
	}

private:
	Eigen::Vector2d z_;
	Eigen::Vector3d led_obj_;
	Eigen::Isometry3d T_cam_world_;
	double fx_, fy_, cx_, cy_;
	MeasurementSquareMatrix covariance_;
};

} // namespace xrt::auxiliary::tracking
