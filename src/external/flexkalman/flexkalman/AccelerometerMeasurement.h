// Copyright 2019, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  PS Move tracker code that is expensive to compile.
 *
 * Typically built as a part of t_kalman.cpp to reduce incremental build times.
 *
 * @author Rylie Pavlik <rylie.pavlik@collabora.com>
 * @author Pete Black <pblack@collabora.com>
 * @author Jakob Bornecrantz <jakob@collabora.com>
 * @ingroup aux_tracking
 */

#include "BaseTypes.h"
#include "flexkalman/FlexibleKalmanBase.h"

namespace flexkalman {
class AccelerometerMeasurement
    : public flexkalman::MeasurementBase<AccelerometerMeasurement> {

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static constexpr size_t Dimension = 3;
    using MeasurementVector = types::Vector<Dimension>;
    using MeasurementSquareMatrix = types::SquareMatrix<Dimension>;

    AccelerometerMeasurement(types::Vector<3> const &accel,
                             types::Vector<3> const &gravity_ref,
                             types::Vector<3> const &variance)
        : accel_(accel), gravity_ref_(gravity_ref), covariance_(variance.asDiagonal()) {}

    template<typename State>
    MeasurementSquareMatrix const &getCovariance(State const & /*s*/)

    {
        return covariance_;
    }

    template<typename State>
    MeasurementVector predictMeasurement(State const &s) const {
        // An accelerometer measures specific force in the sensor (body)
        // frame: the proper acceleration that opposes gravity. Physically
        //   a_meas = R_world->body * (a_world - g_world) + accelBias
        // where a_world is the body's linear acceleration (carried in the
        // pose state) and g_world is gravitational acceleration (gravity_ref_,
        // e.g. (0,-9.81,0)). The state quaternion q is body->world, so
        // R_world->body == q.conjugate(). At rest (a_world == 0) this
        // correctly predicts -R*g_world, i.e. the upward gravity reaction.
        auto q = s.a().getCombinedQuaternion();
        return q.conjugate() * (s.a().acceleration() - gravity_ref_) +
               s.b().accelBias();
    }

    template<typename State>
    MeasurementVector getResidual(MeasurementVector const &predictedMeasurement,
                                  State const &s) const {
        return accel_ - predictedMeasurement;
    }

    template<typename State>
    MeasurementVector getResidual(State const &s) const {
        return getResidual(predictMeasurement(s), s);
    }

  private:
    MeasurementVector accel_;
    MeasurementVector gravity_ref_;
    MeasurementSquareMatrix covariance_;
};
} // namespace flexkalman
