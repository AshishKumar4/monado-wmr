#include "flexkalman/BaseTypes.h"
#include "flexkalman/FlexibleKalmanBase.h"

namespace flexkalman {
class IMUBiasState : public flexkalman::StateBase<IMUBiasState> {
public:
	static constexpr size_t Dimension = 7;
	using StateVector = types::Vector<Dimension>;
        using StateSquareMatrix = types::SquareMatrix<Dimension>;
	using StateVectorBlock3 = StateVector::FixedSegmentReturnType<3>::Type;
	using ConstStateVectorBlock3 = StateVector::ConstFixedSegmentReturnType<3>::Type;

	IMUBiasState(): m_state({0,0,0, 0,0,0, 1}), m_covariance(StateSquareMatrix::Identity() * 1e-5) { }

	void setStateVector(StateVector const &state) { m_state = state; }
	StateVector stateVector() const { return m_state; }

	void setErrorCovariance(StateSquareMatrix covariance) { m_covariance = covariance; }
	StateSquareMatrix errorCovariance() const { return m_covariance; }

	void postCorrect() {}

	StateVectorBlock3 accelBias() { return m_state.segment<3>(0); }
	ConstStateVectorBlock3 accelBias() const { return m_state.segment<3>(0); }
	
	StateVectorBlock3 gyroBias() { return m_state.segment<3>(3); }
	ConstStateVectorBlock3 gyroBias() const { return m_state.segment<3>(3); }

	double &gravityScale() { return m_state(6); }
	const double &gravityScale() const { return m_state(6); }

private:
	StateVector m_state;
	StateSquareMatrix m_covariance;
};
}
