// Copyright 2024, Joel Valenciano
// Copyright 2025, Beyley Cardellio
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Tightly-coupled controller fusion: a purpose-built error-state Kalman
 *         filter (ESKF) over the controller 6-DOF pose, IMU and per-LED
 *         constellation reprojection. Replaces the prior flexkalman UKF + per-LED
 *         measurement, which death-spiralled live (fixed-pixel gate trapped a
 *         drifted state -> no folds -> IMU runaway -> reset to origin -> repeat).
 *
 *         The LED 3D positions in the controller frame are KNOWN (firmware
 *         constellation model), so this is not SLAM: a direct ESKF with
 *         known-landmark reprojection updates is optimal (no MSCKF clones /
 *         null-space machinery, which exist only for unknown features). Full
 *         15x15 covariance, analytic Jacobians, chi-square (covariance-aware)
 *         per-LED gating that self-recovers, Huber robustness, online accel/gyro
 *         bias, Joseph-form covariance, and PnP re-anchor on divergence.
 *
 *         Design + math: docs/ESKF-DESIGN.md. Convention: GLOBAL (world-frame)
 *         angular error, q_true = exp(dtheta_w) (x) q_nominal.
 *
 * @author Joel Valenciano <joelv1907@gmail.com>
 * @author Beyley Cardellio <ep1cm1n10n123@gmail.com>
 * @ingroup aux_tracking
 */
#include <algorithm>
#include <atomic>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include <iomanip>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "xrt/xrt_tracking.h"

#include "tracking/t_tracker_kalman_fusion.hpp"

#include "math/m_eigen_interop.hpp"
#include "math/m_api.h"

#include "os/os_time.h"

#include "util/u_debug.h"
#include "util/u_logging.h"
#include "util/u_misc.h"
#include "util/u_var.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/LU>


DEBUG_GET_ONCE_OPTION(kalman_record_path, "KALMAN_RECORD_PATH", NULL)

#define CSV_EOL "\r\n"
#define CSV_PRECISION 10

namespace xrt::auxiliary::tracking {

using namespace xrt::auxiliary::math;

//! Anonymous namespace to hide implementation names
namespace {
	using Eigen::AngleAxisd;
	using Eigen::Quaterniond;
	using Eigen::Vector2d;
	using Eigen::Vector3d;
	using Mat3 = Eigen::Matrix3d;
	using Mat15 = Eigen::Matrix<double, 15, 15>;
	using Vec15 = Eigen::Matrix<double, 15, 1>;
	using MatX = Eigen::MatrixXd;
	using VecX = Eigen::VectorXd;

	// Error-state index layout (15): position, velocity, orientation (global), accel bias, gyro bias.
	constexpr int EP = 0;  //!< delta position
	constexpr int EV = 3;  //!< delta velocity
	constexpr int ET = 6;  //!< delta orientation (world-frame rotation vector)
	constexpr int EBA = 9; //!< delta accel bias (body)
	constexpr int EBG = 12; //!< delta gyro bias (body)

	//! Compile-time unit helpers: each tuning constant below is written as the meaningful quantity
	//! (a std, an angle in degrees, a time in ms) instead of a pre-squared / pre-radian / pre-scaled
	//! magic number. One source of truth per knob — no value-vs-comment drift, nothing to hand-compute.
	constexpr double sq(double x) { return x * x; }
	constexpr double deg2rad(double d) { return d * double(EIGEN_PI) / 180.0; }
	constexpr int64_t ms_to_ns(double ms) { return (int64_t)(ms * 1.0e6); }
	//! chi-square inverse-CDF for 2 DOF: chi2inv(p) = -2 ln(1-p). Lets the gate/Huber thresholds be
	//! expressed as the meaningful confidence p (not a magic quantile). Not constexpr (std::log isn't
	//! in C++17); the two callers use it at static-init only.
	inline double chi2inv_2dof(double p) { return -2.0 * std::log(1.0 - p); }

	//! Skew-symmetric (cross-product) matrix.
	inline Mat3
	skew(const Vector3d &v)
	{
		Mat3 m;
		m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
		return m;
	}

	//! Full accelerometer ellipsoid calibration: given bias-corrected at-rest samples @p v that span
	//! orientations, find the symmetric correction T (per-axis scale + misalignment) such that the
	//! corrected specific force |T·v| == g in every orientation. At rest the raw readings trace an
	//! ellipsoid (deformed by scale/misalignment); fitting it maps that ellipsoid back to a sphere of
	//! radius g. Magnitude alone fixes only the symmetric part (TᵀT); the residual rotation is the body
	//! frame's, left to the orientation estimate — so the symmetric square root T = g·√A is the unique
	//! physical correction. Returns false if the data is degenerate (≈coplanar) or the fit isn't a
	//! plausible small correction. Pure math (testable in isolation).
	inline bool
	fit_accel_calibration(const std::vector<Vector3d> &v, double g, Mat3 &T_out)
	{
		const int n = (int)v.size();
		if (n < 9) {
			return false; // need enough orientations to constrain the 6-parameter symmetric A
		}
		// The fit is unidentifiable unless the sample directions span 3D (not collinear/coplanar).
		Mat3 dcov = Mat3::Zero();
		for (const Vector3d &s : v) {
			const Vector3d u = s.normalized();
			dcov += u * u.transpose();
		}
		dcov /= n;
		if (Eigen::SelfAdjointEigenSolver<Mat3>(dcov).eigenvalues()(0) < 0.04) {
			return false; // directions ~coplanar
		}
		// Least-squares fit of symmetric A (vᵀ A v = 1): 6 unique params [Axx,Ayy,Azz,Axy,Axz,Ayz].
		MatX M(n, 6);
		VecX rhs = VecX::Ones(n);
		for (int i = 0; i < n; i++) {
			const Vector3d &s = v[i];
			M.row(i) << s.x() * s.x(), s.y() * s.y(), s.z() * s.z(), 2 * s.x() * s.y(), 2 * s.x() * s.z(),
			    2 * s.y() * s.z();
		}
		const Eigen::Matrix<double, 6, 1> p = (M.transpose() * M).ldlt().solve(M.transpose() * rhs);
		Mat3 A;
		A << p(0), p(3), p(4), p(3), p(1), p(5), p(4), p(5), p(2);
		Eigen::SelfAdjointEigenSolver<Mat3> es(A);
		const Vector3d ev = es.eigenvalues();
		if (!ev.allFinite() || ev(0) <= 1e-9) {
			return false; // A not positive-definite -> not a valid ellipsoid
		}
		// T = g·√A (symmetric). |T·v| = g·√(vᵀ A v) = g at rest.
		const Mat3 T = g * (es.eigenvectors() * ev.cwiseSqrt().asDiagonal() * es.eigenvectors().transpose());
		const Vector3d evT = Eigen::SelfAdjointEigenSolver<Mat3>(T).eigenvalues();
		if (evT(0) < 0.8 || evT(2) > 1.25) {
			return false; // a real correction is a small scaling; reject a wild (bad-data) fit
		}
		T_out = T;
		return true;
	}

	//! Rotation vector -> unit quaternion (exp map). Small-angle safe.
	inline Quaterniond
	exp_quat(const Vector3d &dtheta)
	{
		const double a = dtheta.norm();
		if (a < 1e-9) {
			Quaterniond q(1.0, 0.5 * dtheta.x(), 0.5 * dtheta.y(), 0.5 * dtheta.z());
			q.normalize();
			return q;
		}
		const Vector3d axis = dtheta / a;
		const double s = std::sin(0.5 * a);
		return Quaterniond{std::cos(0.5 * a), s * axis.x(), s * axis.y(), s * axis.z()};
	}

	//! Unit quaternion -> rotation vector (log map). Returns the minimal 3-vector.
	inline Vector3d
	log_quat(const Quaterniond &q_in)
	{
		Quaterniond q = q_in.normalized();
		if (q.w() < 0.0) {
			q.coeffs() *= -1.0; // shortest path
		}
		const Vector3d v = q.vec();
		const double n = v.norm();
		if (n < 1e-9) {
			return 2.0 * v;
		}
		const double angle = 2.0 * std::atan2(n, q.w());
		return angle * (v / n);
	}

	class ImuPoseRecorder
	{
	public:
		ImuPoseRecorder(const char *record_path, const char *device_name)
		    : m_lock(), m_recording(false), m_record_path(record_path), m_device_name(device_name)
		{}

		void
		start()
		{
			std::lock_guard<std::mutex> lock(m_lock);
			if (m_recording) {
				return;
			}

			// Create dataset directories with current datetime suffix
			time_t seconds = os_realtime_get_ns() / U_1_000_000_000;
			constexpr size_t size = sizeof("YYYYMMDDHHmmss");
			char datetime[size] = {0};
			(void)strftime(datetime, size, "%Y%m%d%H%M%S", localtime(&seconds));
			std::string record_path = m_record_path + "/" + datetime + "_" + m_device_name;
			std::filesystem::create_directories(record_path);

			m_imu_csv = new std::ofstream{record_path + "/imu.csv"};
			*m_imu_csv << std::fixed << std::setprecision(CSV_PRECISION);
			*m_imu_csv << "#timestamp [ns],w_x [rad s^-1],w_y [rad s^-1],w_z [rad s^-1],"
			              "a_x [m s^-2],a_y [m s^-2],a_z [m s^-2]" CSV_EOL;

			m_pose_csv = new std::ofstream{record_path + "/pose.csv"};
			*m_pose_csv << std::fixed << std::setprecision(CSV_PRECISION);
			*m_pose_csv << "#timestamp [ns],p_x [m],p_y [m],p_z [m],"
			               "q_w [],q_x [],q_y [],q_z []" CSV_EOL;

			// Kalman-filtered output (the pose the fusion actually reports).
			m_fused_csv = new std::ofstream{record_path + "/fused.csv"};
			*m_fused_csv << std::fixed << std::setprecision(CSV_PRECISION);
			*m_fused_csv << "#timestamp [ns],p_x [m],p_y [m],p_z [m],"
			                "q_w [],q_x [],q_y [],q_z [],"
			                "v_x [m s^-1],v_y [m s^-1],v_z [m s^-1]" CSV_EOL;

			m_recording = true;
		}

		void
		stop()
		{
			std::lock_guard<std::mutex> lock(m_lock);
			if (!m_recording) {
				return;
			}

			if (m_imu_csv) {
				delete m_imu_csv;
				m_imu_csv = nullptr;
			}
			if (m_pose_csv) {
				delete m_pose_csv;
				m_pose_csv = nullptr;
			}
			if (m_fused_csv) {
				delete m_fused_csv;
				m_fused_csv = nullptr;
			}
			m_recording = false;
		}

		void
		process_imu_data(const struct xrt_imu_sample *sample)
		{
			std::lock_guard<std::mutex> lock(m_lock);
			if (!m_recording) {
				return;
			}

			timepoint_ns ts = sample->timestamp_ns;
			xrt_vec3_f64 a = sample->accel_m_s2;
			xrt_vec3_f64 w = sample->gyro_rad_secs;

			*m_imu_csv << ts << ",";
			*m_imu_csv << w.x << "," << w.y << "," << w.z << ",";
			*m_imu_csv << a.x << "," << a.y << "," << a.z << CSV_EOL;
		}

		void
		process_pose(const struct xrt_pose_sample *sample)
		{
			std::lock_guard<std::mutex> lock(m_lock);
			if (!m_recording) {
				return;
			}

			timepoint_ns ts = sample->timestamp_ns;
			xrt_vec3 p = sample->pose.position;
			xrt_quat o = sample->pose.orientation;

			*m_pose_csv << ts << ",";
			*m_pose_csv << p.x << "," << p.y << "," << p.z << ",";
			*m_pose_csv << o.w << "," << o.x << "," << o.y << "," << o.z << CSV_EOL;
		}

		//! Log one fused (Kalman output) sample.
		void
		process_fused(timepoint_ns ts, const struct xrt_space_relation *rel)
		{
			std::lock_guard<std::mutex> lock(m_lock);
			if (!m_recording) {
				return;
			}
			xrt_vec3 p = rel->pose.position;
			xrt_quat o = rel->pose.orientation;
			xrt_vec3 v = rel->linear_velocity;
			*m_fused_csv << ts << ",";
			*m_fused_csv << p.x << "," << p.y << "," << p.z << ",";
			*m_fused_csv << o.w << "," << o.x << "," << o.y << "," << o.z << ",";
			*m_fused_csv << v.x << "," << v.y << "," << v.z << CSV_EOL;
		}

	private:
		std::mutex m_lock;
		bool m_recording;
		std::string m_record_path;
		std::string m_device_name;

		std::ofstream *m_imu_csv = nullptr;
		std::ofstream *m_pose_csv = nullptr;
		std::ofstream *m_fused_csv = nullptr;
	};

	struct TrackingInfo
	{
		bool valid{false};
		bool tracked{false};
	};

	//! The ESKF nominal state: the writable best estimate (16 values; the
	//! covariance tracks the 15 error states). Orientation is world<-body.
	struct NominalState
	{
		Vector3d p{0, 0, 0};       //!< position, world
		Vector3d v{0, 0, 0};       //!< velocity, world
		Quaterniond q{1, 0, 0, 0}; //!< orientation, world<-body
		Vector3d ba{0, 0, 0};      //!< accelerometer bias, body
		Vector3d bg{0, 0, 0};      //!< gyroscope bias, body
	};

	//! Immutable filter state published for wait-free reads by get_prediction. A
	//! trivially-copyable POD so the seqlock copies it as a flat block; a torn
	//! read is validated and retried. Quaternion stored in Eigen coefficient
	//! order (x, y, z, w).
	struct FilterSnapshot
	{
		double position[3];
		double orientation[4];
		double linear_velocity[3];
		double angular_velocity[3]; //!< world-frame, last sample
		double acceleration[3];     //!< world-frame incl. gravity, last sample
		double last_good_position[3];
		double position_var_max;    //!< largest eigenvalue of P[EP,EP] (m^2): worst-direction variance
		double orientation_var_max; //!< largest eigenvalue of P[ET,ET] (rad^2): worst-direction variance
		timepoint_ns filter_time_ns;
		timepoint_ns last_optical_ns;
		bool tracked;
		bool position_valid;
		bool position_tracked;
		bool orientation_valid;
		bool orientation_tracked;
	};

	//! A complete copy of the mutable filter state, to rewind for out-of-sequence
	//! (lagged) optical measurements. The ESKF carries a full covariance, so the
	//! checkpoint includes P (unlike the old block-diagonal UKF checkpoint).
	struct FilterCheckpoint
	{
		NominalState nominal;
		Mat15 P{Mat15::Identity()};
		Vector3d accel_world{0, 0, 0};
		Vector3d angvel_world{0, 0, 0};
		Vector3d last_good_position{0, 0, 0};
		timepoint_ns filter_time_ns{0};
		timepoint_ns last_imu_ns{0};
		timepoint_ns last_optical_ns{0};
		timepoint_ns last_led_fold_ns{0};
		TrackingInfo orientation_state;
		TrackingInfo position_state;
		int imu_anomaly_count{0};
		bool tracked{false};
	};

	//! One buffered IMU sample retained for out-of-sequence replay.
	struct ImuLogEntry
	{
		xrt_imu_sample sample;
	};

	class EskfFusion : public KalmanFusionInterface
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		EskfFusion()
		{
			reset_filter_and_imu();
			if (getenv("G2_IMU_ONLY") != nullptr) {
				m_imu_only = true;
				const char *bs = getenv("G2_IMU_ONLY_BOOTSTRAP_S");
				const double s = (bs != nullptr && *bs != '\0') ? atof(bs) : 3.0;
				m_imu_only_bootstrap_ns = (int64_t)(s * 1.0e9);
				U_LOG_I("ESKF: G2_IMU_ONLY diagnostic — pure inertial after %.1fs optical bootstrap", s);
			}
		}

		void
		clear_position_tracked_flag() override;

		void
		process_imu_data(const struct xrt_imu_sample *sample,
		                 const struct xrt_vec3 *accel_variance_optional,
		                 const struct xrt_vec3 *gyro_variance_optional) override;
		void
		process_pose(const struct xrt_pose_sample *sample,
		             const struct xrt_vec3 *position_variance_optional,
		             const struct xrt_vec3 *orientation_variance_optional,
		             const float residual_limit) override;

		float
		process_led_observations(const timepoint_ns timestamp_ns,
		                         const std::vector<LEDObservation> &obs,
		                         const LEDCameraView &view,
		                         const struct xrt_vec2 *pixel_variance,
		                         const float max_innov_px,
		                         const bool feed) override;

		void
		get_prediction(const timepoint_ns when_ns, struct xrt_space_relation *out_relation) override;

		bool
		get_pose_uncertainty(double *position_std, double *orientation_std) override
		{
			const FilterSnapshot snap = read_snapshot();
			if (!snap.tracked) {
				return false;
			}
			// Worst-direction (largest-eigenvalue) 1-sigma, applied isotropically by the caller. This is
			// frame-conservative: the covariance is world-frame and the caller's gate is in another frame,
			// and uncertainty in any direction u is u'Pu <= lambda_max, so this never wrongly tightens.
			if (position_std != nullptr) {
				*position_std = std::sqrt(std::max(0.0, snap.position_var_max));
			}
			if (orientation_std != nullptr) {
				*orientation_std = std::sqrt(std::max(0.0, snap.orientation_var_max));
			}
			return true;
		}

		bool
		debug_get_position_covariance(double cov_row_major[9]) override;

		double
		debug_get_accel_scale() override
		{
			return m_accel_scale;
		}

		bool
		debug_get_accel_calibration(double T_row_major[9]) override
		{
			if (!m_accel_T_valid) {
				return false;
			}
			for (int r = 0; r < 3; r++) {
				for (int c = 0; c < 3; c++) {
					T_row_major[r * 3 + c] = m_accel_T(r, c);
				}
			}
			return true;
		}

		//! Cross-session calibration prior: seed gyro/accel bias + accel scale from the driver's persisted
		//! per-controller cache (applied at the next bootstrap). Call before tracking starts.
		void
		set_imu_calibration(const double gyro_bias[3], const double accel_bias[3], double accel_scale) override
		{
			m_cached_bg = Vector3d(gyro_bias[0], gyro_bias[1], gyro_bias[2]);
			m_cached_ba = Vector3d(accel_bias[0], accel_bias[1], accel_bias[2]);
			m_cached_scale = accel_scale;
			m_cal_primed = true;
		}

		//! Read the current converged gyro/accel bias + scale (for the driver to persist). Returns false
		//! unless a calibrated stance occurred this session (otherwise the estimate is not trustworthy).
		bool
		get_imu_calibration(double gyro_bias[3], double accel_bias[3], double *accel_scale) override
		{
			std::lock_guard<std::mutex> lock(m_filter_lock);
			if (!m_stance_snap) {
				return false; // no plausible stance captured -> nothing trustworthy to persist
			}
			for (int i = 0; i < 3; i++) {
				gyro_bias[i] = m_stance_bg[i];   // the last good-stance estimate, not the (maybe-drifted) live state
				accel_bias[i] = m_stance_ba[i];
			}
			*accel_scale = m_stance_scale;
			return true;
		}

		void
		add_ui(void *root, const char *device_name) override;

	private:
		// ---- Tuning. Each knob is written via the unit helpers above as the meaningful quantity
		// (std / degrees / ms); the few genuinely-independent physical+statistical params are grouped. ----
		//! IMU continuous noise spectral densities (ICM-20602 datasheet, inflated for unmodeled
		//! effects). accel white (m/s^2/sqrt(Hz)), gyro white (rad/s/sqrt(Hz)), and bias random walks.
		static constexpr double SIGMA_A = 0.02;        //!< accel white noise
		static constexpr double SIGMA_G = 2.0e-3;      //!< gyro white noise
		static constexpr double SIGMA_BA = 5.0e-4;     //!< accel bias random walk
		static constexpr double SIGMA_BG = 1.0e-4;     //!< gyro bias random walk (kept adaptive)
		//! Accel gravity-tilt anchor. When the controller is in low linear acceleration (||a|-g| <
		//! GRAV_BAND), the (bias-corrected) accel direction IS the gravity direction, which pins
		//! roll+pitch absolutely (2-DOF) INDEPENDENT of optical — it bleeds gyro tilt-drift and keeps
		//! orientation correct through optical-sparse / flipping stretches (the daylight case). Yaw still
		//! needs optical. GRAV_VAR is the unit-vector measurement variance (loose: the gyro leads
		//! short-term, gravity slowly anchors). Replaces the old m_imu_3dof gravity correction.
		static constexpr double GRAV_BAND = 0.6;       //!< ||accel|-g| tolerance (m/s^2) to trust as gravity
		static constexpr double GRAV_VAR = 0.02;       //!< gravity-direction unit-vector measurement var
		//! Gravity anchor also requires near-stationarity: an accelerometer cannot tell "tilted at rest"
		//! from "level but accelerating" (a horizontal accel barely changes |accel|). The FILTER'S
		//! velocity resolves it — only anchor when the estimated speed is below this, so a moving/
		//! dead-reckoning controller is never mis-tilted. (At rest, brief residual motion is tolerated.)
		static constexpr double GRAV_MAX_VEL = 0.1;    //!< m/s; above this, skip the gravity anchor
		//! Stance (device-at-rest) detector shared by ZUPT + online accel-scale. At rest the gyro reads ~0
		//! and the scale-corrected |accel| equals g, so: rest <=> |gyro| < ZUPT_GYRO_MAX AND
		//! | |a_m|*scale - g | < ZUPT_ACCEL_BAND. The band is ~3 sigma of the measured rest accel noise
		//! (spectral floor ~0.05 m/s^2), so a real linear acceleration (which adds to |a_m| in quadrature)
		//! breaks the hypothesis. (Idealised constant-velocity/-acceleration with zero rotation is the IMU's
		//! unobservable mode; like every ZUPT it assumes real motion has rotation/jerk/off-g magnitude.)
		static constexpr double ZUPT_GYRO_MAX = deg2rad(4.0);  //!< rad/s; rest requires |gyro| below this
		static constexpr double ZUPT_ACCEL_BAND = 0.15;        //!< m/s^2; rest requires ||a|*scale - g| below (3 sigma)
		//! ZUPT: at rest, fold a velocity==0 pseudo-measurement (var ZUPT_VAR) once rest is sustained
		//! ZUPT_MIN_REST samples — BUT only while optical is stale (the dead-reckon regime). ZUPT is the
		//! velocity constraint that substitutes for the absent optical one; with optical live it would
		//! wrongly fight an observed velocity, so it yields to it. Bounds the pure-inertial rest drift.
		static constexpr double ZUPT_VAR = sq(0.01);           //!< (m/s)^2 velocity-measurement var (tight)
		static constexpr int ZUPT_MIN_REST = 8;                //!< consecutive rest samples before ZUPT
		//! ZARU (zero angular-rate update): at rest the true angular rate is 0, so the measured gyro IS a
		//! direct observation of the gyro bias (z=gyro, h=bg). The gyro analog of ZUPT — observes ALL THREE
		//! bias axes including YAW, which the gravity anchor fundamentally cannot. An uncorrected gyro bias
		//! drives orientation drift during motion -> gravity leaks into horizontal accel -> position
		//! runaway, so nailing bg at every stance is the single biggest lever for unaided-motion accuracy.
		static constexpr double ZARU_VAR = sq(0.01);           //!< (rad/s)^2 gyro-rate measurement var
		//! Online accel-scale: the per-unit absolute scale is ~2-3% off, and a constant body bias CANNOT
		//! absorb a SCALE error across orientations (it re-projects with the device -> a velocity kick on
		//! every rotation). At rest the true |specific force| is g, so k = g/|a_m| is the ML gain estimate.
		//! Bootstrapped once at first stance, then slow-adapted (SCALE_ALPHA) at stance; clamped to a
		//! physical range. Estimated from RAW |a_m| (not |a_m - ba|) so it does not fight the bias state.
		static constexpr double SCALE_ALPHA = 0.002;           //!< per-stance-sample low-pass rate for k
		static constexpr double SCALE_MIN = 0.9, SCALE_MAX = 1.1; //!< clamp k (physical scale error <10%)
		//! Process noise for a clock advance with NO IMU input (pose-only callers, sub-sample gaps to an
		//! optical capture time). Constant-velocity white-noise-acceleration model: how much the
		//! controller's (unobserved) acceleration / angular rate may vary. Keeps P from collapsing so the
		//! filter stays responsive when IMU propagation is absent.
		static constexpr double PROC_ACCEL_CV = 10.0;  //!< unmodeled linear accel (m/s^2)
		static constexpr double PROC_GYRO_CV = 5.0;    //!< unmodeled angular rate (rad/s)
		//! Per-LED reprojection measurement noise (px std) -> R diagonal = LED_PIXEL_STD^2.
		static constexpr double LED_PIXEL_STD = 1.5;
		//! Per-LED robustness as two confidence levels (the real knobs); the chi-square thresholds are
		//! DERIVED from them via chi2inv_2dof. GATE = acceptance confidence (covariance-aware via S_i, so
		//! it widens automatically as the filter grows uncertain — the self-recovery the old gate lacked);
		//! HUBER = where the robust kernel starts down-weighting (R inflated by sqrt(d2/HUBER), a bend not
		//! a hard cut). GATE must be looser than HUBER (gate >= huber confidence).
		static constexpr double GATE_CONFIDENCE = 0.99;
		static constexpr double HUBER_CONFIDENCE = 0.95;
		static inline const double CHI2_GATE_2DOF = chi2inv_2dof(GATE_CONFIDENCE);  // 9.21
		static inline const double HUBER_DELTA2 = chi2inv_2dof(HUBER_CONFIDENCE);   // 5.99
		//! Divergence: if at least REANCHOR_NMIN LEDs are matched but fewer than this FRACTION pass the
		//! gate, the PREDICTION is wrong (not the LEDs) -> re-anchor + inflate P, never reset to origin.
		static constexpr int REANCHOR_NMIN = 4;
		static constexpr double REANCHOR_FRAC = 0.34;
		//! Re-anchor candidate (last PnP pose) is only trusted this fresh.
		static constexpr int64_t REANCHOR_MAX_AGE_NS = ms_to_ns(100);
		//! Covariance inflation on re-anchor to a TRUSTED PnP pose (accurate, so modest inflation).
		static constexpr double REANCHOR_POS_VAR = sq(0.5);   // 0.5 m
		static constexpr double REANCHOR_ORI_VAR = sq(0.5);   // 0.5 rad ~ 29 deg
		//! Covariance when the filter is LOST (long optical gap, or mass gate-out with no fresh PnP):
		//! inflate large so the chi-square gate admits the drifted LEDs and the joint fold re-solves the
		//! pose from them (an EKF-PnP), converging in 1-2 frames instead of limit-cycling.
		static constexpr double LOST_POS_VAR = sq(1.5);       // 1.5 m
		static constexpr double LOST_ORI_VAR = sq(1.0);       // 1 rad ~ 57 deg
		//! Initial covariance at bootstrap (std per axis).
		static constexpr double P0_POS = sq(0.2);             // 0.2 m
		static constexpr double P0_VEL = sq(1.0);             // 1 m/s
		static constexpr double P0_ORI = sq(0.3);             // 0.3 rad ~ 17 deg
		static constexpr double P0_BA = sq(0.2);              // 0.2 m/s^2 — accel bias initially unknown
		static constexpr double P0_BG = sq(0.063);            // 0.063 rad/s — learn the gyro bias fast
		                                                      // (avoids a slow gravity-leak velocity transient)
		//! Default optical pose-measurement noise (bootstrap/re-anchor reference only).
		static constexpr double OPT_POS_STD = 0.02;    // m
		static constexpr double OPT_ORI_STD = 0.02;    // rad
		//! Max horizon get_prediction extrapolates the snapshot forward. Bounds pathological large-dt
		//! extrapolation; never clips normal use (render queries are within a frame of the latest IMU).
		static constexpr double MAX_PREDICT_AHEAD_S = 0.1;
		//! Position dead-reckoning is trusted only this long after the last optical pose; past it
		//! get_prediction freezes the reported position.
		static constexpr int64_t OPTICAL_FREEZE_NS = ms_to_ns(500);
		//! Physical sanity caps on a single IMU sample (glitch rejection).
		static constexpr double MAX_GYRO_RAD_PER_SEC = deg2rad(7200);
		static constexpr double MAX_ACCEL_M_S2 = 16.0 * MATH_GRAVITY_M_S2; // 16 g
		//! Consecutive anomalous IMU samples before a real fault is declared (vs a lone glitch skipped).
		static constexpr int IMU_ANOMALY_RESET_RUN = 10;
		//! Optical jump plausibility gate on the PnP re-anchor candidate.
		static constexpr double OPTICAL_MAX_SPEED_M_S = 12.0;
		static constexpr double OPTICAL_JUMP_SLACK_M = 0.5;
		//! A PnP pose this far from the prediction means the filter has diverged (per-LED can't fix a
		//! large reprojection error — too nonlinear); snap-re-anchor to the PnP pose instead of a slow
		//! EKF nudge. Below it, apply the pose as a normal absolute EKF update.
		static constexpr double REANCHOR_SNAP_M = 0.5;
		//! Optical-orientation flip rejection (gyro arbitration). A few-LED PnP intermittently returns a
		//! ~90-180 deg mirror-flipped orientation; the 200 Hz gyro never rotated that far between frames.
		//! When the gyro is fresh (optical recent, so it has not drifted) and a candidate optical
		//! orientation disagrees with the gyro-propagated filter orientation by more than this, keep the
		//! gyro orientation (adopt optical POSITION only). ~75 deg between frames is >3000 deg/s —
		//! implausibly fast for a hand, so this rejects only flips, never real fast rotation. After a long
		//! gap (> FLIP_GUARD_TRUST_NS) the gyro is no longer trusted, so the optical orientation is
		//! accepted (sole ref).
		static constexpr double FLIP_REJECT_RAD = deg2rad(75);
		//! Gyro-trust horizon for flip arbitration — DECOUPLED from the 0.5 s position freeze. The gyro
		//! orientation drifts only slowly (bias ~0.06 rad/s, estimated online), so it stays a valid flip
		//! reference for seconds; optical dropouts are routinely 0.5–3 s (BT-limited controllers), and
		//! during them the few-LED PnP keeps proposing mirror flips. Trusting the gyro this long rejects
		//! those flips through the gap; past it (a true re-acquisition) the optical orientation is adopted.
		static constexpr int64_t FLIP_GUARD_TRUST_NS = ms_to_ns(3000);
		//! Hard physical sanity bound on an adopted optical position. An LED-constellation-tracked
		//! controller is within head-camera range of the HMD (~arm's reach); a pose hundreds of metres out
		//! is a degenerate few-blob PnP, never a real position. Reject such poses outright (don't fold,
		//! don't re-anchor) so a garbage solve can't snap the filter away. Generous (head-excursion +
		//! reach) so it never clips real motion — it only catches the math artifacts.
		static constexpr double MAX_PLAUSIBLE_POS_M = 4.0;
		//! Plausibility bounds for PERSISTING the cross-session IMU calibration. A real MEMS gyro/accel
		//! bias is far below these; a larger converged estimate means this session tracked poorly (e.g. a
		//! controller with daylight optical flips) and mis-attributed the error to bias — never persist
		//! that (it would poison the next session). Reject the whole calibration if any axis is implausible.
		static constexpr double GYRO_BIAS_MAX = 0.1;  //!< rad/s (~5.7 deg/s); real bias << this
		static constexpr double ACCEL_BIAS_MAX = 0.5; //!< m/s^2 (~5% g); real bias << this
		//! Full-ellipsoid accel calibration: collect a rest sample only when its gravity direction differs
		//! from every prior one by at least this, so the accumulated set spans orientations (the fit needs
		//! a 3D spread, and >=9 directions, to identify the 6-parameter shape matrix).
		static constexpr double ACCEL_CAL_DIR_SEP = deg2rad(15.0);
		//! If a per-LED fold happened within this window, process_pose is reference-only (bootstrap +
		//! re-anchor candidate) — folding the PnP pose too would double-count the same LEDs. With no
		//! recent per-LED fold (a pose-only caller), process_pose applies the pose as an absolute
		//! measurement, so the filter is robust to either input mode.
		static constexpr int64_t PER_LED_RECENT_NS = ms_to_ns(80);
		//! Cap on the out-of-sequence IMU replay buffer.
		static constexpr size_t IMU_LOG_CAP = 512;

		// ---- ESKF core ----
		NominalState m_x;          //!< nominal state
		Mat15 m_P{Mat15::Identity()}; //!< error-state covariance
		Vector3d m_accel_world{0, 0, 0};  //!< last world-frame acceleration (incl. gravity), for prediction
		Vector3d m_angvel_world{0, 0, 0}; //!< last world-frame angular velocity, for prediction

		timepoint_ns filter_time_ns{0};
		//! Timestamp of the last IMU sample that actually propagated. Lets propagate_to tell a tiny
		//! post-IMU sub-sample gap (use small IMU noise) from genuine pose-only operation (use the larger
		//! constant-velocity process noise) — otherwise the CV noise would jitter velocity on every fold.
		timepoint_ns m_last_imu_ns{0};
		bool tracked{false};
		TrackingInfo orientation_state;
		TrackingInfo position_state;
		int m_imu_anomaly_count{0};

		timepoint_ns last_optical_ns{0};
		Vector3d last_good_position{0, 0, 0};
		//! Time of the last successful per-LED fold; gates process_pose's measurement mode (see above).
		timepoint_ns m_last_led_fold_ns{0};

		//! Latest PnP pose (bootstrap + divergence re-anchor reference). Not fed as a steady-state
		//! measurement (that would double-count the LEDs the per-LED fold already uses).
		xrt_pose m_pnp_pose{};
		timepoint_ns m_pnp_ns{0};
		bool m_pnp_valid{false};

		//! Diagnostic (env G2_IMU_ONLY): pure-inertial mode. Optical is allowed only to bootstrap (lock at
		//! the true pose for G2_IMU_ONLY_BOOTSTRAP_S, default 3 s); thereafter ALL optical folds are
		//! suppressed and the filter runs on the IMU alone — to gauge the inertial odometry's smoothness in
		//! isolation from the matcher. (Pure-inertial position must drift; what we read is continuity.)
		bool m_imu_only{false};
		int64_t m_imu_only_bootstrap_ns{0};
		timepoint_ns m_imu_only_until_ns{0};

		//! Online accel-scale correction (nominal 1.0) tracked at rest; multiplies the bias-corrected
		//! specific force before integration. Persists across re-anchors (it's a calibration, not state).
		//! This is the single-orientation fallback; once enough rest orientations are seen it is superseded
		//! by m_accel_T (the full ellipsoid: per-axis scale + misalignment).
		double m_accel_scale{1.0};
		bool m_scale_bootstrapped{false}; //!< one-shot k = g/|a_m| set at the first detected stance
		int m_rest_count{0};              //!< consecutive at-rest IMU samples (gates ZUPT)

		//! Full accelerometer ellipsoid calibration (per-axis scale + misalignment), fit from at-rest
		//! samples spanning orientations. When valid it replaces the scalar scale: f = m_accel_T·(a_m-ba).
		Mat3 m_accel_T{Mat3::Identity()};
		bool m_accel_T_valid{false};
		std::vector<Vector3d> m_accel_cal_dirs; //!< distinct-orientation rest samples accumulated for the fit

		//! Cross-session IMU calibration prior (loaded from disk by the driver, keyed per controller serial).
		//! The converged gyro/accel bias + accel scale are quasi-constant per unit, so seeding the first
		//! bootstrap from history makes the first second (before any stance) accurate and survives a large
		//! physical bias that would otherwise defeat the stance gate. m_had_stance gates persisting back.
		Vector3d m_cached_bg{0, 0, 0}, m_cached_ba{0, 0, 0};
		double m_cached_scale{1.0};
		bool m_cal_primed{false}; //!< a prior was loaded and not yet consumed by a bootstrap
		bool m_had_stance{false}; //!< a calibrated stance occurred this session
		//! Snapshot of the bias/scale at the last PLAUSIBLE stance. Persisted (not the live end-state),
		//! so a session that later dead-reckons off (corrupting the live bias) still caches the good
		//! calibration captured while truly at rest.
		Vector3d m_stance_bg{0, 0, 0}, m_stance_ba{0, 0, 0};
		double m_stance_scale{1.0};
		bool m_stance_snap{false};

		// ---- Out-of-sequence (lagged) optical handling ----
		FilterCheckpoint m_anchor;
		bool m_anchor_valid{false};
		std::deque<ImuLogEntry> m_imu_log;

		//! Serialises the two writer threads (IMU ~200 Hz, optical ~60 Hz). get_prediction never takes
		//! it — that path is wait-free via the seqlock.
		std::mutex m_filter_lock;

		//! Seqlock-published snapshot for the ~1 kHz wait-free get_prediction reader.
		alignas(64) std::atomic<uint32_t> m_snapshot_seq{0};
		FilterSnapshot m_snapshot{};

		// ---- core ops ----
		void
		reset_filter();
		void
		reset_filter_and_imu();
		void
		bootstrap_from_pose(const xrt_pose &pose);
		void
		reanchor(const Vector3d &p, const Quaterniond &q);
		//! Gyro arbitration: return @p cand unless it is a likely optical mirror-flip — i.e. the gyro is
		//! fresh (optical recent) AND @p cand disagrees with the gyro-propagated orientation by more than
		//! FLIP_REJECT_RAD; in that case return the current (gyro) orientation so a flip cannot be adopted.
		Quaterniond
		flip_guard(const Quaterniond &cand) const;
		//! Fold an accel gravity-direction measurement (anchors roll+pitch) when the bias-corrected
		//! body accel magnitude is within GRAV_BAND of g, i.e. linear acceleration is small enough that
		//! the accel direction is the gravity direction. No-op otherwise. Caller holds m_filter_lock.
		void
		fold_gravity_tilt(const Vector3d &accel_body_corrected);
		//! Zero-velocity update (ZUPT): when the device is detected at rest (low gyro + |accel|~g,
		//! sustained), fold a velocity==0 measurement. Without optical, any velocity error otherwise
		//! persists and integrates into unbounded position drift; ZUPT bounds it between motions.
		void
		fold_zupt();
		//! Zero angular-rate update (ZARU): at detected rest the true angular rate is 0, so the measured
		//! gyro directly observes the gyro bias (all three axes incl. yaw). Caller holds m_filter_lock.
		void
		fold_zaru(const Vector3d &gyro_measured);
		void
		propagate_to(int64_t target_ns);
		bool
		integrate_imu_sample(const xrt_imu_sample &sample);
		//! Generic EKF measurement update: error += K r; Joseph-form covariance. Returns false on a
		//! non-finite result (caller re-anchors). H is m x 15, r is m, R is m x m.
		bool
		ekf_update(const MatX &H, const VecX &r, const MatX &R);
		void
		inject(const Vec15 &dx);
		//! Apply a PnP absolute pose as a 6-DOF EKF measurement (pose-only input mode / no recent
		//! per-LED). Jump-gates an implausible candidate (reject) and re-anchors on a gross residual
		//! (never reset to origin). Returns false on a non-finite update.
		bool
		integrate_pose_measurement(const xrt_pose &pose,
		                           const Vector3d &pos_variance,
		                           const Vector3d &orient_variance,
		                           double residual_limit);
		int
		fold_led_observations(const std::vector<LEDObservation> &obs,
		                      const LEDCameraView &view,
		                      const struct xrt_vec2 *pixel_variance,
		                      float max_innov_px,
		                      int *out_seen);
		bool
		apply_optical_at(timepoint_ns t_pose, const std::function<void()> &apply);
		FilterCheckpoint
		capture_checkpoint() const;
		void
		restore_checkpoint(const FilterCheckpoint &c);
		void
		publish_snapshot();
		FilterSnapshot
		read_snapshot() const;

		//! Recording
		std::string m_device_name;
		bool m_recording = false;
		struct u_var_button m_recording_btn;
		ImuPoseRecorder *m_recorder = nullptr;

		static void
		recorder_btn_cb(void *ptr)
		{
			EskfFusion *self = static_cast<EskfFusion *>(ptr);
			if (self->m_recording) {
				self->m_recorder->stop();
				(void)snprintf(self->m_recording_btn.label, sizeof(self->m_recording_btn.label),
				               "Record dataset");
				self->m_recording = false;
			} else {
				self->m_recorder->start();
				(void)snprintf(self->m_recording_btn.label, sizeof(self->m_recording_btn.label),
				               "Stop recording");
				self->m_recording = true;
			}
		}
	};

	// ---------------------------------------------------------------------------
	// Reset / checkpoint / snapshot
	// ---------------------------------------------------------------------------

	void
	EskfFusion::reset_filter()
	{
		m_x = NominalState{};
		m_P.setZero();
		m_P.block<3, 3>(EP, EP) = Mat3::Identity() * P0_POS;
		m_P.block<3, 3>(EV, EV) = Mat3::Identity() * P0_VEL;
		m_P.block<3, 3>(ET, ET) = Mat3::Identity() * P0_ORI;
		m_P.block<3, 3>(EBA, EBA) = Mat3::Identity() * P0_BA;
		m_P.block<3, 3>(EBG, EBG) = Mat3::Identity() * P0_BG;
		m_accel_world.setZero();
		m_angvel_world.setZero();
		tracked = false;
		position_state = TrackingInfo{};
	}

	void
	EskfFusion::reset_filter_and_imu()
	{
		reset_filter();
		orientation_state = TrackingInfo{};
		m_imu_anomaly_count = 0;
		m_anchor_valid = false;
		m_imu_log.clear();
	}

	void
	EskfFusion::bootstrap_from_pose(const xrt_pose &pose)
	{
		const Vector3d keep_bg = m_x.bg; // preserve a learned bias across a re-acquire (bias is physical,
		const Vector3d keep_ba = m_x.ba; // it does not vanish on tracking loss)
		m_x = NominalState{};
		m_x.p = map_vec3(pose.position).cast<double>();
		m_x.q = map_quat(pose.orientation).cast<double>().normalized();
		if (m_cal_primed) {
			// First bootstrap: seed from the persisted cross-session prior (consumed once).
			m_x.bg = m_cached_bg;
			m_x.ba = m_cached_ba;
			m_accel_scale = m_cached_scale;
			m_scale_bootstrapped = true; // the prior beats the first-sample estimate; don't re-bootstrap
			m_cal_primed = false;
		} else {
			m_x.bg = keep_bg;
			m_x.ba = keep_ba;
		}
		m_P.setZero();
		m_P.block<3, 3>(EP, EP) = Mat3::Identity() * P0_POS;
		m_P.block<3, 3>(EV, EV) = Mat3::Identity() * P0_VEL;
		m_P.block<3, 3>(ET, ET) = Mat3::Identity() * P0_ORI;
		m_P.block<3, 3>(EBA, EBA) = Mat3::Identity() * P0_BA;
		m_P.block<3, 3>(EBG, EBG) = Mat3::Identity() * P0_BG;
		m_accel_world.setZero();
		m_angvel_world.setZero();
		tracked = true;
		position_state.valid = position_state.tracked = true;
		orientation_state.valid = orientation_state.tracked = true;
		last_good_position = m_x.p;
	}

	//! Snap pose to the PnP estimate, keep velocity + biases, inflate P so the chi-square gate widens
	//! and the LEDs re-enter on the next frame. Never resets to origin.
	void
	EskfFusion::reanchor(const Vector3d &p, const Quaterniond &q)
	{
		m_x.p = p;
		m_x.q = q.normalized();
		// A runaway velocity is what diverged us; drop it and re-open its uncertainty so the per-LED
		// folds rebuild it cleanly, rather than re-injecting the bad value.
		m_x.v.setZero();
		// Inflate position/orientation/velocity uncertainty; clear their cross-covariances so the
		// inflated blocks are not fought by stale correlations.
		for (int b : {EP, EV, ET}) {
			m_P.block<3, 15>(b, 0).setZero();
			m_P.block<15, 3>(0, b).setZero();
		}
		m_P.block<3, 3>(EP, EP) = Mat3::Identity() * REANCHOR_POS_VAR;
		m_P.block<3, 3>(EV, EV) = Mat3::Identity() * P0_VEL;
		m_P.block<3, 3>(ET, ET) = Mat3::Identity() * REANCHOR_ORI_VAR;
		last_good_position = p;
	}

	Quaterniond
	EskfFusion::flip_guard(const Quaterniond &cand) const
	{
		// Gyro untrustworthy only after a LONG optical gap (FLIP_GUARD_TRUST_NS, not the 0.5 s position
		// freeze) — until then it stays a valid flip reference through dropouts. Past it -> accept the
		// optical orientation as the sole reference. Otherwise compare against the gyro-propagated state.
		const bool gyro_fresh = last_optical_ns != 0 && (filter_time_ns - last_optical_ns) < FLIP_GUARD_TRUST_NS;
		if (!gyro_fresh) {
			return cand;
		}
		const double ang = log_quat(cand.normalized() * m_x.q.conjugate()).norm(); // rad between cand & gyro
		return (ang > FLIP_REJECT_RAD) ? m_x.q : cand; // a flip -> keep gyro orientation
	}

	void
	EskfFusion::fold_gravity_tilt(const Vector3d &f_body)
	{
		const double fmag = f_body.norm();
		if (!std::isfinite(fmag) || std::abs(fmag - MATH_GRAVITY_M_S2) > GRAV_BAND) {
			return; // accelerating (or non-finite): accel direction is not gravity -> skip
		}
		if (m_x.v.norm() > GRAV_MAX_VEL) {
			return; // moving: a horizontal accel masquerades as gravity here -> skip (trust gyro)
		}
		// Measurement: the unit accel direction (body) equals the world "up" rotated into body. At rest
		// f = R^T (a_world - g) = R^T(0,+g,0); so f/|f| = R^T up_w. Constrains roll+pitch (delta-theta),
		// not yaw/position. Global-error Jacobian: d(R^T up_w)/d(theta) = R^T [up_w]_x.
		const Mat3 R = m_x.q.toRotationMatrix();
		const Vector3d up_w(0.0, 1.0, 0.0);
		const Vector3d z = f_body / fmag;
		const Vector3d h = R.transpose() * up_w;
		MatX H = MatX::Zero(3, 15);
		H.block<3, 3>(0, ET) = R.transpose() * skew(up_w);
		const VecX r = z - h;
		const MatX Rm = Mat3::Identity() * GRAV_VAR;
		(void)ekf_update(H, r, Rm); // tilt-only correction; a non-finite result is harmless (skipped)
	}

	void
	EskfFusion::fold_zupt()
	{
		// Velocity == 0 measurement (3-DOF). Innovation r = 0 - v. Pins residual velocity to zero at rest
		// so it cannot integrate into position drift between motions. Caller holds m_filter_lock.
		MatX H = MatX::Zero(3, 15);
		H.block<3, 3>(0, EV) = Mat3::Identity();
		const VecX r = -m_x.v;
		const MatX Rm = Mat3::Identity() * ZUPT_VAR;
		(void)ekf_update(H, r, Rm);
	}

	void
	EskfFusion::fold_zaru(const Vector3d &gyro_measured)
	{
		// Gyro bias measurement: at rest the true rate is 0, so z = gyro_measured, h = bg, r = z - bg.
		// Observes all three gyro-bias axes (incl. yaw, unobservable from gravity). Caller holds the lock.
		MatX H = MatX::Zero(3, 15);
		H.block<3, 3>(0, EBG) = Mat3::Identity();
		const VecX r = gyro_measured - m_x.bg;
		const MatX Rm = Mat3::Identity() * ZARU_VAR;
		(void)ekf_update(H, r, Rm);
	}

	FilterCheckpoint
	EskfFusion::capture_checkpoint() const
	{
		FilterCheckpoint c;
		c.nominal = m_x;
		c.P = m_P;
		c.accel_world = m_accel_world;
		c.angvel_world = m_angvel_world;
		c.last_good_position = last_good_position;
		c.filter_time_ns = filter_time_ns;
		c.last_imu_ns = m_last_imu_ns;
		c.last_optical_ns = last_optical_ns;
		c.last_led_fold_ns = m_last_led_fold_ns;
		c.orientation_state = orientation_state;
		c.position_state = position_state;
		c.imu_anomaly_count = m_imu_anomaly_count;
		c.tracked = tracked;
		return c;
	}

	void
	EskfFusion::restore_checkpoint(const FilterCheckpoint &c)
	{
		m_x = c.nominal;
		m_P = c.P;
		m_accel_world = c.accel_world;
		m_angvel_world = c.angvel_world;
		last_good_position = c.last_good_position;
		filter_time_ns = c.filter_time_ns;
		m_last_imu_ns = c.last_imu_ns;
		last_optical_ns = c.last_optical_ns;
		m_last_led_fold_ns = c.last_led_fold_ns;
		orientation_state = c.orientation_state;
		position_state = c.position_state;
		m_imu_anomaly_count = c.imu_anomaly_count;
		tracked = c.tracked;
	}

	void
	EskfFusion::publish_snapshot()
	{
		FilterSnapshot s;
		Eigen::Map<Vector3d>{s.position} = m_x.p;
		Eigen::Map<Quaterniond>{s.orientation} = m_x.q;
		Eigen::Map<Vector3d>{s.linear_velocity} = m_x.v;
		Eigen::Map<Vector3d>{s.angular_velocity} = m_angvel_world;
		Eigen::Map<Vector3d>{s.acceleration} = m_accel_world;
		Eigen::Map<Vector3d>{s.last_good_position} = last_good_position;
		// Worst-direction (largest-eigenvalue) variance, so a consumer's isotropic n-sigma bound is
		// conservative in EVERY frame (rotating P can move variance up to its max eigenvalue).
		s.position_var_max =
		    Eigen::SelfAdjointEigenSolver<Mat3>(m_P.block<3, 3>(EP, EP), Eigen::EigenvaluesOnly)
		        .eigenvalues()
		        .maxCoeff();
		s.orientation_var_max =
		    Eigen::SelfAdjointEigenSolver<Mat3>(m_P.block<3, 3>(ET, ET), Eigen::EigenvaluesOnly)
		        .eigenvalues()
		        .maxCoeff();
		s.filter_time_ns = filter_time_ns;
		s.last_optical_ns = last_optical_ns;
		s.tracked = tracked;
		s.position_valid = position_state.valid;
		s.position_tracked = position_state.tracked;
		s.orientation_valid = orientation_state.valid;
		s.orientation_tracked = orientation_state.tracked;

		uint32_t seq = m_snapshot_seq.load(std::memory_order_relaxed);
		m_snapshot_seq.store(seq + 1, std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_release);
		m_snapshot = s;
		std::atomic_thread_fence(std::memory_order_release);
		m_snapshot_seq.store(seq + 2, std::memory_order_relaxed);
	}

	FilterSnapshot
	EskfFusion::read_snapshot() const
	{
		FilterSnapshot s;
		uint32_t seq1, seq2;
		do {
			seq1 = m_snapshot_seq.load(std::memory_order_acquire);
			while (seq1 & 1u) {
				seq1 = m_snapshot_seq.load(std::memory_order_acquire);
			}
			s = m_snapshot;
			std::atomic_thread_fence(std::memory_order_acquire);
			seq2 = m_snapshot_seq.load(std::memory_order_relaxed);
		} while (seq1 != seq2);
		return s;
	}

	void
	EskfFusion::clear_position_tracked_flag()
	{
		std::lock_guard<std::mutex> lock(m_filter_lock);
		position_state.tracked = false;
		publish_snapshot();
	}

	// ---------------------------------------------------------------------------
	// IMU propagation (nominal + error covariance)
	// ---------------------------------------------------------------------------

	void
	EskfFusion::propagate_to(int64_t target_ns)
	{
		if (filter_time_ns == 0) {
			filter_time_ns = target_ns;
			return;
		}
		if (target_ns <= filter_time_ns) {
			return; // the clock already covers this time
		}
		double dt = time_ns_to_s(target_ns - filter_time_ns);
		const bool post_imu = (m_last_imu_ns != 0 && filter_time_ns == m_last_imu_ns);
		filter_time_ns = target_ns;
		if (dt > 0.2) {
			dt = 0.2;
		}
		// Constant-velocity nominal prediction (no IMU input at this advance): position integrates
		// velocity; orientation/biases hold. Covariance grows so P never collapses. A tiny gap right
		// after an IMU sample uses the small IMU spectral density (the IMU already informed the accel);
		// genuine pose-only operation uses the larger CV white-noise-acceleration PSD.
		m_x.p += m_x.v * dt;

		const double acc_psd = post_imu ? (SIGMA_A * SIGMA_A) : (PROC_ACCEL_CV * PROC_ACCEL_CV);
		const double gyr_psd = post_imu ? (SIGMA_G * SIGMA_G) : (PROC_GYRO_CV * PROC_GYRO_CV);
		Mat15 F = Mat15::Identity();
		F.block<3, 3>(EP, EV) = Mat3::Identity() * dt;
		Mat15 Q = Mat15::Zero();
		const double d2 = dt * dt, d3 = d2 * dt;
		Q.block<3, 3>(EP, EP) = Mat3::Identity() * (acc_psd * d3 / 3.0);
		Q.block<3, 3>(EP, EV) = Mat3::Identity() * (acc_psd * d2 / 2.0);
		Q.block<3, 3>(EV, EP) = Mat3::Identity() * (acc_psd * d2 / 2.0);
		Q.block<3, 3>(EV, EV) = Mat3::Identity() * (acc_psd * dt);
		Q.block<3, 3>(ET, ET) = Mat3::Identity() * (gyr_psd * dt);
		Q.block<3, 3>(EBA, EBA) = Mat3::Identity() * (SIGMA_BA * SIGMA_BA * dt);
		Q.block<3, 3>(EBG, EBG) = Mat3::Identity() * (SIGMA_BG * SIGMA_BG * dt);
		m_P = F * m_P * F.transpose() + Q;
		m_P = 0.5 * (m_P + m_P.transpose()).eval();
	}

	bool
	EskfFusion::integrate_imu_sample(const xrt_imu_sample &sample)
	{
		const Vector3d G = Vector3d::UnitY() * -MATH_GRAVITY_M_S2;
		Vector3d a_m = map_vec3_f64(sample.accel_m_s2);
		Vector3d w_m = map_vec3_f64(sample.gyro_rad_secs);

		// Glitch rejection: a consumer MEMS controller IMU occasionally emits a single corrupt sample.
		// Drop a lone outlier and keep tracking; only a sustained run forces a reset.
		const bool gyro_outlier =
		    !w_m.allFinite() || w_m.squaredNorm() > MAX_GYRO_RAD_PER_SEC * MAX_GYRO_RAD_PER_SEC;
		const bool accel_outlier =
		    !a_m.allFinite() || a_m.squaredNorm() > MAX_ACCEL_M_S2 * MAX_ACCEL_M_S2;
		if (gyro_outlier || accel_outlier) {
			if (++m_imu_anomaly_count >= IMU_ANOMALY_RESET_RUN) {
				U_LOG_E("Sustained anomalous IMU samples (%d) - resetting filter", m_imu_anomaly_count);
				reset_filter_and_imu();
				m_imu_anomaly_count = 0;
			} else {
				U_LOG_W("Rejected outlier IMU sample (glitch %d/%d), keeping filter",
				        m_imu_anomaly_count, IMU_ANOMALY_RESET_RUN);
			}
			return true;
		}
		m_imu_anomaly_count = 0;

		double dt = time_ns_to_s(sample.timestamp_ns - filter_time_ns);
		filter_time_ns = sample.timestamp_ns;
		if (dt <= 0.0) {
			return true; // duplicate/again-stamped sample; clock already covers it
		}
		m_last_imu_ns = sample.timestamp_ns; // a real IMU propagation just happened (post_imu marker)
		if (dt > 0.1) {
			dt = 0.1; // bound a long gap so a single step cannot inject a huge process update
		}

		const Mat3 R = m_x.q.toRotationMatrix();
		const Vector3d w = w_m - m_x.bg;       // body angular rate, bias-corrected
		const double am_mag = a_m.norm();      // RAW accel magnitude (bias-independent for scale)

		// Online accel-scale (k = g/|a_m| at stance): bootstrap once on the first low-rotation sample, so
		// the stance band below is valid despite the initial 2-3% gain error; then slow-adapt at stance.
		const bool gyro_rest = w.norm() < ZUPT_GYRO_MAX;
		auto clamp_scale = [](double k) { return std::min(std::max(k, SCALE_MIN), SCALE_MAX); };
		if (gyro_rest && am_mag > 1.0 && !m_scale_bootstrapped) {
			m_accel_scale = clamp_scale(MATH_GRAVITY_M_S2 / am_mag);
			m_scale_bootstrapped = true;
		}
		// Stance: no rotation AND scale-corrected |accel| within ~3 sigma of g (rejects real linear accel).
		const bool at_rest =
		    gyro_rest && std::abs(am_mag * m_accel_scale - MATH_GRAVITY_M_S2) < ZUPT_ACCEL_BAND;
		if (at_rest && am_mag > 1.0) {
			m_accel_scale = clamp_scale(m_accel_scale + SCALE_ALPHA * (MATH_GRAVITY_M_S2 / am_mag - m_accel_scale));
			m_rest_count++;
		} else {
			m_rest_count = 0;
		}

		// Full-ellipsoid accel calibration: accumulate at-rest samples across orientations, then fit the
		// per-axis scale + misalignment matrix once the directions span 3D. Gated on no-rotation + |accel|
		// within the LOOSE gravity band (not the tight stance band): the per-axis scale being solved makes
		// |accel| vary a few % between orientations, and the tight band would reject exactly that variety.
		// On success it supersedes the scalar scale. (The factory mix_matrix is applied upstream; this is
		// the residual fit on top of it.)
		if (!m_accel_T_valid && gyro_rest && am_mag > 1.0 &&
		    std::abs(am_mag - MATH_GRAVITY_M_S2) < GRAV_BAND) {
			const Vector3d vcal = a_m - m_x.ba;
			bool distinct = true;
			for (const Vector3d &s : m_accel_cal_dirs) {
				if (vcal.normalized().dot(s.normalized()) > std::cos(ACCEL_CAL_DIR_SEP)) {
					distinct = false;
					break;
				}
			}
			if (distinct && m_accel_cal_dirs.size() < 64) {
				m_accel_cal_dirs.push_back(vcal);
				Mat3 T;
				if (fit_accel_calibration(m_accel_cal_dirs, MATH_GRAVITY_M_S2, T)) {
					m_accel_T = T;
					m_accel_T_valid = true;
					U_LOG_I("ESKF: full accel ellipsoid calibration fitted from %zu rest orientations",
					        m_accel_cal_dirs.size());
				}
			}
		}

		// Bias- and scale-corrected specific force. Use the full ellipsoid matrix once fitted (per-axis
		// scale + misalignment); until then the scalar scale (single-orientation) is the fallback.
		const Mat3 T_a = m_accel_T_valid ? m_accel_T : (m_accel_scale * Mat3::Identity());
		const Vector3d f = T_a * (a_m - m_x.ba);
		const Vector3d a_world = R * f + G;    // world acceleration (incl. gravity)

		// --- error-state covariance propagation (global angular error) ---
		Mat15 F = Mat15::Zero();
		F.block<3, 3>(EP, EV) = Mat3::Identity();
		F.block<3, 3>(EV, ET) = -skew(R * f);
		F.block<3, 3>(EV, EBA) = -R * T_a; // a_world = R*T_a*(a_m-ba)+G -> d/d(ba) = -R*T_a
		F.block<3, 3>(ET, EBG) = -R;
		const Mat15 Phi = Mat15::Identity() + F * dt;

		Mat15 Q = Mat15::Zero();
		// G Qc G^T dt, with G mapping [n_a, n_g, n_ba, n_bg]; R R^T = I so accel/gyro white land as I.
		Q.block<3, 3>(EV, EV) = Mat3::Identity() * (SIGMA_A * SIGMA_A) * dt;
		Q.block<3, 3>(ET, ET) = Mat3::Identity() * (SIGMA_G * SIGMA_G) * dt;
		Q.block<3, 3>(EBA, EBA) = Mat3::Identity() * (SIGMA_BA * SIGMA_BA) * dt;
		Q.block<3, 3>(EBG, EBG) = Mat3::Identity() * (SIGMA_BG * SIGMA_BG) * dt;

		m_P = Phi * m_P * Phi.transpose() + Q;
		m_P = 0.5 * (m_P + m_P.transpose()).eval();

		// --- nominal integration ---
		m_x.p += m_x.v * dt + 0.5 * a_world * dt * dt;
		m_x.v += a_world * dt;
		m_x.q = (m_x.q * exp_quat(w * dt)).normalized();

		m_accel_world = a_world;
		m_angvel_world = R * w;

		// Accel gravity-tilt anchor: when this sample shows low linear acceleration, fold the gravity
		// direction to pin roll+pitch absolutely (covers optical-sparse / flip stretches; see GRAV_BAND).
		fold_gravity_tilt(f);

		// Stance updates once rest is sustained. ZARU (gyro-bias from the measured rate) fires ALWAYS —
		// it's a pure bias calibration consistent with any optical. ZUPT (velocity=0) fires only while
		// optical is stale (dead-reckon regime), so it substitutes for — never fights — a live optical
		// velocity estimate. Together with the gravity anchor (roll/pitch), this fully calibrates the IMU
		// at rest: bg (ZARU), ba+velocity (ZUPT), tilt (gravity), scale (above).
		if (m_rest_count >= ZUPT_MIN_REST) {
			fold_zaru(w_m);
			m_had_stance = true;
			// Snapshot the calibration whenever this stance's estimate is physically plausible — this is
			// what gets persisted, so later dead-reckon drift can't corrupt the cached value.
			if (m_x.bg.cwiseAbs().maxCoeff() < GYRO_BIAS_MAX && m_x.ba.cwiseAbs().maxCoeff() < ACCEL_BIAS_MAX) {
				m_stance_bg = m_x.bg;
				m_stance_ba = m_x.ba;
				m_stance_scale = m_accel_scale;
				m_stance_snap = true;
			}
			const bool optical_stale =
			    last_optical_ns == 0 || (filter_time_ns - last_optical_ns) > OPTICAL_FREEZE_NS;
			if (optical_stale) {
				fold_zupt();
			}
		}

		if (!m_x.p.allFinite() || !m_x.v.allFinite() || !std::isfinite(m_x.q.norm())) {
			U_LOG_E("Non-finite state after IMU integration - resetting");
			reset_filter_and_imu();
			return false;
		}

		// Velocity divergence watchdog: a hand-held controller never exceeds OPTICAL_MAX_SPEED_M_S. A
		// larger estimated speed means unaided dead-reckoning has diverged (no optical to bound it); clamp
		// it and re-open the velocity covariance so the reported pose can't fly off and optical re-anchors
		// hard on return. Physical sanity bound (cf. MAX_PLAUSIBLE_POS_M), not a tuning knob.
		const double speed = m_x.v.norm();
		if (speed > OPTICAL_MAX_SPEED_M_S) {
			m_x.v *= OPTICAL_MAX_SPEED_M_S / speed;
			m_P.block<3, 3>(EV, EV) += Mat3::Identity() * P0_VEL;
		}
		return true;
	}

	// ---------------------------------------------------------------------------
	// Generic EKF update + error injection
	// ---------------------------------------------------------------------------

	bool
	EskfFusion::ekf_update(const MatX &H, const VecX &r, const MatX &R)
	{
		const int m = (int)r.size();
		if (m == 0) {
			return true;
		}
		const MatX PHt = m_P * H.transpose();        // 15 x m
		const MatX S = H * PHt + R;                   // m x m
		const MatX Sinv = S.inverse();
		if (!Sinv.allFinite()) {
			return false;
		}
		const MatX K = PHt * Sinv;                    // 15 x m
		const Vec15 dx = K * r;
		if (!dx.allFinite()) {
			return false;
		}
		// Joseph form: numerically stable, stays positive-definite.
		const Mat15 IKH = Mat15::Identity() - K * H;
		m_P = IKH * m_P * IKH.transpose() + K * R * K.transpose();
		m_P = 0.5 * (m_P + m_P.transpose()).eval();
		inject(dx);
		return m_P.allFinite();
	}

	void
	EskfFusion::inject(const Vec15 &dx)
	{
		m_x.p += dx.segment<3>(EP);
		m_x.v += dx.segment<3>(EV);
		m_x.q = (exp_quat(dx.segment<3>(ET)) * m_x.q).normalized(); // global error: left-multiply
		m_x.ba += dx.segment<3>(EBA);
		m_x.bg += dx.segment<3>(EBG);
	}

	// ---------------------------------------------------------------------------
	// Per-LED reprojection update (the primary optical measurement)
	// ---------------------------------------------------------------------------

	int
	EskfFusion::fold_led_observations(const std::vector<LEDObservation> &obs,
	                                  const LEDCameraView &view,
	                                  const struct xrt_vec2 *pixel_variance,
	                                  float max_innov_px,
	                                  int *out_seen)
	{
		Vector2d px_var{LED_PIXEL_STD * LED_PIXEL_STD, LED_PIXEL_STD * LED_PIXEL_STD};
		if (pixel_variance != nullptr) {
			px_var = Vector2d{pixel_variance->x, pixel_variance->y};
		}
		const Quaterniond q_cw = map_quat(view.cam_world_orient).cast<double>().normalized();
		const Mat3 R_cw = q_cw.toRotationMatrix();
		const Vector3d t_cw = map_vec3(view.cam_world_pos).cast<double>();
		const Mat3 R = m_x.q.toRotationMatrix();
		const double max_innov_sq = (max_innov_px > 0.f) ? double(max_innov_px) * double(max_innov_px) : -1.0;

		// Re-acquisition after a long optical gap: the dead-reckoned velocity (and position) have
		// diverged on IMU alone. Drop the velocity and re-open position+velocity uncertainty so the
		// folds re-converge cleanly instead of fighting a stale runaway. (Past OPTICAL_FREEZE_NS we
		// already distrust dead-reckoned position; velocity is even less trustworthy.)
		if (last_optical_ns != 0 && (filter_time_ns - last_optical_ns) > OPTICAL_FREEZE_NS) {
			m_x.v.setZero();
			m_P.block<3, 3>(EV, EV) = Mat3::Identity() * P0_VEL;
			m_P.block<3, 3>(EP, EP) = Mat3::Identity() * LOST_POS_VAR;
			m_P.block<3, 3>(ET, ET) = Mat3::Identity() * LOST_ORI_VAR;
		}

		// Collect gated per-LED rows, then apply ONE joint Kalman update (correct cross-correlations).
		std::vector<Eigen::Matrix<double, 2, 15>> Hs;
		std::vector<Vector2d> rs;
		std::vector<Vector2d> Rs;
		int seen = 0;

		for (const LEDObservation &o : obs) {
			Vector2d z{o.observed_px.x, o.observed_px.y};
			Vector3d led_obj = map_vec3(o.led_obj).cast<double>();
			if (!z.allFinite() || !led_obj.allFinite()) {
				continue;
			}
			seen++;

			const Vector3d p_world = m_x.p + R * led_obj;
			const Vector3d p_cam = R_cw * p_world + t_cw;
			const double zc = (p_cam.z() > 1e-6) ? p_cam.z() : 1e-6;
			Vector2d zhat;
			zhat[0] = view.fx * (p_cam.x() / zc) + view.cx;
			zhat[1] = view.fy * (p_cam.y() / zc) + view.cy;
			const Vector2d innov = z - zhat;
			if (!innov.allFinite()) {
				continue;
			}

			// Hard mislabel reject (raw pixel innovation), independent of covariance.
			if (max_innov_sq > 0.0 && innov.squaredNorm() > max_innov_sq) {
				continue;
			}

			// Jacobian H = dpi/dp_cam * [ R_cw | 0 | -R_cw R [led_obj]_x | 0 | 0 ]   (global error)
			Eigen::Matrix<double, 2, 3> dpi;
			dpi << view.fx / zc, 0.0, -view.fx * p_cam.x() / (zc * zc), 0.0, view.fy / zc,
			    -view.fy * p_cam.y() / (zc * zc);
			Eigen::Matrix<double, 2, 15> H = Eigen::Matrix<double, 2, 15>::Zero();
			H.block<2, 3>(0, EP) = dpi * R_cw;
			H.block<2, 3>(0, ET) = dpi * (-R_cw * R * skew(led_obj));

			// Covariance-aware (chi-square) gate: S = H P H^T + R. Widens as P grows -> self-recovering.
			const Eigen::Matrix2d Rmeas = px_var.asDiagonal();
			const Eigen::Matrix2d S = H * m_P * H.transpose() + Rmeas;
			const Eigen::Matrix2d Sinv = S.inverse();
			if (!Sinv.allFinite()) {
				continue;
			}
			const double d2 = innov.transpose() * Sinv * innov;
			if (d2 > CHI2_GATE_2DOF) {
				continue; // outlier under current uncertainty
			}
			// Huber: inflate R for a borderline LED instead of cutting it.
			Vector2d r_eff = px_var;
			if (d2 > HUBER_DELTA2) {
				r_eff *= std::sqrt(d2 / HUBER_DELTA2);
			}
			Hs.push_back(H);
			rs.push_back(innov);
			Rs.push_back(r_eff);
		}

		if (out_seen != nullptr) {
			*out_seen = seen;
		}
		const int k = (int)Hs.size();
		if (k == 0) {
			return 0;
		}

		// Stack into one joint update.
		MatX H(2 * k, 15);
		VecX r(2 * k);
		MatX Rm = MatX::Zero(2 * k, 2 * k);
		for (int i = 0; i < k; i++) {
			H.block<2, 15>(2 * i, 0) = Hs[i];
			r.segment<2>(2 * i) = rs[i];
			Rm(2 * i, 2 * i) = Rs[i].x();
			Rm(2 * i + 1, 2 * i + 1) = Rs[i].y();
		}
		if (!ekf_update(H, r, Rm)) {
			U_LOG_E("Non-finite per-LED update - re-anchoring");
			const Vector3d pnp_pos = map_vec3(m_pnp_pose.position).cast<double>();
			if (m_pnp_valid && pnp_pos.allFinite() && pnp_pos.norm() < MAX_PLAUSIBLE_POS_M) {
				reanchor(pnp_pos, flip_guard(map_quat(m_pnp_pose.orientation).cast<double>()));
			} else {
				reset_filter();
			}
			return 0;
		}

		tracked = true;
		position_state.valid = position_state.tracked = true;
		orientation_state.valid = orientation_state.tracked = true;
		last_optical_ns = filter_time_ns;
		last_good_position = m_x.p;
		m_last_led_fold_ns = filter_time_ns;
		return k;
	}

	bool
	EskfFusion::integrate_pose_measurement(const xrt_pose &pose,
	                                       const Vector3d &pos_variance,
	                                       const Vector3d &orient_variance,
	                                       double residual_limit)
	{
		const Vector3d pos = map_vec3(pose.position).cast<double>();
		const Quaterniond orient = map_quat(pose.orientation).cast<double>().normalized();
		if (!pos.allFinite() || !std::isfinite(orient.norm())) {
			return true; // never feed a non-finite pose
		}
		if (pos.norm() >= MAX_PLAUSIBLE_POS_M) {
			U_LOG_W("Optical pose physically implausible (%.0f m) - degenerate PnP, rejecting", pos.norm());
			return true; // a constellation controller is never this far out; this is a degenerate solve
		}

		// Jump plausibility gate: reject (don't apply) a candidate that moved further from the last
		// optically-anchored position than a controller could physically travel in the elapsed time.
		if (tracked && last_optical_ns != 0) {
			double dt = time_ns_to_s(filter_time_ns - last_optical_ns);
			if (dt < 0.0) {
				dt = 0.0;
			}
			const double max_jump = OPTICAL_MAX_SPEED_M_S * dt + OPTICAL_JUMP_SLACK_M;
			if ((pos - last_good_position).norm() > max_jump) {
				U_LOG_W("Optical pose jumped implausibly far - rejecting candidate");
				return true;
			}
		}

		// Divergence: the prediction is far from the PnP pose. Per-LED folding cannot fix a large
		// reprojection error (the projection is too nonlinear there), so snap-re-anchor to the PnP pose
		// (never reset to origin), inflating P so the following per-LED folds re-converge. residual_limit
		// is the caller's looser catastrophic bound; REANCHOR_SNAP_M is the (tighter) divergence trigger.
		const double resid = (pos - m_x.p).norm();
		if (resid > REANCHOR_SNAP_M || resid > residual_limit) {
			reanchor(pos, flip_guard(orient)); // gyro-arbitrated: a flipped PnP re-anchors position only
			tracked = true;
			position_state.valid = position_state.tracked = true;
			orientation_state.valid = orientation_state.tracked = true;
			last_optical_ns = filter_time_ns;
			return true;
		}

		// Absolute pose measurement. Position is always folded; the optical ORIENTATION is folded only
		// if it agrees with the gyro-propagated orientation — a fresh-gyro disagreement past
		// FLIP_REJECT_RAD is a PnP mirror flip, so we do a position-only update and keep gyro orientation.
		const Vector3d dpos = pos - m_x.p;
		const Vector3d dtheta = log_quat(orient * m_x.q.conjugate());
		const bool gyro_fresh =
		    last_optical_ns != 0 && (filter_time_ns - last_optical_ns) < FLIP_GUARD_TRUST_NS;
		const bool ori_flip = gyro_fresh && dtheta.norm() > FLIP_REJECT_RAD;

		bool ok;
		if (ori_flip) {
			MatX H = MatX::Zero(3, 15);
			H.block<3, 3>(0, EP) = Mat3::Identity();
			MatX R = MatX(pos_variance.asDiagonal());
			ok = ekf_update(H, dpos, R);
		} else {
			MatX H = MatX::Zero(6, 15);
			H.block<3, 3>(0, EP) = Mat3::Identity();
			H.block<3, 3>(3, ET) = Mat3::Identity();
			VecX r(6);
			r.segment<3>(0) = dpos;
			r.segment<3>(3) = dtheta;
			VecX rd(6);
			rd << pos_variance, orient_variance;
			MatX R = MatX(rd.asDiagonal());
			ok = ekf_update(H, r, R);
		}
		if (!ok) {
			reanchor(pos, flip_guard(orient));
			tracked = true;
			position_state.valid = position_state.tracked = true;
			orientation_state.valid = orientation_state.tracked = true;
			last_optical_ns = filter_time_ns;
			return false;
		}
		tracked = true;
		position_state.valid = position_state.tracked = true;
		orientation_state.valid = orientation_state.tracked = true;
		last_optical_ns = filter_time_ns;
		last_good_position = m_x.p;
		return true;
	}

	// ---------------------------------------------------------------------------
	// Out-of-sequence (lagged) optical: rewind to anchor, replay IMU, apply, replay forward.
	// ---------------------------------------------------------------------------

	bool
	EskfFusion::apply_optical_at(timepoint_ns t_pose, const std::function<void()> &apply)
	{
		if (!m_anchor_valid || !tracked || filter_time_ns == 0 || t_pose >= filter_time_ns) {
			propagate_to(t_pose);
			apply();
			m_anchor = capture_checkpoint();
			m_anchor_valid = true;
			m_imu_log.clear();
			return true;
		}
		if (t_pose < m_anchor.filter_time_ns) {
			return false; // superseded / older than the rewind horizon
		}

		restore_checkpoint(m_anchor);
		for (const ImuLogEntry &e : m_imu_log) {
			if (e.sample.timestamp_ns > t_pose) {
				break;
			}
			integrate_imu_sample(e.sample);
		}
		propagate_to(t_pose);
		apply();

		m_anchor = capture_checkpoint();
		while (!m_imu_log.empty() && m_imu_log.front().sample.timestamp_ns <= t_pose) {
			m_imu_log.pop_front();
		}
		for (const ImuLogEntry &e : m_imu_log) {
			integrate_imu_sample(e.sample);
		}
		return true;
	}

	// ---------------------------------------------------------------------------
	// Public interface
	// ---------------------------------------------------------------------------

	void
	EskfFusion::process_imu_data(const struct xrt_imu_sample *sample,
	                             const struct xrt_vec3 * /*accel_variance_optional*/,
	                             const struct xrt_vec3 * /*gyro_variance_optional*/)
	{
		// The ESKF treats IMU as a propagation INPUT (process noise = SIGMA_A/SIGMA_G spectral
		// densities), not as a measurement, so the caller's optional measurement variances do not apply.
		{
			std::lock_guard<std::mutex> lock(m_filter_lock);
			if (tracked) {
				integrate_imu_sample(*sample);
				m_imu_log.push_back(ImuLogEntry{*sample});
				if (m_imu_log.size() > IMU_LOG_CAP) {
					m_anchor = capture_checkpoint();
					m_anchor_valid = true;
					m_imu_log.clear();
				}
				publish_snapshot();
			}
		}
		if (m_recorder) {
			m_recorder->process_imu_data(sample);
		}
	}

	void
	EskfFusion::process_pose(const struct xrt_pose_sample *sample,
	                         const struct xrt_vec3 *position_variance_optional,
	                         const struct xrt_vec3 *orientation_variance_optional,
	                         float residual_limit)
	{
		Vector3d pos_var{OPT_POS_STD * OPT_POS_STD, OPT_POS_STD * OPT_POS_STD, OPT_POS_STD * OPT_POS_STD};
		Vector3d ori_var{OPT_ORI_STD * OPT_ORI_STD, OPT_ORI_STD * OPT_ORI_STD, OPT_ORI_STD * OPT_ORI_STD};
		if (position_variance_optional) {
			pos_var = map_vec3(*position_variance_optional).cast<double>();
		}
		if (orientation_variance_optional) {
			ori_var = map_vec3(*orientation_variance_optional).cast<double>();
		}

		const Vector3d pos = map_vec3(sample->pose.position).cast<double>();
		const Quaterniond orient = map_quat(sample->pose.orientation).cast<double>();
		const bool finite = pos.allFinite() && std::isfinite(orient.norm());

		{
			std::lock_guard<std::mutex> lock(m_filter_lock);
			if (finite) {
				// Always keep the PnP pose as the bootstrap / divergence re-anchor reference.
				m_pnp_pose = sample->pose;
				m_pnp_ns = sample->timestamp_ns;
				m_pnp_valid = true;

				if (!tracked) {
					// Bootstrap only from a PLAUSIBLE pose. A constellation-tracked controller is within
					// arm's reach, so a degenerate few-blob PnP hundreds of metres out must not seed the
					// filter — it would capture last_good_position at an impossible point and be reported
					// (frozen) there. Mirrors the plausibility gate at every other optical-adoption site;
					// stay untracked until a real pose arrives. (Found via real-data replay: a session
					// whose only optical solves were ~33 m degenerates used to bootstrap at 33 m.)
					if (pos.norm() < MAX_PLAUSIBLE_POS_M) {
						apply_optical_at(sample->timestamp_ns,
						                 [&]() { bootstrap_from_pose(sample->pose); });
						if (m_imu_only && m_imu_only_until_ns == 0) {
							m_imu_only_until_ns = sample->timestamp_ns + m_imu_only_bootstrap_ns;
						}
					}
				} else if (m_imu_only && sample->timestamp_ns > m_imu_only_until_ns) {
					// Pure-inertial diagnostic: bootstrap window elapsed -> ignore optical entirely.
				} else {
					// Dual mode: with per-LED actively folding (live G2 path), the per-LED update IS the
					// measurement — feeding the PnP pose too would double-count. With no recent per-LED
					// fold (a pose-only caller), apply the pose as an absolute EKF measurement so the
					// filter tracks either input mode.
					const bool per_led_active =
					    m_last_led_fold_ns != 0 &&
					    (sample->timestamp_ns - m_last_led_fold_ns) < PER_LED_RECENT_NS;
					if (!per_led_active) {
						apply_optical_at(sample->timestamp_ns, [&]() {
							integrate_pose_measurement(sample->pose, pos_var, ori_var,
							                           residual_limit);
						});
					}
				}
			}
			publish_snapshot();
		}
		if (m_recorder) {
			m_recorder->process_pose(sample);
		}
	}

	float
	EskfFusion::process_led_observations(const timepoint_ns timestamp_ns,
	                                     const std::vector<LEDObservation> &obs,
	                                     const LEDCameraView &view,
	                                     const struct xrt_vec2 *pixel_variance,
	                                     const float max_innov_px,
	                                     const bool feed)
	{
		// DIAGNOSTIC (feed == false): per-LED reprojection RMS against the current pose, read-only.
		if (!feed) {
			std::lock_guard<std::mutex> lock(m_filter_lock);
			if (!tracked) {
				return -1.0f;
			}
			const Quaterniond q_cw = map_quat(view.cam_world_orient).cast<double>().normalized();
			const Mat3 R_cw = q_cw.toRotationMatrix();
			const Vector3d t_cw = map_vec3(view.cam_world_pos).cast<double>();
			const Mat3 R = m_x.q.toRotationMatrix();
			double sum_sq = 0.0;
			int n = 0;
			for (const LEDObservation &o : obs) {
				Vector2d z{o.observed_px.x, o.observed_px.y};
				Vector3d led_obj = map_vec3(o.led_obj).cast<double>();
				if (!z.allFinite() || !led_obj.allFinite()) {
					continue;
				}
				const Vector3d p_cam = R_cw * (m_x.p + R * led_obj) + t_cw;
				const double zc = (p_cam.z() > 1e-6) ? p_cam.z() : 1e-6;
				Vector2d zhat{view.fx * (p_cam.x() / zc) + view.cx, view.fy * (p_cam.y() / zc) + view.cy};
				sum_sq += (z - zhat).squaredNorm();
				n++;
			}
			return n == 0 ? -1.0f : (float)std::sqrt(sum_sq / n);
		}

		// FEED: fold the LEDs at the true capture time (rewinding through buffered IMU when lagged).
		int folded = 0;
		int seen = 0;
		{
			std::lock_guard<std::mutex> lock(m_filter_lock);
			if (!tracked) {
				return -1.0f; // awaiting a process_pose bootstrap
			}
			if (m_imu_only && m_imu_only_until_ns != 0 && timestamp_ns > m_imu_only_until_ns) {
				return -1.0f; // pure-inertial diagnostic: optical fold suppressed
			}
			apply_optical_at(timestamp_ns, [&]() {
				folded = fold_led_observations(obs, view, pixel_variance, max_innov_px, &seen);

				// Divergence: many LEDs matched but few passed the gate => the prediction is wrong, not
				// the LEDs. Re-anchor to a fresh PnP pose (or just inflate P) so the gate widens and the
				// LEDs re-enter next frame. Re-fold once at the inflated covariance.
				if (seen >= REANCHOR_NMIN && folded < (int)std::ceil(REANCHOR_FRAC * seen)) {
					const Vector3d pnp_pos = map_vec3(m_pnp_pose.position).cast<double>();
					const bool pnp_fresh = m_pnp_valid &&
					    std::llabs((long long)(m_pnp_ns - timestamp_ns)) < REANCHOR_MAX_AGE_NS &&
					    pnp_pos.allFinite() && pnp_pos.norm() < MAX_PLAUSIBLE_POS_M;
					if (pnp_fresh) {
						reanchor(pnp_pos, flip_guard(map_quat(m_pnp_pose.orientation).cast<double>()));
					} else {
						// No fresh PnP: the filter is lost. Inflate P large so the gate admits the
						// drifted LEDs and the re-fold re-solves the pose from them; drop the runaway
						// velocity so it cannot fight the correction.
						m_x.v.setZero();
						m_P.block<3, 3>(EP, EP) = Mat3::Identity() * LOST_POS_VAR;
						m_P.block<3, 3>(ET, ET) = Mat3::Identity() * LOST_ORI_VAR;
						m_P.block<3, 3>(EV, EV) = Mat3::Identity() * P0_VEL;
					}
					int seen2 = 0;
					folded = fold_led_observations(obs, view, pixel_variance, max_innov_px, &seen2);
				}
			});
			publish_snapshot();
		}
		return (float)folded;
	}

	void
	EskfFusion::get_prediction(timepoint_ns when_ns, struct xrt_space_relation *out_relation)
	{
		if (out_relation == NULL) {
			return;
		}
		U_ZERO(out_relation);
		out_relation->pose.orientation.w = 1;

		const FilterSnapshot snap = read_snapshot();
		if (!snap.tracked || snap.filter_time_ns == 0) {
			return;
		}
		const Eigen::Map<const Vector3d> s_pos{snap.position};
		const Eigen::Map<const Vector3d> s_lvel{snap.linear_velocity};
		const Eigen::Map<const Vector3d> s_avel{snap.angular_velocity};
		const Eigen::Map<const Vector3d> s_acc{snap.acceleration};
		const Eigen::Map<const Vector3d> s_lgp{snap.last_good_position};
		const Eigen::Map<const Quaterniond> s_orient{snap.orientation};

		// Constant-acceleration extrapolation of the snapshot, horizon-bounded. No artificial damping:
		// the ESKF velocity is a directly estimated state, trustworthy over a render frame.
		double dt = time_ns_to_s(when_ns - snap.filter_time_ns);
		if (dt > MAX_PREDICT_AHEAD_S) {
			dt = MAX_PREDICT_AHEAD_S;
		} else if (dt < 0.0) {
			dt = 0.0;
		}
		const Vector3d vel = s_lvel + s_acc * dt;
		const Vector3d pos = s_pos + s_lvel * dt + 0.5 * s_acc * dt * dt;

		// Orientation advances by the world-frame rotation vector omega*dt (global error: left-multiply).
		const Vector3d rotvec = s_avel * dt;
		const double angle = rotvec.norm();
		Quaterniond orient = s_orient;
		if (angle > 1e-9) {
			orient = Quaterniond{AngleAxisd{angle, rotvec / angle}} * s_orient;
		}
		orient.normalize();

		// IMU position dead-reckoning is only good for a fraction of a second; once optical has been
		// absent past OPTICAL_FREEZE_NS, freeze the reported position. Gyro orientation lasts longer.
		// (G2_IMU_ONLY diagnostic deliberately keeps reporting the dead-reckoned position so we can SEE
		// the pure-inertial drift — m_imu_only is set once at construction, so this lock-free read is safe.)
		const bool optical_stale =
		    !m_imu_only &&
		    (snap.last_optical_ns == 0 || (when_ns - snap.last_optical_ns) > OPTICAL_FREEZE_NS);

		map_quat(out_relation->pose.orientation) = orient.cast<float>();
		map_vec3(out_relation->angular_velocity) = s_avel.cast<float>();
		if (optical_stale) {
			map_vec3(out_relation->pose.position) = s_lgp.cast<float>();
			out_relation->linear_velocity = xrt_vec3{0.f, 0.f, 0.f};
		} else {
			map_vec3(out_relation->pose.position) = pos.cast<float>();
			map_vec3(out_relation->linear_velocity) = vel.cast<float>();
		}

		uint64_t flags = 0;
		if (snap.position_valid) {
			flags |= XRT_SPACE_RELATION_POSITION_VALID_BIT;
			if (!optical_stale) {
				flags |= XRT_SPACE_RELATION_LINEAR_VELOCITY_VALID_BIT;
				if (snap.position_tracked) {
					flags |= XRT_SPACE_RELATION_POSITION_TRACKED_BIT;
				}
			}
		}
		if (snap.orientation_valid) {
			flags |= XRT_SPACE_RELATION_ORIENTATION_VALID_BIT;
			flags |= XRT_SPACE_RELATION_ANGULAR_VELOCITY_VALID_BIT;
			if (snap.orientation_tracked) {
				flags |= XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT;
			}
		}
		out_relation->relation_flags = (xrt_space_relation_flags)flags;

		if (m_recorder) {
			m_recorder->process_fused(when_ns, out_relation);
		}
	}

	bool
	EskfFusion::debug_get_position_covariance(double cov_row_major[9])
	{
		std::lock_guard<std::mutex> lock(m_filter_lock);
		const Mat3 Ppos = m_P.block<3, 3>(EP, EP);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				cov_row_major[i * 3 + j] = Ppos(i, j);
			}
		}
		return tracked;
	}

	void
	EskfFusion::add_ui(void *root, const char *device_name)
	{
		const char *record_path = debug_get_option_kalman_record_path();
		if (record_path != NULL) {
			m_recorder = new ImuPoseRecorder(record_path, device_name);
			m_recorder->start();
			m_recording = true;
			m_recording_btn.cb = EskfFusion::recorder_btn_cb;
			m_recording_btn.ptr = this;
			u_var_add_button(root, &m_recording_btn, "Stop recording");
		}
	}

} // namespace


std::unique_ptr<KalmanFusionInterface>
KalmanFusionInterface::create()
{
	return std::make_unique<EskfFusion>();
}


} // namespace xrt::auxiliary::tracking
