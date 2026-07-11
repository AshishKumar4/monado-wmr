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
#include "ransac_pnp.h"
#include "util/u_logging.h"
#include "math/m_api.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#if CV_MAJOR_VERSION >= 4
#include <opencv2/calib3d/calib3d_c.h>
#endif
#include <cmath>
#include <vector>

using namespace std;

static void
quat_to_3x3(cv::Mat &mat, struct xrt_quat *me)
{
	mat.at<double>(0, 0) = 1 - 2 * me->y * me->y - 2 * me->z * me->z;
	mat.at<double>(0, 1) = 2 * me->x * me->y - 2 * me->w * me->z;
	mat.at<double>(0, 2) = 2 * me->x * me->z + 2 * me->w * me->y;

	mat.at<double>(1, 0) = 2 * me->x * me->y + 2 * me->w * me->z;
	mat.at<double>(1, 1) = 1 - 2 * me->x * me->x - 2 * me->z * me->z;
	mat.at<double>(1, 2) = 2 * me->y * me->z - 2 * me->w * me->x;

	mat.at<double>(2, 0) = 2 * me->x * me->z - 2 * me->w * me->y;
	mat.at<double>(2, 1) = 2 * me->y * me->z + 2 * me->w * me->x;
	mat.at<double>(2, 2) = 1 - 2 * me->x * me->x - 2 * me->y * me->y;
}

//! Convert an OpenCV (rvec,tvec) to an xrt_pose (Rodrigues angle-axis -> quaternion).
static void
rtvec_to_pose(const cv::Mat &rvec, const cv::Mat &tvec, struct xrt_pose *pose)
{
	const double angle = std::sqrt(rvec.dot(rvec));
	struct xrt_vec3 axis = {0.f, 0.f, 1.f};
	if (angle > 1e-9) {
		const double inorm = 1.0 / angle;
		axis.x = (float)(rvec.at<double>(0) * inorm);
		axis.y = (float)(rvec.at<double>(1) * inorm);
		axis.z = (float)(rvec.at<double>(2) * inorm);
	}
	math_quat_from_angle_vector(angle, &axis, &pose->orientation);
	pose->position.x = (float)tvec.at<double>(0);
	pose->position.y = (float)tvec.at<double>(1);
	pose->position.z = (float)tvec.at<double>(2);
}

//! Geodesic angle (rad) between two rotations given as Rodrigues vectors.
static double
rot_angle_between(const cv::Mat &rvec_a, const cv::Mat &rvec_b)
{
	cv::Mat Ra, Rb;
	cv::Rodrigues(rvec_a, Ra);
	cv::Rodrigues(rvec_b, Rb);
	const cv::Mat Rd = Ra.t() * Rb;
	const double c = (Rd.at<double>(0, 0) + Rd.at<double>(1, 1) + Rd.at<double>(2, 2) - 1.0) * 0.5;
	return std::acos(std::min(1.0, std::max(-1.0, c)));
}

static bool
vec3_mat_is_finite(const cv::Mat &v)
{
	if (v.empty() || v.total() != 3 || v.depth() != CV_64F) {
		return false;
	}
	for (int r = 0; r < v.rows; r++) {
		for (int c = 0; c < v.cols; c++) {
			if (!std::isfinite(v.at<double>(r, c))) {
				return false;
			}
		}
	}
	return true;
}

static double
coplanarity_ratio(const std::vector<cv::Point3f> &pts)
{
	if (pts.size() < 4) {
		return 1.0;
	}
	cv::Point3d c(0, 0, 0);
	for (const cv::Point3f &p : pts) {
		c += cv::Point3d(p.x, p.y, p.z);
	}
	c *= 1.0 / (double)pts.size();
	cv::Mat m((int)pts.size(), 3, CV_64F);
	for (size_t i = 0; i < pts.size(); i++) {
		m.at<double>((int)i, 0) = pts[i].x - c.x;
		m.at<double>((int)i, 1) = pts[i].y - c.y;
		m.at<double>((int)i, 2) = pts[i].z - c.z;
	}
	cv::Mat w; // singular values, descending
	cv::SVD::compute(m, w, cv::SVD::NO_UV);
	const double s0 = w.at<double>(0), s2 = w.at<double>(2);
	return s0 > 1e-9 ? (s2 / s0) : 1.0;
}

#define MIRROR_TWIN_COPLANAR_RATIO 0.30
#define TILT_CLAMP_COPLANAR_RATIO 0.10

static bool
near_coplanar(const std::vector<cv::Point3f> &pts)
{
	return coplanarity_ratio(pts) < TILT_CLAMP_COPLANAR_RATIO;
}

static void
undistort_blob_points(std::vector<cv::Point2f> in_points,
                      std::vector<cv::Point2f> &out_points,
                      struct camera_model *calib)
{
	for (size_t i = 0; i < in_points.size(); i++) {
		t_camera_models_undistort(&calib->calib, in_points[i].x, in_points[i].y, &out_points[i].x,
		                          &out_points[i].y);
	}
}

/* vrserver statically links its own C++ EH runtime, so a cv::Exception thrown inside the live driver
 * unwinds past our catch blocks and aborts the process. These finite-input preconditions keep the
 * OpenCV solvers off their CV_Error paths; the catch blocks below only protect offline tools. */
static bool
points_finite(const std::vector<cv::Point3f> &p3d, const std::vector<cv::Point2f> &p2d)
{
	for (const cv::Point3f &p : p3d) {
		if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
			return false;
		}
	}
	for (const cv::Point2f &p : p2d) {
		if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
			return false;
		}
	}
	return true;
}

/* Re-gate then Levenberg-Marquardt refine: reproject (rvec,tvec) over all correspondences, keep the
 * ones within `thresh`, and LM-refine over that set in place. Returns the inlier count used. Shared by
 * the polish + re-gate passes so the project-filter-refine logic lives in one place. */
static int
refine_lm_over_inliers(const std::vector<cv::Point3f> &p3d,
                       const std::vector<cv::Point2f> &p2d,
                       const cv::Mat &K,
                       const cv::Mat &D,
                       cv::Mat &rvec,
                       cv::Mat &tvec,
                       double thresh,
                       std::vector<cv::Point3f> *in3d_out = nullptr,
                       std::vector<cv::Point2f> *in2d_out = nullptr)
{
	std::vector<cv::Point2f> proj;
	cv::projectPoints(p3d, rvec, tvec, K, D, proj);

	std::vector<cv::Point3f> in3d;
	std::vector<cv::Point2f> in2d;
	in3d.reserve(p3d.size());
	in2d.reserve(p2d.size());
	for (size_t k = 0; k < p3d.size(); k++) {
		const cv::Point2f d = proj[k] - p2d[k];
		if (std::hypot(d.x, d.y) <= thresh) {
			in3d.push_back(p3d[k]);
			in2d.push_back(p2d[k]);
		}
	}
	if (in3d.size() >= 4) {
		cv::solvePnPRefineLM(in3d, in2d, K, D, rvec, tvec);
	}
	if (in3d_out != nullptr) {
		*in3d_out = in3d;
	}
	if (in2d_out != nullptr) {
		*in2d_out = in2d;
	}
	return (int)in3d.size();
}

static bool
analytic_planar_twin(const std::vector<cv::Point3f> &in3d,
                     const cv::Mat &rvec_primary,
                     const cv::Mat &tvec_primary,
                     cv::Mat &rvec_twin,
                     cv::Mat &tvec_twin)
{
	if (in3d.size() < 4) {
		return false;
	}

	cv::Mat R_primary;
	cv::Rodrigues(rvec_primary, R_primary);

	cv::Point3d centroid(0, 0, 0);
	for (const cv::Point3f &p : in3d) {
		centroid += cv::Point3d(p.x, p.y, p.z);
	}
	centroid *= 1.0 / (double)in3d.size();

	cv::Mat centered((int)in3d.size(), 3, CV_64F);
	for (size_t i = 0; i < in3d.size(); i++) {
		centered.at<double>((int)i, 0) = in3d[i].x - centroid.x;
		centered.at<double>((int)i, 1) = in3d[i].y - centroid.y;
		centered.at<double>((int)i, 2) = in3d[i].z - centroid.z;
	}

	cv::Mat w, u, vt;
	cv::SVD::compute(centered, w, u, vt);
	cv::Mat n_obj = vt.row(2).t();
	cv::Mat n_cam = R_primary * n_obj;

	cv::Mat c_obj = (cv::Mat_<double>(3, 1) << centroid.x, centroid.y, centroid.z);
	cv::Mat c_cam = R_primary * c_obj + tvec_primary;
	const double c_norm = cv::norm(c_cam);
	if (!(c_norm > 1e-9)) {
		return false;
	}
	cv::Mat bearing = c_cam / c_norm;

	cv::Mat reflected = 2.0 * bearing.dot(n_cam) * bearing - n_cam;
	const double reflected_norm = cv::norm(reflected);
	if (!(reflected_norm > 1e-9)) {
		return false;
	}
	reflected /= reflected_norm;

	cv::Mat axis = n_cam.cross(reflected);
	const double sin_angle = cv::norm(axis);
	const double cos_angle = std::max(-1.0, std::min(1.0, n_cam.dot(reflected)));
	cv::Mat R_align;
	if (sin_angle < 1e-9) {
		R_align = cv::Mat::eye(3, 3, CV_64F);
	} else {
		cv::Mat unit_axis = axis / sin_angle;
		cv::Rodrigues(unit_axis * std::atan2(sin_angle, cos_angle), R_align);
	}

	cv::Mat R_twin = R_align * R_primary;
	cv::Rodrigues(R_twin, rvec_twin);
	tvec_twin = c_cam - R_twin * c_obj;
	return vec3_mat_is_finite(rvec_twin) && vec3_mat_is_finite(tvec_twin) && tvec_twin.at<double>(2) > 0.0;
}

/* Materialise the mirror pose for near-planar LED sets so the caller can rank both modes against priors. */
static void
compute_mirror_twin(const std::vector<cv::Point3f> &in3d,
                    const std::vector<cv::Point2f> &in2d,
                    const cv::Mat &K,
                    const cv::Mat &D,
                    const cv::Mat &rvec_primary,
                    const cv::Mat &tvec_primary,
                    double thresh,
                    struct xrt_pose *twin,
                    bool *has_twin)
{
	*has_twin = false;
	if (in3d.size() < 4 || coplanarity_ratio(in3d) >= MIRROR_TWIN_COPLANAR_RATIO) {
		return;
	}

	cv::Mat rvec, tvec;
	try {
		if (!analytic_planar_twin(in3d, rvec_primary, tvec_primary, rvec, tvec)) {
			return;
		}
	} catch (const cv::Exception &) {
		return;
	} catch (...) {
		return;
	}
	if (rot_angle_between(rvec_primary, rvec) <= 20.0 * M_PI / 180.0) {
		return;
	}
	refine_lm_over_inliers(in3d, in2d, K, D, rvec, tvec, thresh); // polish the twin on its own inliers
	if (!vec3_mat_is_finite(rvec) || !vec3_mat_is_finite(tvec) || tvec.at<double>(2) <= 0.0) {
		return;
	}
	rtvec_to_pose(rvec, tvec, twin);
	*has_twin = true;
}

static cv::Mat
quat_to_R(const struct xrt_quat *q)
{
	cv::Mat R(3, 3, CV_64FC1);
	struct xrt_quat qn = *q;
	quat_to_3x3(R, &qn);
	return R;
}

static bool
tilt_clamped_pnp(const std::vector<cv::Point3f> &p3d,
                 const std::vector<cv::Point2f> &p2d,
                 const struct xrt_quat *q_prior_cam,
                 const struct xrt_vec3 *up_cam,
                 const struct xrt_pose *seed,
                 struct xrt_pose *out,
                 struct xrt_pose *yaw_twin,
                 bool *has_yaw_twin)
{
	if (has_yaw_twin != nullptr) {
		*has_yaw_twin = false;
	}
	if (p3d.size() < 3) {
		return false;
	}

	struct xrt_vec3 up = *up_cam;
	const double up_len = std::sqrt((double)up.x * up.x + (double)up.y * up.y + (double)up.z * up.z);
	if (!(up_len > 1e-6)) {
		return false;
	}
	up.x /= (float)up_len;
	up.y /= (float)up_len;
	up.z /= (float)up_len;

	auto R_of_theta = [&](double theta) -> cv::Mat {
		struct xrt_quat ryaw;
		math_quat_from_angle_vector((float)theta, &up, &ryaw);
		struct xrt_quat q;
		math_quat_rotate(&ryaw, q_prior_cam, &q);
		math_quat_normalize(&q);
		return quat_to_R(&q);
	};

	double theta = 0.0;
	cv::Mat t = (cv::Mat_<double>(3, 1) << seed->position.x, seed->position.y, seed->position.z);
	const int n = (int)p3d.size();
	cv::Mat K = cv::Mat::eye(3, 3, CV_64FC1);
	cv::Mat D = cv::Mat::zeros(4, 1, CV_64FC1);

	auto residuals = [&](double th, const cv::Mat &tt, std::vector<double> &r) {
		cv::Mat rvec;
		cv::Rodrigues(R_of_theta(th), rvec);
		std::vector<cv::Point2f> proj;
		cv::projectPoints(p3d, rvec, tt, K, D, proj);
		r.resize(2 * n);
		for (int i = 0; i < n; i++) {
			r[2 * i] = proj[i].x - p2d[i].x;
			r[2 * i + 1] = proj[i].y - p2d[i].y;
		}
	};

	{
		std::vector<double> r;
		double best_cost = INFINITY;
		const int steps = 24;
		for (int i = 0; i < steps; i++) {
			const double th = (2.0 * M_PI * (double)i) / (double)steps;
			residuals(th, t, r);
			double cost = 0.0;
			for (double v : r) {
				cost += v * v;
			}
			if (cost < best_cost) {
				best_cost = cost;
				theta = th;
			}
		}
	}

	std::vector<double> r0(2 * n);
	double prev_cost = INFINITY;
	for (int it = 0; it < 12; it++) {
		residuals(theta, t, r0);
		double cost = 0.0;
		for (double v : r0) {
			cost += v * v;
		}
		if (cost > prev_cost * (1.0 - 1e-6) && it > 0) {
			break;
		}
		prev_cost = cost;

		cv::Mat J(2 * n, 4, CV_64FC1);
		const double dth = 1e-4;
		const double dt = 1e-4;
		{
			std::vector<double> rp, rm;
			residuals(theta + dth, t, rp);
			residuals(theta - dth, t, rm);
			for (int i = 0; i < 2 * n; i++) {
				J.at<double>(i, 0) = (rp[i] - rm[i]) / (2.0 * dth);
			}
		}
		for (int c = 0; c < 3; c++) {
			cv::Mat tp = t.clone();
			cv::Mat tm = t.clone();
			tp.at<double>(c) += dt;
			tm.at<double>(c) -= dt;
			std::vector<double> rp, rm;
			residuals(theta, tp, rp);
			residuals(theta, tm, rm);
			for (int i = 0; i < 2 * n; i++) {
				J.at<double>(i, 1 + c) = (rp[i] - rm[i]) / (2.0 * dt);
			}
		}

		cv::Mat rv(2 * n, 1, CV_64FC1);
		for (int i = 0; i < 2 * n; i++) {
			rv.at<double>(i) = r0[i];
		}
		cv::Mat JtJ = J.t() * J;
		JtJ += cv::Mat::eye(4, 4, CV_64FC1) * (1e-9 * cv::trace(JtJ)[0] + 1e-12);
		cv::Mat dx;
		if (!cv::solve(JtJ, -(J.t() * rv), dx, cv::DECOMP_CHOLESKY)) {
			break;
		}
		theta += dx.at<double>(0);
		t.at<double>(0) += dx.at<double>(1);
		t.at<double>(1) += dx.at<double>(2);
		t.at<double>(2) += dx.at<double>(3);
		if (cv::norm(dx) < 1e-9) {
			break;
		}
	}

	/* KNOWN, CALIBRATED-IN: this GN has no step acceptance, so it can end on a
	 * cost-increasing final step (the divergence break fires one iteration late).
	 * The audit's best-so-far rollback (R1-L8) was implemented and REFUTED on the
	 * frozen matrices — xv1/bursts-long objective 40.10->39.24 (dev1), 39.28->38.57
	 * (dev2); the downstream selection stack is calibrated WITH this behavior.
	 * See results/w3-preupstream-20260711/l8-tilt-rollback-adjudication/. Re-open
	 * only inside the W5 analytic-Jacobian rework of this solver, matrix-gated. */

	if (t.at<double>(2) <= 0.0) {
		return false;
	}

	cv::Mat rvec;
	cv::Rodrigues(R_of_theta(theta), rvec);
	rtvec_to_pose(rvec, t, out);

	if (yaw_twin != nullptr && has_yaw_twin != nullptr) {
		double th2 = theta + M_PI;
		cv::Mat t2 = t.clone();
		double prev2 = INFINITY;
		for (int it = 0; it < 8; it++) {
			std::vector<double> r;
			residuals(th2, t2, r);
			double cost = 0.0;
			for (double v : r) {
				cost += v * v;
			}
			if (cost > prev2 * (1.0 - 1e-6) && it > 0) {
				break;
			}
			prev2 = cost;
			cv::Mat J(2 * n, 3, CV_64FC1);
			const double dt = 1e-4;
			for (int c = 0; c < 3; c++) {
				cv::Mat tp = t2.clone();
				cv::Mat tm = t2.clone();
				tp.at<double>(c) += dt;
				tm.at<double>(c) -= dt;
				std::vector<double> rp, rm;
				residuals(th2, tp, rp);
				residuals(th2, tm, rm);
				for (int i = 0; i < 2 * n; i++) {
					J.at<double>(i, c) = (rp[i] - rm[i]) / (2.0 * dt);
				}
			}
			cv::Mat rv(2 * n, 1, CV_64FC1);
			for (int i = 0; i < 2 * n; i++) {
				rv.at<double>(i) = r[i];
			}
			cv::Mat JtJ = J.t() * J;
			JtJ += cv::Mat::eye(3, 3, CV_64FC1) * (1e-9 * cv::trace(JtJ)[0] + 1e-12);
			cv::Mat dx;
			if (!cv::solve(JtJ, -(J.t() * rv), dx, cv::DECOMP_CHOLESKY)) {
				break;
			}
			t2.at<double>(0) += dx.at<double>(0);
			t2.at<double>(1) += dx.at<double>(1);
			t2.at<double>(2) += dx.at<double>(2);
			if (cv::norm(dx) < 1e-9) {
				break;
			}
		}
		if (t2.at<double>(2) > 0.0) {
			cv::Mat rvec2;
			cv::Rodrigues(R_of_theta(th2), rvec2);
			rtvec_to_pose(rvec2, t2, yaw_twin);
			*has_yaw_twin = true;
		}
	}

	return true;
}

bool
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
	if (has_twin != nullptr) {
		*has_twin = false;
	}

	int i, j;
	int num_leds = 0;
	uint64_t taken = 0;
	const int flags = cv::SOLVEPNP_SQPNP;
	cv::Mat inliers;
	const int iterationsCount = 100;
	const float confidence = 0.99f;
	/* RANSAC stays the labelled-set solver: a direct SQPnP-on-all-points + all-inlier gate was tried
	 * (2026-06-11) and regressed the dropout matrices (blackout geomean 48.57 -> 47.45, five cells down)
	 * for only ~3% wall — best-of-100 genuinely beats one global solve here. Do not re-walk. */
	static const cv::Mat dummyK = cv::Mat::eye(3, 3, CV_64FC1);
	static const cv::Mat dummyD = cv::Mat::zeros(4, 1, CV_64FC1);
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

	/* count identified leds */
	for (i = 0; i < num_blobs; i++) {
		int led_id = blobs[i].led_id;
		if (LED_OBJECT_ID(led_id) != leds_model->id)
			continue; /* invalid or LED id for another object */
		led_id = LED_LOCAL_ID(led_id);
		if (led_id < 0 || led_id >= leds_model->num_leds || led_id >= 64)
			continue; /* corrupt label: out of model range / taken-mask width */

		if (taken & (1ULL << led_id))
			continue;
		taken |= (1ULL << led_id);
		num_leds++;
	}
	if (num_leds_out)
		*num_leds_out = num_leds;

	if (num_leds < 4)
		return false;

	std::vector<cv::Point3f> list_points3d(num_leds);
	std::vector<cv::Point2f> list_points2d(num_leds);
	std::vector<cv::Point2f> list_points2d_undistorted(num_leds);

	taken = 0;
	for (i = 0, j = 0; i < num_blobs && j < num_leds; i++) {
		int led_id = blobs[i].led_id;
		if (LED_OBJECT_ID(led_id) != leds_model->id)
			continue; /* invalid or LED id for another object */
		led_id = LED_LOCAL_ID(led_id);
		if (led_id < 0 || led_id >= leds_model->num_leds || led_id >= 64)
			continue; /* corrupt label: out of model range / taken-mask width */
		if (taken & (1ULL << led_id))
			continue;
		taken |= (1ULL << led_id);
		list_points3d[j].x = leds_model->leds[led_id].pos.x;
		list_points3d[j].y = leds_model->leds[led_id].pos.y;
		list_points3d[j].z = leds_model->leds[led_id].pos.z;
		list_points2d[j].x = blobs[i].x;
		list_points2d[j].y = blobs[i].y;
		j++;
	}

	num_leds = j;
	if (num_leds < 4)
		return false;
	list_points3d.resize(num_leds);
	list_points2d.resize(num_leds);
	list_points2d_undistorted.resize(num_leds);

	// we undistort the image points manually before passing them to the PnpRansac solver
	// and we give the solver identity camera + null distortion matrices
	undistort_blob_points(list_points2d, list_points2d_undistorted, calib);

	/* 3 pixel reprojection threshold (normalised by focal since we solve in normalised coords) */
	const float reprojectionError = 3.0f / calib->calib.fx;

	if (!points_finite(list_points3d, list_points2d_undistorted)) {
		return false;
	}

	if (!cv::solvePnPRansac(list_points3d, list_points2d_undistorted, dummyK, dummyD, rvec, tvec, false,
	                        iterationsCount, reprojectionError, confidence, inliers, flags)) {
		return false;
	}

	/* SQPnP-RANSAC gives a good global estimate, but the LM reprojection refinement is what drives the
	 * per-LED error down (the weak 6-7-inlier poses were accepted at ~2 px). Re-gate + refine twice
	 * (IRLS-style): the first pass tightens the pose, the second folds back any point the tightened pose
	 * now fits. Keep the final inlier set for the optional twin recovery. */
	int final_inliers = inliers.rows;
	std::vector<cv::Point3f> in3d;
	std::vector<cv::Point2f> in2d;
	if (final_inliers >= 4) {
		refine_lm_over_inliers(list_points3d, list_points2d_undistorted, dummyK, dummyD, rvec, tvec,
		                       reprojectionError);
		final_inliers = refine_lm_over_inliers(list_points3d, list_points2d_undistorted, dummyK, dummyD, rvec,
		                                       tvec, reprojectionError, &in3d, &in2d);
	}

	if (num_inliers)
		*num_inliers = final_inliers;

	rtvec_to_pose(rvec, tvec, pose);

	if (has_twin != nullptr && final_inliers >= 4) {
		compute_mirror_twin(in3d, in2d, dummyK, dummyD, rvec, tvec, reprojectionError, twin, has_twin);
	}

	U_LOG_T("Got PnP pose quat %f %f %f %f  pos %f %f %f%s", pose->orientation.x, pose->orientation.y,
	        pose->orientation.z, pose->orientation.w, pose->position.x, pose->position.y, pose->position.z,
	        (has_twin != nullptr && *has_twin) ? " (+mirror twin)" : "");
	return true;
}

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
                      bool *has_yaw_twin)
{
	if (has_yaw_twin != nullptr) {
		*has_yaw_twin = false;
	}
	if (n < 4 || pose == nullptr || obj_pts == nullptr || img_pts == nullptr || calib == nullptr ||
	    prior_cam == nullptr || up_cam == nullptr || out == nullptr) {
		return false;
	}

	std::vector<cv::Point3f> p3d(n);
	std::vector<cv::Point2f> p2d_dist(n);
	std::vector<cv::Point2f> p2d(n);
	for (int i = 0; i < n; i++) {
		p3d[i] = cv::Point3f(obj_pts[i].x, obj_pts[i].y, obj_pts[i].z);
		p2d_dist[i] = cv::Point2f(img_pts[i].x, img_pts[i].y);
	}
	if (require_coplanar && !near_coplanar(p3d)) {
		return false;
	}

	undistort_blob_points(p2d_dist, p2d, calib);
	struct xrt_pose clamped;
	if (!tilt_clamped_pnp(p3d, p2d, &prior_cam->orientation, up_cam, pose, &clamped, yaw_twin,
	                      has_yaw_twin)) {
		return false;
	}
	*out = clamped;
	return true;
}

int
pnp_solve_p3p(struct blob *blobs,
              int num_blobs,
              struct t_constellation_led_model *leds_model,
              struct camera_model *calib,
              struct xrt_pose *out_poses,
              int max_out)
{
	if (max_out <= 0) {
		return 0;
	}

	std::vector<cv::Point3f> p3d;
	std::vector<cv::Point2f> p2d;
	p3d.reserve(3);
	p2d.reserve(3);
	uint64_t taken = 0;
	for (int i = 0; i < num_blobs && p3d.size() < 3; i++) {
		int led_id = blobs[i].led_id;
		if (LED_OBJECT_ID(led_id) != leds_model->id) {
			continue;
		}
		led_id = LED_LOCAL_ID(led_id);
		if (led_id < 0 || led_id >= leds_model->num_leds || led_id >= 64) {
			continue;
		}
		if (taken & (1ULL << led_id)) {
			continue;
		}
		taken |= (1ULL << led_id);
		p3d.push_back(cv::Point3f(leds_model->leds[led_id].pos.x, leds_model->leds[led_id].pos.y,
		                           leds_model->leds[led_id].pos.z));
		p2d.push_back(cv::Point2f(blobs[i].x, blobs[i].y));
	}
	if (p3d.size() != 3) {
		return 0;
	}

	std::vector<cv::Point2f> p2d_und(3);
	undistort_blob_points(p2d, p2d_und, calib);

	cv::Mat K = cv::Mat::eye(3, 3, CV_64FC1);
	cv::Mat D = cv::Mat::zeros(4, 1, CV_64FC1);
	std::vector<cv::Mat> rvecs, tvecs;
	if (!points_finite(p3d, p2d_und)) {
		return 0;
	}
	try {
		cv::solveP3P(p3d, p2d_und, K, D, rvecs, tvecs, cv::SOLVEPNP_AP3P);
	} catch (const cv::Exception &) {
		return 0; // offline-only safety; live can't rely on this catch (see points_finite)
	}

	int written = 0;
	for (size_t i = 0; i < rvecs.size() && written < max_out; i++) {
		if (tvecs[i].at<double>(2) <= 0.0) {
			continue;
		}
		rtvec_to_pose(rvecs[i], tvecs[i], &out_poses[written++]);
	}
	return written;
}
