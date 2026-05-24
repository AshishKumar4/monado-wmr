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

//! True if the 3D points are near-coplanar (thin in one dimension) — the configuration that admits the
//! PnP mirror two-fold ambiguity. Ratio of the smallest to largest singular value of the centred points.
static bool
near_coplanar(const std::vector<cv::Point3f> &pts)
{
	if (pts.size() < 4) {
		return false;
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
	return s0 > 1e-9 && (s2 / s0) < 0.10;
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

/* When the inlier set is near-coplanar, recover the second (mirror-twin) pose via IPPE so the caller can
 * disambiguate against the prior. Writes @p twin + sets @p has_twin only if a distinct second solution
 * (> ~20 deg from the primary) is found. Robust to OpenCV throwing on a degenerate planar config. */
static void
compute_mirror_twin(const std::vector<cv::Point3f> &in3d,
                    const std::vector<cv::Point2f> &in2d,
                    const cv::Mat &K,
                    const cv::Mat &D,
                    const cv::Mat &rvec_primary,
                    double thresh,
                    struct xrt_pose *twin,
                    bool *has_twin)
{
	*has_twin = false;
	if (in3d.size() < 4 || !near_coplanar(in3d)) {
		return;
	}
	std::vector<cv::Mat> rvecs, tvecs;
	try {
		// IPPE returns up to two solutions for a planar set; it cannot pick between them from one view.
		cv::solvePnPGeneric(in3d, in2d, K, D, rvecs, tvecs, false, cv::SOLVEPNP_IPPE);
	} catch (const cv::Exception &) {
		return; // degenerate planar config — no twin
	}
	// Pick the solution rotationally farthest from the primary (the mirror), if it is genuinely distinct.
	int best = -1;
	double best_ang = 20.0 * M_PI / 180.0; // require a real second mode, not a near-duplicate
	for (size_t i = 0; i < rvecs.size(); i++) {
		const double a = rot_angle_between(rvec_primary, rvecs[i]);
		if (a > best_ang) {
			best_ang = a;
			best = (int)i;
		}
	}
	if (best < 0) {
		return;
	}
	cv::Mat rvec = rvecs[best].clone(), tvec = tvecs[best].clone();
	refine_lm_over_inliers(in3d, in2d, K, D, rvec, tvec, thresh); // polish the twin on its own inliers
	rtvec_to_pose(rvec, tvec, twin);
	*has_twin = true;
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
	cv::Mat dummyK = cv::Mat::eye(3, 3, CV_64FC1);
	cv::Mat dummyD = cv::Mat::zeros(4, 1, CV_64FC1);
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat R = cv::Mat::zeros(3, 3, CV_64FC1);

	tvec.at<double>(0) = pose->position.x;
	tvec.at<double>(1) = pose->position.y;
	tvec.at<double>(2) = pose->position.z;

	quat_to_3x3(R, &pose->orientation);
	cv::Rodrigues(R, rvec);

	/* count identified leds */
	for (i = 0; i < num_blobs; i++) {
		int led_id = blobs[i].led_id;
		if (LED_OBJECT_ID(led_id) != leds_model->id)
			continue; /* invalid or LED id for another object */
		led_id = LED_LOCAL_ID(led_id);

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

	cv::solvePnPRansac(list_points3d, list_points2d_undistorted, dummyK, dummyD, rvec, tvec, false, iterationsCount,
	                   reprojectionError, confidence, inliers, flags);

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
		compute_mirror_twin(in3d, in2d, dummyK, dummyD, rvec, reprojectionError, twin, has_twin);
	}

	U_LOG_T("Got PnP pose quat %f %f %f %f  pos %f %f %f%s", pose->orientation.x, pose->orientation.y,
	        pose->orientation.z, pose->orientation.w, pose->position.x, pose->position.y, pose->position.z,
	        (has_twin != nullptr && *has_twin) ? " (+mirror twin)" : "");
	return true;
}

bool
ransac_pnp_pose(struct xrt_pose *pose,
                struct blob *blobs,
                int num_blobs,
                struct t_constellation_led_model *leds_model,
                struct camera_model *calib,
                int *num_leds_out,
                int *num_inliers)
{
	return ransac_pnp_pose_with_twin(pose, blobs, num_blobs, leds_model, calib, num_leds_out, num_inliers,
	                                 nullptr, nullptr);
}
