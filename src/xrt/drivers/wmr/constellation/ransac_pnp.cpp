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
 * @ingroup drv_wmr
 */
#include "ransac_pnp.h"
#include "util/u_logging.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#if CV_MAJOR_VERSION >= 4
#include <opencv2/calib3d/calib3d_c.h>
#endif
#include <iostream>

using namespace std;

#include <stdio.h>

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

bool
ransac_pnp_pose(struct xrt_pose *pose,
                struct blob *blobs,
                int num_blobs,
                struct constellation_led_model *leds_model,
                struct camera_model *calib,
                int *num_leds_out,
                int *num_inliers)
{
	int i, j;
	int num_leds = 0;
	uint64_t taken = 0;
	int flags = cv::SOLVEPNP_ITERATIVE;
	cv::Mat inliers;
	int iterationsCount = 100;
	float confidence = 0.99;
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

	// cout << "R = " << R << ", rvec = " << rvec << endl;

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

		U_LOG_D("LED %d at %f,%f (3D %f %f %f)", blobs[i].led_id, blobs[i].x, blobs[i].y,
		        leds_model->leds[led_id].pos.x, leds_model->leds[led_id].pos.y, leds_model->leds[led_id].pos.z);
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

	/* 3 pixel reprojection threshold */
	float reprojectionError = 3.0 / calib->calib.fx;

	cv::solvePnPRansac(list_points3d, list_points2d_undistorted, dummyK, dummyD, rvec, tvec, false, iterationsCount,
	                   reprojectionError, confidence, inliers, flags);

	if (num_inliers)
		*num_inliers = inliers.rows;

	struct xrt_vec3 v;
	double angle = sqrt(rvec.dot(rvec));
	double inorm = 1.0f / angle;

	v.x = rvec.at<double>(0) * inorm;
	v.y = rvec.at<double>(1) * inorm;
	v.z = rvec.at<double>(2) * inorm;
	math_quat_from_angle_vector(angle, &v, &pose->orientation);
	pose->position.x = tvec.at<double>(0);
	pose->position.y = tvec.at<double>(1);
	pose->position.z = tvec.at<double>(2);

	U_LOG_T("Got PnP pose quat %f %f %f %f  pos %f %f %f", pose->orientation.x, pose->orientation.y,
	        pose->orientation.z, pose->orientation.w, pose->position.x, pose->position.y, pose->position.z);
	return true;
}
