/*
 * Camera distortion/projection model
 * Copyright 2014-2015 Philipp Zabel
 * Copyright 2019-2023 Jan Schmidt
 * SPDX-License-Identifier:	LGPL-2.0+ or BSL-1.0
 */
#pragma once

#include "tracking/t_camera_models.h"

struct camera_model
{
	/* Frame width and height */
	int width;
	int height;

	/* Distortion / projection parameters */
	struct t_camera_model_params calib;
};
