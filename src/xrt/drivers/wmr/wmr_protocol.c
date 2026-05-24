// Copyright 2018, Philipp Zabel.
// Copyright 2020-2021, N Madsen.
// Copyright 2020-2021, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  WMR and MS HoloLens protocol helpers implementation.
 * @author Philipp Zabel <philipp.zabel@gmail.com>
 * @author nima01 <nima_zero_one@protonmail.com>
 * @ingroup drv_wmr
 */

#include "wmr_protocol.h"

#include <stdlib.h>


/*
 *
 * WMR and MS HoloLens Sensors protocol helpers
 *
 */

// Absolute accel scale (LSB -> m/s^2). The original flat 0.001 read the at-rest magnitude ~1.8% high
// (|a| ~9.98 vs 9.80665) on this G2, consistently across every capture. The factory mix-matrix is
// ~unity and Basalt only estimates accel *bias* (never scale), so this constant is the single place
// the absolute scale lives -> correcting it here fixes the SLAM and 3DOF paths at once, no
// double-correction. 0.000982 = 0.001 * (9.80665/9.984 mean-at-rest). Env-overridable
// (G2_ACCEL_SCALE) so a clean still-capture calibration can refine it to <0.1% without a rebuild.
// Read once (HMD HoloLens sensor thread only -> the lazy init has no meaningful race).
static float
g2_hololens_accel_scale(void)
{
	static float s = -1.0f;
	if (s < 0.0f) {
		s = 0.000982f;
		const char *e = getenv("G2_ACCEL_SCALE");
		if (e != NULL && *e != '\0') {
			float v = (float)atof(e);
			if (v > 0.0f) {
				s = v;
			}
		}
	}
	return s;
}

void
vec3_from_hololens_accel(int32_t sample[3][4], int i, struct xrt_vec3 *out_vec)
{
	const float sc = g2_hololens_accel_scale();
	out_vec->x = (float)sample[0][i] * sc;
	out_vec->y = (float)sample[1][i] * sc;
	out_vec->z = (float)sample[2][i] * sc;
}

void
vec3_from_hololens_gyro(int16_t sample[3][32], int i, struct xrt_vec3 *out_vec)
{
	out_vec->x = (float)(sample[0][8 * i + 0] + //
	                     sample[0][8 * i + 1] + //
	                     sample[0][8 * i + 2] + //
	                     sample[0][8 * i + 3] + //
	                     sample[0][8 * i + 4] + //
	                     sample[0][8 * i + 5] + //
	                     sample[0][8 * i + 6] + //
	                     sample[0][8 * i + 7]) *
	             0.001f * 0.125f;
	out_vec->y = (float)(sample[1][8 * i + 0] + //
	                     sample[1][8 * i + 1] + //
	                     sample[1][8 * i + 2] + //
	                     sample[1][8 * i + 3] + //
	                     sample[1][8 * i + 4] + //
	                     sample[1][8 * i + 5] + //
	                     sample[1][8 * i + 6] + //
	                     sample[1][8 * i + 7]) *
	             0.001f * 0.125f;
	out_vec->z = (float)(sample[2][8 * i + 0] + //
	                     sample[2][8 * i + 1] + //
	                     sample[2][8 * i + 2] + //
	                     sample[2][8 * i + 3] + //
	                     sample[2][8 * i + 4] + //
	                     sample[2][8 * i + 5] + //
	                     sample[2][8 * i + 6] + //
	                     sample[2][8 * i + 7]) *
	             0.001f * 0.125f;
}
