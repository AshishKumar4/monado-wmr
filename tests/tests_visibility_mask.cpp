// Copyright 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Visibility mask tests: boundary-based mask construction and the
 * poly 3K (WMR) hidden area derivation.
 */

#include "catch_amalgamated.hpp"

#include "math/m_api.h"
#include "util/u_distortion_mesh.h"
#include "util/u_visibility_mask.h"
#include "xrt/xrt_visibility_mask.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// Inverse of the UV -> tangent conversion the mask builders apply, the same
// conversion the SteamVR plugin (ovrd_driver.cpp) uses to serve OpenVR
// hidden area meshes. Returns a flat vertex list in indexed order.
std::vector<xrt_vec2>
mask_to_uv(const xrt_visibility_mask *mask, const xrt_fov &fov)
{
	const float tan_left = tanf(fov.angle_left);
	const float tan_right = tanf(fov.angle_right);
	const float tan_up = tanf(fov.angle_up);
	const float tan_down = tanf(fov.angle_down);

	const xrt_vec2 *verts = xrt_visibility_mask_get_vertices(mask);
	const uint32_t *indices = xrt_visibility_mask_get_indices(mask);

	std::vector<xrt_vec2> uv(mask->index_count);
	for (uint32_t i = 0; i < mask->index_count; i++) {
		const xrt_vec2 &v = verts[indices[i]];
		uv[i].x = (v.x - tan_left) / (tan_right - tan_left);
		uv[i].y = (tan_up - v.y) / (tan_up - tan_down);
	}
	return uv;
}

double
triangle_list_area(const std::vector<xrt_vec2> &uv)
{
	double area = 0;
	for (size_t i = 0; i + 2 < uv.size(); i += 3) {
		const double ax = uv[i + 1].x - uv[i].x;
		const double ay = uv[i + 1].y - uv[i].y;
		const double bx = uv[i + 2].x - uv[i].x;
		const double by = uv[i + 2].y - uv[i].y;
		area += std::abs(ax * by - ay * bx) / 2;
	}
	return area;
}

void
require_uv_in_unit_square(const std::vector<xrt_vec2> &uv)
{
	for (const xrt_vec2 &p : uv) {
		REQUIRE(p.x >= -1e-4f);
		REQUIRE(p.x <= 1.f + 1e-4f);
		REQUIRE(p.y >= -1e-4f);
		REQUIRE(p.y <= 1.f + 1e-4f);
	}
}

} // namespace

TEST_CASE("u_visibility_mask_from_uv_boundary partitions the unit square")
{
	constexpr uint32_t n = 64;
	constexpr float radius = 0.45f;

	xrt_vec2 boundary[n];
	for (uint32_t i = 0; i < n; i++) {
		const double theta = 2.0 * M_PI * i / n;
		boundary[i].x = 0.5f + (float)cos(theta) * radius;
		boundary[i].y = 0.5f + (float)sin(theta) * radius;
	}

	xrt_fov fov = {};
	fov.angle_left = (float)(-M_PI / 4);
	fov.angle_right = (float)(M_PI / 4);
	fov.angle_up = (float)(M_PI / 4);
	fov.angle_down = (float)(-M_PI / 4);

	// Area of the inscribed regular 64-gon, the polygon the mask actually uses.
	const double polygon_area = 0.5 * n * radius * radius * sin(2.0 * M_PI / n);

	xrt_visibility_mask *hidden = NULL;
	xrt_visibility_mask *visible = NULL;
	xrt_visibility_mask *loop = NULL;
	u_visibility_mask_from_uv_boundary(XRT_VISIBILITY_MASK_TYPE_HIDDEN_TRIANGLE_MESH, boundary, n, &fov, &hidden);
	u_visibility_mask_from_uv_boundary(XRT_VISIBILITY_MASK_TYPE_VISIBLE_TRIANGLE_MESH, boundary, n, &fov,
	                                   &visible);
	u_visibility_mask_from_uv_boundary(XRT_VISIBILITY_MASK_TYPE_LINE_LOOP, boundary, n, &fov, &loop);
	REQUIRE(hidden != NULL);
	REQUIRE(visible != NULL);
	REQUIRE(loop != NULL);

	CHECK(hidden->type == XRT_VISIBILITY_MASK_TYPE_HIDDEN_TRIANGLE_MESH);
	CHECK(hidden->vertex_count == n * 2 + 4);
	// Two triangles per boundary segment plus one per square corner the
	// perimeter path crosses (up to 4, fewer when a projection lands
	// exactly on a corner).
	CHECK(hidden->index_count >= n * 6);
	CHECK(hidden->index_count <= n * 6 + 4 * 3);
	CHECK(visible->type == XRT_VISIBILITY_MASK_TYPE_VISIBLE_TRIANGLE_MESH);
	CHECK(visible->vertex_count == n + 1);
	CHECK(visible->index_count == n * 3);
	CHECK(loop->type == XRT_VISIBILITY_MASK_TYPE_LINE_LOOP);
	CHECK(loop->vertex_count == n);
	CHECK(loop->index_count == n);

	const std::vector<xrt_vec2> hidden_uv = mask_to_uv(hidden, fov);
	const std::vector<xrt_vec2> visible_uv = mask_to_uv(visible, fov);
	const std::vector<xrt_vec2> loop_uv = mask_to_uv(loop, fov);
	require_uv_in_unit_square(hidden_uv);
	require_uv_in_unit_square(visible_uv);
	require_uv_in_unit_square(loop_uv);

	const double hidden_area = triangle_list_area(hidden_uv);
	const double visible_area = triangle_list_area(visible_uv);
	CHECK_THAT(visible_area, Catch::Matchers::WithinAbs(polygon_area, 1e-3));
	CHECK_THAT(hidden_area, Catch::Matchers::WithinAbs(1.0 - polygon_area, 1e-3));
	// Hidden ring + visible fan must exactly partition the unit square.
	CHECK_THAT(hidden_area + visible_area, Catch::Matchers::WithinAbs(1.0, 1e-5));

	// The line loop is the visible/hidden boundary itself.
	REQUIRE(loop_uv.size() == n);
	for (uint32_t i = 0; i < n; i++) {
		CHECK_THAT(loop_uv[i].x, Catch::Matchers::WithinAbs(boundary[i].x, 1e-5));
		CHECK_THAT(loop_uv[i].y, Catch::Matchers::WithinAbs(boundary[i].y, 1e-5));
	}

	free(hidden);
	free(visible);
	free(loop);
}

TEST_CASE("u_compute_visibility_mask_poly_3k on Odyssey+ factory calibration")
{
	// Real WMR display calibration read from a Samsung Odyssey+ (source:
	// CIFASIS/basalt-xr data/monado/wmr-tools/odysseyplus_wmrcalib_example.json),
	// the same JSON shape wmr_config.c parses. Layout per eye: row-major
	// affine, then (cx, cy, k1, k2, k3) for red/green/blue.
	struct display_calib
	{
		float affine[9];
		double channels[3][5];
		xrt_vec2 visible_center;
		float visible_radius;
	};

	static const display_calib eyes[2] = {
	    {
	        {907.9921264648438f, -0.5576117634773254f, 785.1776123046875f, //
	         0.f, 908.129150390625f, 796.9616088867188f,                   //
	         0.f, 0.f, 1.f},
	        {
	            {787.3162958384937, 795.2429185372553, 3.9138434409665894e-07, 2.875815973465491e-13,
	             4.910649964288027e-19},
	            {786.6702072553893, 794.6923805001559, 4.2007202329361826e-07, 1.8878438698927907e-13,
	             7.025060016938386e-19},
	            {785.8925215555417, 797.6765657316258, 5.020213132586381e-07, -1.281348706158102e-13,
	             1.2345214819185544e-18},
	        },
	        {785.1776315693005f, 796.9616285756942f},
	        820.f,
	    },
	    {
	        {907.9639892578125f, -0.49046245217323303f, 2103.386474609375f, //
	         0.f, 907.9780883789062f, 801.8683471679688f,                   //
	         0.f, 0.f, 1.f},
	        {
	            {2101.5711896433636, 800.9683806433902, 3.9383231143483177e-07, 2.422405601887851e-13,
	             5.844860687929612e-19},
	            {2101.1059339164444, 799.9633479769144, 4.245482284354187e-07, 1.412440585952293e-13,
	             7.9387878585863295e-19},
	            {2100.8534549194974, 802.5782513180413, 5.116191780309974e-07, -1.9534633908166077e-13,
	             1.3528407038619122e-18},
	        },
	        {2103.3864389371893f, 801.8683736319904f},
	        820.f,
	    },
	};

	// Independently re-derived with numpy from the same calibration
	// (results/render-l2-ham-20260612/ham_crosscheck.py in the research tree).
	static const double expected_hidden[2] = {0.0191, 0.0144};

	for (uint32_t view = 0; view < 2; view++) {
		const display_calib &dc = eyes[view];

		u_poly_3k_eye_values values = {};
		xrt_matrix_3x3 affine;
		std::memcpy(affine.v, dc.affine, sizeof(affine.v));
		math_matrix_3x3_inverse(&affine, &values.inv_affine_xform);
		for (uint32_t c = 0; c < 3; c++) {
			values.channels[c].display_size = {2880, 1600};
			values.channels[c].eye_center = {(float)dc.channels[c][0], (float)dc.channels[c][1]};
			values.channels[c].k[0] = dc.channels[c][2];
			values.channels[c].k[1] = dc.channels[c][3];
			values.channels[c].k[2] = dc.channels[c][4];
		}

		xrt_fov fov = {};
		u_compute_distortion_bounds_poly_3k(&values.inv_affine_xform, values.channels, view, &fov,
		                                    &values.tex_x_range, &values.tex_y_range);

		xrt_visibility_mask *hidden = NULL;
		xrt_visibility_mask *visible = NULL;
		u_compute_visibility_mask_poly_3k(&values, view, dc.visible_center, dc.visible_radius, &fov,
		                                  XRT_VISIBILITY_MASK_TYPE_HIDDEN_TRIANGLE_MESH, &hidden);
		u_compute_visibility_mask_poly_3k(&values, view, dc.visible_center, dc.visible_radius, &fov,
		                                  XRT_VISIBILITY_MASK_TYPE_VISIBLE_TRIANGLE_MESH, &visible);
		REQUIRE(hidden != NULL);
		REQUIRE(visible != NULL);

		const std::vector<xrt_vec2> hidden_uv = mask_to_uv(hidden, fov);
		const std::vector<xrt_vec2> visible_uv = mask_to_uv(visible, fov);
		require_uv_in_unit_square(hidden_uv);
		require_uv_in_unit_square(visible_uv);

		const double hidden_area = triangle_list_area(hidden_uv);
		const double visible_area = triangle_list_area(visible_uv);
		printf("Odyssey+ eye %u: hidden area %.2f%% of the render texture (%u triangles)\n", view,
		       hidden_area * 100.0, hidden->index_count / 3);

		// Sane band plus agreement with the independent reimplementation.
		CHECK(hidden_area > 0.005);
		CHECK(hidden_area < 0.25);
		CHECK_THAT(hidden_area, Catch::Matchers::WithinAbs(expected_hidden[view], 0.002));
		CHECK_THAT(hidden_area + visible_area, Catch::Matchers::WithinAbs(1.0, 1e-5));

		free(hidden);
		free(visible);
	}
}
