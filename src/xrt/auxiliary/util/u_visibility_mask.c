// Copyright 2023, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Visibility mask utilitary
 * @author Simon Zeni <simon.zeni@collabora.com>
 * @ingroup aux_util
 */

#include "math/m_mathinclude.h"

#include "u_misc.h"
#include "u_visibility_mask.h"
#include "u_logging.h"


#include <assert.h>
#include <stdlib.h>
#include <string.h>

static const struct xrt_vec2 vertices_hidden[] = {
    {1.0, 0.75},   {1.0, 1.0},   {0.75, 1.0},   {-1.0, 1.0},  {-1.0, 0.75}, {-0.75, 1.0},
    {-1.0, -0.75}, {-1.0, -1.0}, {-0.75, -1.0}, {0.75, -1.0}, {1.0, -1.0},  {1.0, -0.75},
};

static const uint32_t indices_hidden[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

static const struct xrt_vec2 vertices_visible[] = {
    {1.f, -.75f}, {.75f, -1.f}, {-.75f, -1.f}, {-1.f, -.75f}, {-1.f, .75f},
    {-.75f, 1.f}, {.75f, 1.f},  {1.f, .75f},   {0.f, 0.f},
};

static const uint32_t indices_visible[] = {
    8, 2, 1, 3, 2, 8, 8, 1, 0, 6, 8, 7, 4, 8, 5, 8, 0, 7, 5, 8, 6, 4, 3, 8,
};

static const struct xrt_vec2 vertices_line[] = {
    {-.75f, -1.f}, {.75f, -1.f}, {1.f, -.75f}, {1.f, .75f}, {.75f, 1.f}, {-.75f, 1.f}, {-1.f, .75f}, {-1.f, -.75f},
};

static const uint32_t indices_line[] = {0, 1, 2, 3, 4, 5, 6, 7};

void
u_visibility_mask_get_default(enum xrt_visibility_mask_type type,
                              const struct xrt_fov *fov,
                              struct xrt_visibility_mask **out_mask)
{
	struct xrt_visibility_mask *mask = NULL;
	uint32_t nvertices = 0, nindices = 0;

	switch (type) {
	case XRT_VISIBILITY_MASK_TYPE_HIDDEN_TRIANGLE_MESH:
		nvertices = ARRAY_SIZE(vertices_hidden);
		nindices = ARRAY_SIZE(indices_hidden);
		break;
	case XRT_VISIBILITY_MASK_TYPE_VISIBLE_TRIANGLE_MESH:
		nvertices = ARRAY_SIZE(vertices_visible);
		nindices = ARRAY_SIZE(indices_visible);
		break;
	case XRT_VISIBILITY_MASK_TYPE_LINE_LOOP:
		nvertices = ARRAY_SIZE(vertices_line);
		nindices = ARRAY_SIZE(indices_line);
		break;
	}

	const size_t size =
	    sizeof(struct xrt_visibility_mask) + sizeof(uint32_t) * nindices + sizeof(struct xrt_vec2) * nvertices;
	mask = U_CALLOC_WITH_CAST(struct xrt_visibility_mask, size);
	if (mask == NULL) {
		U_LOG_E("failed to allocate out xrt_visibility_mask");
		goto out;
	}


	mask->index_count = nindices;
	mask->vertex_count = nvertices;

	const struct xrt_vec2 *vertices = NULL;
	const uint32_t *indices = NULL;

	switch (type) {
	case XRT_VISIBILITY_MASK_TYPE_HIDDEN_TRIANGLE_MESH:
		vertices = vertices_hidden;
		indices = indices_hidden;
		break;
	case XRT_VISIBILITY_MASK_TYPE_VISIBLE_TRIANGLE_MESH:
		vertices = vertices_visible;
		indices = indices_visible;
		break;
	case XRT_VISIBILITY_MASK_TYPE_LINE_LOOP:
		vertices = vertices_line;
		indices = indices_line;
		break;
	}

	memcpy(xrt_visibility_mask_get_indices(mask), indices, sizeof(uint32_t) * nindices);


	const struct xrt_fov copy = *fov;

	const float tan_left = tanf(copy.angle_left);
	const float tan_right = tanf(copy.angle_right);

	const float tan_down = tanf(copy.angle_down);
	const float tan_up = tanf(copy.angle_up);

	const float tan_half_width = (tan_right - tan_left);
	const float tan_half_height = (tan_up - tan_down);

	const float tan_offset_x = ((tan_right + tan_left) - tan_half_width) / 2;
	const float tan_offset_y = (-(tan_up + tan_down) - tan_half_height) / 2;

	struct xrt_vec2 *dst = xrt_visibility_mask_get_vertices(mask);
	for (uint32_t i = 0; i < nvertices; i++) {
		struct xrt_vec2 v = vertices[i];

		// Yes this is really the simplest form, WolframAlpha agrees.
		v.x = (v.x * 0.5 + 0.5) * tan_half_width + tan_offset_x;
		v.y = (v.y * 0.5 + 0.5) * tan_half_height + tan_offset_y;

		dst[i] = v;
	}

out:
	*out_mask = mask; // Always NULL or allocated data.
}

static struct xrt_vec2
clamp_to_unit_square(struct xrt_vec2 p)
{
	p.x = fminf(fmaxf(p.x, 0.f), 1.f);
	p.y = fminf(fmaxf(p.y, 0.f), 1.f);
	return p;
}

/*
 * Extend the ray from @p origin through @p through until it hits the unit
 * square boundary. @p origin must be inside the square and @p through must
 * not coincide with it.
 */
static struct xrt_vec2
ray_to_unit_square_edge(struct xrt_vec2 origin, struct xrt_vec2 through)
{
	const float dx = through.x - origin.x;
	const float dy = through.y - origin.y;

	float t = INFINITY;
	if (dx > 0.f) {
		t = fminf(t, (1.f - origin.x) / dx);
	} else if (dx < 0.f) {
		t = fminf(t, -origin.x / dx);
	}
	if (dy > 0.f) {
		t = fminf(t, (1.f - origin.y) / dy);
	} else if (dy < 0.f) {
		t = fminf(t, -origin.y / dy);
	}
	if (t == INFINITY) {
		return through;
	}

	const struct xrt_vec2 edge = {origin.x + dx * t, origin.y + dy * t};
	return clamp_to_unit_square(edge);
}

/*
 * Perimeter parameter in [0,4) for a point on the unit square boundary,
 * walking (0,0) -> (1,0) -> (1,1) -> (0,1); corner c sits at parameter c.
 * Classifies by nearest edge: ray hits land within rounding of an edge,
 * not exactly on it.
 */
static float
unit_square_perimeter_pos(struct xrt_vec2 p)
{
	const float d_edge[4] = {p.y, 1.f - p.x, 1.f - p.y, p.x};
	uint32_t edge = 0;
	for (uint32_t e = 1; e < 4; e++) {
		if (d_edge[e] < d_edge[edge]) {
			edge = e;
		}
	}

	switch (edge) {
	case 0: return p.x;
	case 1: return 1.f + p.y;
	case 2: return 2.f + (1.f - p.x);
	default: return 3.f + (1.f - p.y);
	}
}

/*
 * Corners of the unit square strictly between perimeter positions s0 and s1,
 * walking from s0 in direction dir (+1/-1), in walk order. Returns the count.
 */
static uint32_t
corners_between(float s0, float s1, float dir, uint32_t out_corners[4])
{
	float span = fmodf((s1 - s0) * dir + 4.f, 4.f);

	/* Projections of a star-shaped boundary advance monotonically along the
	 * perimeter, but rounding can step coincident projections minutely
	 * backwards, which would alias to a near-full lap. No real segment of a
	 * dense boundary spans most of the perimeter; treat it as no step. */
	if (span > 3.5f) {
		span = 0.f;
	}

	uint32_t count = 0;
	for (uint32_t step = 1; step <= 4; step++) {
		const float c = dir > 0.f ? floorf(s0) + step : ceilf(s0) - step;
		if (fabsf(c - s0) >= span) {
			break;
		}
		out_corners[count++] = (uint32_t)(((int32_t)c % 4 + 4) % 4);
	}
	return count;
}

void
u_visibility_mask_from_uv_boundary(enum xrt_visibility_mask_type type,
                                   const struct xrt_vec2 *boundary_uv,
                                   uint32_t point_count,
                                   const struct xrt_fov *fov,
                                   struct xrt_visibility_mask **out_mask)
{
	*out_mask = NULL;

	const uint32_t n = point_count;
	if (n < 3) {
		U_LOG_E("need at least 3 boundary points, got %u", n);
		return;
	}

	struct xrt_vec2 *boundary = U_TYPED_ARRAY_CALLOC(struct xrt_vec2, n * 2);
	if (boundary == NULL) {
		U_LOG_E("failed to allocate boundary scratch space");
		return;
	}
	struct xrt_vec2 *edge = boundary + n;

	struct xrt_vec2 centroid = {0.f, 0.f};
	float area2 = 0.f;
	for (uint32_t i = 0; i < n; i++) {
		boundary[i] = clamp_to_unit_square(boundary_uv[i]);
		centroid.x += boundary[i].x / n;
		centroid.y += boundary[i].y / n;
	}
	for (uint32_t i = 0; i < n; i++) {
		const struct xrt_vec2 p = boundary[i];
		const struct xrt_vec2 q = boundary[(i + 1) % n];
		edge[i] = ray_to_unit_square_edge(centroid, p);
		area2 += p.x * q.y - q.x * p.y;
	}

	// The edge projections walk the square perimeter in the same direction
	// the boundary winds; corners they pass between consecutive projections
	// must be stitched into the hidden mesh or it would cut them off.
	const float dir = area2 >= 0.f ? 1.f : -1.f;
	uint32_t corner_total = 0;
	if (type == XRT_VISIBILITY_MASK_TYPE_HIDDEN_TRIANGLE_MESH) {
		for (uint32_t i = 0; i < n; i++) {
			uint32_t corners[4];
			corner_total += corners_between(unit_square_perimeter_pos(edge[i]),
			                                unit_square_perimeter_pos(edge[(i + 1) % n]), dir, corners);
		}
	}

	uint32_t nvertices = 0, nindices = 0;
	switch (type) {
	case XRT_VISIBILITY_MASK_TYPE_HIDDEN_TRIANGLE_MESH:
		// Per boundary point: the point itself and its perimeter projection,
		// plus the four square corners; per segment: a fan filling the area
		// between the boundary chord and the perimeter path.
		nvertices = n * 2 + 4;
		nindices = (n * 2 + corner_total) * 3;
		break;
	case XRT_VISIBILITY_MASK_TYPE_VISIBLE_TRIANGLE_MESH:
		// Fan from the centroid.
		nvertices = n + 1;
		nindices = n * 3;
		break;
	case XRT_VISIBILITY_MASK_TYPE_LINE_LOOP: //
		nvertices = n;
		nindices = n;
		break;
	}

	const size_t size =
	    sizeof(struct xrt_visibility_mask) + sizeof(uint32_t) * nindices + sizeof(struct xrt_vec2) * nvertices;
	struct xrt_visibility_mask *mask = U_CALLOC_WITH_CAST(struct xrt_visibility_mask, size);
	if (mask == NULL) {
		U_LOG_E("failed to allocate out xrt_visibility_mask");
		free(boundary);
		return;
	}

	mask->type = type;
	mask->index_count = nindices;
	mask->vertex_count = nvertices;

	uint32_t *indices = xrt_visibility_mask_get_indices(mask);
	struct xrt_vec2 *vertices = xrt_visibility_mask_get_vertices(mask);

	// Build the mask in UV space first.
	switch (type) {
	case XRT_VISIBILITY_MASK_TYPE_HIDDEN_TRIANGLE_MESH: {
		static const struct xrt_vec2 square_corners[4] = {{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}};
		for (uint32_t i = 0; i < n; i++) {
			vertices[i * 2 + 0] = boundary[i];
			vertices[i * 2 + 1] = edge[i];
		}
		for (uint32_t c = 0; c < 4; c++) {
			vertices[n * 2 + c] = square_corners[c];
		}

		uint32_t *tri = indices;
		for (uint32_t i = 0; i < n; i++) {
			const uint32_t j = (i + 1) % n;

			uint32_t corners[4];
			const uint32_t corner_count = corners_between(unit_square_perimeter_pos(edge[i]),
			                                              unit_square_perimeter_pos(edge[j]), dir, corners);

			// Perimeter path from this projection to the next one.
			uint32_t path[6];
			uint32_t path_len = 0;
			path[path_len++] = i * 2 + 1; // edge i
			for (uint32_t c = 0; c < corner_count; c++) {
				path[path_len++] = n * 2 + corners[c];
			}
			path[path_len++] = j * 2 + 1; // edge j

			for (uint32_t p = 0; p + 1 < path_len; p++) {
				*tri++ = i * 2 + 0; // boundary i
				*tri++ = path[p];
				*tri++ = path[p + 1];
			}
			*tri++ = i * 2 + 0; // boundary i
			*tri++ = j * 2 + 1; // edge j
			*tri++ = j * 2 + 0; // boundary j
		}
		assert(tri == indices + nindices);
		break;
	}
	case XRT_VISIBILITY_MASK_TYPE_VISIBLE_TRIANGLE_MESH:
		for (uint32_t i = 0; i < n; i++) {
			vertices[i] = boundary[i];
		}
		vertices[n] = centroid;
		for (uint32_t i = 0; i < n; i++) {
			const uint32_t j = (i + 1) % n;
			uint32_t *tri = &indices[i * 3];
			tri[0] = n; // centroid
			tri[1] = i;
			tri[2] = j;
		}
		break;
	case XRT_VISIBILITY_MASK_TYPE_LINE_LOOP:
		for (uint32_t i = 0; i < n; i++) {
			vertices[i] = boundary[i];
			indices[i] = i;
		}
		break;
	}

	free(boundary);

	// Convert UV to view tangent space, x right and y up per XR_KHR_visibility_mask.
	const float tan_left = tanf(fov->angle_left);
	const float tan_right = tanf(fov->angle_right);
	const float tan_up = tanf(fov->angle_up);
	const float tan_down = tanf(fov->angle_down);

	for (uint32_t i = 0; i < nvertices; i++) {
		const struct xrt_vec2 uv = vertices[i];
		vertices[i].x = tan_left + uv.x * (tan_right - tan_left);
		vertices[i].y = tan_up - uv.y * (tan_up - tan_down);
	}

	*out_mask = mask;
}
