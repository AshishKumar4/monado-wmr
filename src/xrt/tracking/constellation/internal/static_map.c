/*
 * World-anchored static-clutter map
 * Copyright 2026, G2-on-Linux project
 * SPDX-License-Identifier: BSL-1.0
 */
/*!
 * @file
 * @brief  Per-camera evidence-accumulating static-clutter map (H5 retention).
 * @ingroup constellation
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "math/m_api.h"

#include "static_map.h"

/* Depth sweep for fitting a world point to a blob ray: the depth whose world point best
 * re-projects onto the next observation wins. Spans arm's length to far walls. */
static const float depth_sweep_m[] = {0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 3.0f, 5.0f, 10.0f, 30.0f};
#define DEPTH_SWEEP_N ((int)(sizeof(depth_sweep_m) / sizeof(depth_sweep_m[0])))
/* Provisional depth for a brand-new entry (refit on its first re-observation). */
#define PROVISIONAL_DEPTH_M 3.0f
/* Minimum camera-frame depth for a valid entry projection. */
#define MIN_PROJECT_Z_M 0.1f
/* Bound on (entry, blob) candidate pairs per frame; ~1.5x the blob cap covers the measured
 * regime (one continuing spot ~= one pair) with deterministic truncation if ever exceeded. */
#define MAX_MATCH_PAIRS 1024

struct match_pair
{
	float dist;
	int16_t entry;
	int16_t blob;
};

static int
match_pair_cmp(const void *va, const void *vb)
{
	const struct match_pair *a = va, *b = vb;
	if (a->dist != b->dist) {
		return a->dist < b->dist ? -1 : 1;
	}
	if (a->entry != b->entry) {
		return a->entry < b->entry ? -1 : 1;
	}
	return a->blob < b->blob ? -1 : 1;
}

static bool
project_world_point(const struct camera_model *calib,
                    const struct xrt_pose *P_cam_world,
                    const struct xrt_vec3 *world_point,
                    float *out_u,
                    float *out_v)
{
	struct xrt_vec3 cam_point;
	math_pose_transform_point(P_cam_world, world_point, &cam_point);
	if (cam_point.z < MIN_PROJECT_Z_M) {
		return false;
	}
	return t_camera_models_project(&calib->calib, cam_point.x, cam_point.y, cam_point.z, out_u, out_v);
}

void
static_map_update(struct static_map *sm,
                  blobservation *ob,
                  const struct camera_model *calib,
                  const struct xrt_pose *P_world_cam,
                  const struct xrt_pose *P_cam_world,
                  uint64_t frame_ts_ns,
                  const struct xrt_vec2 *exempt_px,
                  int n_exempt)
{
	/* Expire entries unseen for longer than the flicker/gap tolerance (order-preserving
	 * compaction keeps the update deterministic). A >gap head-pose outage simply empties
	 * the map: retention degrades to today's behaviour, never to false suppression. */
	int kept = 0;
	for (int j = 0; j < sm->num_entries; j++) {
		if ((int64_t)(frame_ts_ns - sm->entries[j].last_seen_ns) <= STATIC_MAP_GAP_NS) {
			if (kept != j) {
				sm->entries[kept] = sm->entries[j];
			}
			kept++;
		}
	}
	sm->num_entries = kept;

	/* Project alive entries with the CURRENT head pose (survives rotation and translation). */
	float entry_u[STATIC_MAP_MAX_ENTRIES];
	float entry_v[STATIC_MAP_MAX_ENTRIES];
	bool entry_ok[STATIC_MAP_MAX_ENTRIES];
	for (int j = 0; j < sm->num_entries; j++) {
		entry_ok[j] =
		    project_world_point(calib, P_cam_world, &sm->entries[j].world_point, &entry_u[j], &entry_v[j]);
	}

	/* Greedy globally-nearest-first matching of blobs to entries within the match gate. */
	struct match_pair pairs[MAX_MATCH_PAIRS];
	int n_pairs = 0;
	for (int j = 0; j < sm->num_entries && n_pairs < MAX_MATCH_PAIRS; j++) {
		if (!entry_ok[j]) {
			continue;
		}
		for (int i = 0; i < ob->num_blobs && n_pairs < MAX_MATCH_PAIRS; i++) {
			const float du = entry_u[j] - ob->blobs[i].x;
			const float dv = entry_v[j] - ob->blobs[i].y;
			const float dist = sqrtf(du * du + dv * dv);
			if (dist <= STATIC_MAP_MATCH_PX) {
				pairs[n_pairs++] = (struct match_pair){dist, (int16_t)j, (int16_t)i};
			}
		}
	}
	qsort(pairs, (size_t)n_pairs, sizeof(pairs[0]), match_pair_cmp);

	int assigned_entry[MAX_BLOBS_PER_FRAME];
	bool entry_used[STATIC_MAP_MAX_ENTRIES] = {false};
	for (int i = 0; i < ob->num_blobs; i++) {
		assigned_entry[i] = -1;
	}
	for (int p = 0; p < n_pairs; p++) {
		const int j = pairs[p].entry, i = pairs[p].blob;
		if (entry_used[j] || assigned_entry[i] >= 0) {
			continue;
		}
		entry_used[j] = true;
		assigned_entry[i] = j;
	}

	for (int i = 0; i < ob->num_blobs; i++) {
		struct blob *b = &ob->blobs[i];

		/* The blob's normalized camera ray (undistorted), reused for depth fits. */
		float nx = 0.0f, ny = 0.0f;
		t_camera_models_undistort(&calib->calib, b->x, b->y, &nx, &ny);
		const struct xrt_vec3 ray = {nx, ny, 1.0f};

		bool exempt = false;
		for (int k = 0; k < n_exempt && !exempt; k++) {
			const float du = exempt_px[k].x - b->x;
			const float dv = exempt_px[k].y - b->y;
			exempt = du * du + dv * dv <=
			         STATIC_MAP_EXEMPT_RADIUS_PX * STATIC_MAP_EXEMPT_RADIUS_PX;
		}

		float dwell_s = 0.0f;
		struct static_map_entry *e = NULL;
		if (assigned_entry[i] >= 0) {
			e = &sm->entries[assigned_entry[i]];
			/* Depth refit: sweep the entry's previous ray (in its previous camera pose)
			 * and keep the world point that best explains THIS observation. */
			float best_err = 1e9f;
			struct xrt_vec3 best_point;
			bool have_best = false;
			for (int z = 0; z < DEPTH_SWEEP_N; z++) {
				struct xrt_vec3 cam_prev = {e->ray_prev.x * depth_sweep_m[z],
				                            e->ray_prev.y * depth_sweep_m[z],
				                            e->ray_prev.z * depth_sweep_m[z]};
				struct xrt_vec3 world_cand;
				math_pose_transform_point(&e->P_world_cam_prev, &cam_prev, &world_cand);
				float u, v;
				if (!project_world_point(calib, P_cam_world, &world_cand, &u, &v)) {
					continue;
				}
				const float err = hypotf(u - b->x, v - b->y);
				if (err < best_err) {
					best_err = err;
					best_point = world_cand;
					have_best = true;
				}
			}
			if (have_best && best_err <= STATIC_MAP_MATCH_PX) {
				e->world_point = best_point;
			}
			e->last_seen_ns = frame_ts_ns;
			e->hits++;
			dwell_s = (float)((double)(frame_ts_ns - e->first_seen_ns) * 1e-9);
		} else {
			/* New entry at a provisional depth; the first re-observation refits it.
			 * At capacity, evict the entry closest to expiry (never one updated this
			 * frame); ties go to the lowest index — deterministic. */
			int slot = sm->num_entries;
			if (slot >= STATIC_MAP_MAX_ENTRIES) {
				slot = -1;
				uint64_t oldest = frame_ts_ns;
				for (int j = 0; j < sm->num_entries; j++) {
					if (sm->entries[j].last_seen_ns < oldest) {
						oldest = sm->entries[j].last_seen_ns;
						slot = j;
					}
				}
			} else {
				sm->num_entries++;
			}
			if (slot >= 0) {
				e = &sm->entries[slot];
				struct xrt_vec3 cam_point = {ray.x * PROVISIONAL_DEPTH_M, ray.y * PROVISIONAL_DEPTH_M,
				                             ray.z * PROVISIONAL_DEPTH_M};
				math_pose_transform_point(P_world_cam, &cam_point, &e->world_point);
				e->first_seen_ns = frame_ts_ns;
				e->last_seen_ns = frame_ts_ns;
				e->hits = 1;
			}
		}
		if (e != NULL) {
			e->ray_prev = ray;
			e->P_world_cam_prev = *P_world_cam;
		}

		b->static_dwell_s = dwell_s;
		if (exempt) {
			b->retention_class = BLOB_RETENTION_DEVICE_NEAR;
		} else if (dwell_s >= STATIC_MAP_STATIC_DWELL_S) {
			b->retention_class = BLOB_RETENTION_STATIC_CLUTTER;
		} else {
			b->retention_class = BLOB_RETENTION_FRESH;
		}
	}
}
