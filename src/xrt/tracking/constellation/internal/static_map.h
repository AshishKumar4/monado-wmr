/*
 * World-anchored static-clutter map
 * Copyright 2026, G2-on-Linux project
 * SPDX-License-Identifier: BSL-1.0
 */
/*!
 * @file
 * @brief  Per-camera evidence-accumulating static-clutter map (H5 retention).
 *
 * White visible-light LEDs cannot be separated from a lit room by any per-frame static
 * feature (best single feature: contrast, AUC 0.807 on the hand-GT pool). The separating
 * signal is geometric-temporal: a static room spot (window, lamp, reflection) is predictable
 * to sub-pixel from head motion alone, while controller LEDs follow the device. Each camera
 * keeps a small map of world points (blob ray x fitted depth, refit on every match by a
 * depth sweep) re-projected with the live head pose each frame, so dwell accumulation
 * survives head rotation AND translation plus detection flicker (clutter only re-detects at
 * the same spot 63%/frame; entries tolerate 400 ms gaps).
 *
 * A blob matched to an entry whose dwell exceeds the threshold is classified
 * BLOB_RETENTION_STATIC_CLUTTER — positively identified room furniture. Blobs within the
 * exemption radius of any device's predicted or last-seen LEDs are BLOB_RETENTION_DEVICE_NEAR
 * and never demoted: resting hands are MORE world-static than clutter (instantaneous static
 * culling false-suppresses 38-73% of LEDs; without the exemption the map false-suppresses
 * 19.6% at 1 s dwell). The class is consumed as a soft priority (cold-search ordering, blob-cap
 * retention) — suppression never deletes a blob.
 *
 * @ingroup constellation
 */
#pragma once

#include "xrt/xrt_defines.h"

#include "blobwatch.h"
#include "camera_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define STATIC_MAP_MAX_ENTRIES 256
/* Image-space match gate between a re-projected entry and a blob centroid. Static-prediction
 * residual for clutter measures 0.53 px median / 4.5 px p90 on the clutter benchmark. */
#define STATIC_MAP_MATCH_PX 3.0f
/* Entries survive detection flicker / short head-pose gaps up to this long. */
#define STATIC_MAP_GAP_NS ((int64_t)400000000)
/* Accumulated dwell at which a world-still spot is positively static clutter. 1 s suppresses
 * 46.3% of clutter observations at 0.00% LED false-suppression (with the device exemption). */
#define STATIC_MAP_STATIC_DWELL_S 1.0f
/* Exemption radius around projected device LEDs (predicted pose, tracked or not, and the
 * last optically-seen pose). Load-bearing: see the file comment. */
#define STATIC_MAP_EXEMPT_RADIUS_PX 25.0f

struct static_map_entry
{
	struct xrt_vec3 world_point;
	/* The matched blob's normalized camera ray and the camera's world pose at the previous
	 * match, kept so the next match can refit depth by sweeping this ray. */
	struct xrt_vec3 ray_prev;
	struct xrt_pose P_world_cam_prev;
	uint64_t first_seen_ns;
	uint64_t last_seen_ns;
	uint32_t hits;
};

struct static_map
{
	int num_entries;
	struct static_map_entry entries[STATIC_MAP_MAX_ENTRIES];
};

/*!
 * Update the map with one frame's blobs and write each blob's retention_class /
 * static_dwell_s. @p P_world_cam is the camera's pose in the (OpenCV) world frame and
 * @p P_cam_world its inverse; @p exempt_px are projected device-LED pixel positions
 * (predicted + last-seen poses of every device, tracked or not). Deterministic: identical
 * inputs produce identical classes and map state.
 */
void
static_map_update(struct static_map *sm,
                  blobservation *ob,
                  const struct camera_model *calib,
                  const struct xrt_pose *P_world_cam,
                  const struct xrt_pose *P_cam_world,
                  uint64_t frame_ts_ns,
                  const struct xrt_vec2 *exempt_px,
                  int n_exempt);

#ifdef __cplusplus
}
#endif
