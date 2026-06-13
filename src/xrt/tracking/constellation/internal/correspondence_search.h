// Copyright 2020-2023 Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
/*
 * Ported from OpenHMD - Free and Open Source API and drivers for immersive technology.
 */
/*!
 * @file
 * @brief  Ab-initio blob<->LED correspondence search
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup constellation
 */
#pragma once

#include "tracking/t_led_models.h"

#include "blobwatch.h"
#include "camera_model.h"
#include "pose_metrics.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_BLOB_SEARCH_DEPTH 5
#define CORRESPONDENCE_SEARCH_MAX_RESULTS 8

enum correspondence_search_flags
{
	CS_FLAG_NONE = 0x0,
	CS_FLAG_SHALLOW_SEARCH = 0x1, /* do quick search @ depth 1-2 neighbour depth */
	CS_FLAG_DEEP_SEARCH = 0x2,    /* do deeper searches @ up to MAX_LED_SEARCH_DEPTH/MAX_BLOB_SEARCH_DEPTH */
	CS_FLAG_STOP_FOR_STRONG_MATCH =
	    0x4, /* Stop searching if a strong match is found, otherwise search all and return best match */
	CS_FLAG_MATCH_ALL_BLOBS =
	    0x8, /* Allow matching against all blobs, not just unlabelled ones or for the current device */
	CS_FLAG_HAVE_POSE_PRIOR = 0x10, /* If the input obj_cam_pose contains a valid prior */
	CS_FLAG_TRUST_PRIOR_ORIENT =
	    0x20, /* Apply the soft anisotropic prior-consistency cost (mirror-flip re-rank): add a robust prior
	             penalty to each candidate's reprojection cost so the prior-consistent twin out-ranks the
	             flip, without ever dropping a candidate. The penalty is the anisotropic Mahalanobis distance
	             of the candidate orientation from the prior (TILT scaled by the tight driftless sigma, YAW by
	             the live fusion sigma), Huber-robustified. Requires CS_FLAG_HAVE_POSE_PRIOR + an up_vector
	             (the world-up in camera frame). */
	/* 0x40 retired (BOUNDED_SEARCH; subsumed by the per-pass work allowance). Never reuse —
	 * search_flags is recorded verbatim in the search telemetry stream. */
	CS_FLAG_RETURN_BEST_PARTIAL =
	    0x80, /* If no GOOD pose was found, return the best tight non-GOOD candidate so callers can use its
	             matched LED evidence for partial/position-only fusion without accepting a full pose lock. */
};

struct cs_image_point
{
	struct blob *blob;

	double point_homog[3]; // Homogeneous version of the point
	double size[2];        // w/h of the blob, in homogeneous coordinates
	double max_dist;       // norm of (W/H) for distance checks

	/* List of the nearest blobs, filtered for the active model */
	int num_neighbours;
	struct cs_image_point *neighbours[MAX_BLOB_SEARCH_DEPTH];
};

struct correspondence_search_result
{
	struct xrt_pose pose;
	struct pose_metrics score;
};

struct cs_model_info
{
	int id;

	struct t_constellation_search_model *model;

	double best_pose_found_time; /* Time (in secs) at which the best pose was found */
	int best_pose_blob_depth;    /* Blob neighbor depth the best pose is from */
	int best_pose_led_depth;     /* LED neigbour depth the best pose is from */
	struct xrt_pose best_pose;
	enum pose_match_flags match_flags;

	struct pose_metrics best_score;
	double best_prior_cost; /* soft prior penalty (px^2) of best_pose; folded into the candidate ranking */
	struct pose_metrics best_any_score;
	struct xrt_pose best_any_pose;
	int best_any_pose_blob_depth;
	int best_any_pose_led_depth;
	int result_count;
	struct correspondence_search_result results[CORRESPONDENCE_SEARCH_MAX_RESULTS];
	double result_prior_cost[CORRESPONDENCE_SEARCH_MAX_RESULTS];

	/* Admissible-bound prune state, cached when best_pose updates (CS_FLAG_TRUST_PRIOR_ORIENT). A new
	 * candidate's prior penalty alone lower-bounds its combined cost (reproj >= 0), so a candidate whose
	 * cheap prior cost / orient error already loses every score_is_better branch against the current GOOD
	 * best is skipped before the expensive reprojection — the accepted pose is unchanged. */
	double best_combined_cost;  /* best_score.reprojection_error + best_prior_cost */
	double best_cost_per_led;   /* best_combined_cost / best_score.matched_blobs */
	double best_orient_err_len; /* |ln(best^-1 . prior)| (the score_is_better orient tiebreak metric) */

	/* Search parameters */
	double search_start_time;
	/* Pre-assigned work-unit allowance for this pass (1 unit = 1 P3P trial, pose check = 5;
	 * see correspondence_search.c). The search stops — never overspends — when a charge
	 * would exceed it. */
	uint32_t work_allowance;
	int led_depth;
	int led_index;
	int blob_index;

	enum correspondence_search_flags search_flags;
	int min_led_depth, max_led_depth;
	int max_blob_depth;

	/* Valid when CS_FLAG_HAVE_POSE_PRIOR is set */
	struct xrt_pose pose_prior;
	struct xrt_vec3 *pos_error_thresh;
	struct xrt_vec3 *rot_error_thresh;

	/* Anisotropic soft prior-orientation cost (CS_FLAG_TRUST_PRIOR_ORIENT). up_vector = world-up in the
	 * camera frame (the swing/twist axis). sigma_tilt_rad = the tight DRIFTLESS gravity-swing (tilt) scale;
	 * sigma_yaw_rad = the live fusion yaw 1-sigma (large when stale -> the yaw term vanishes). huber_knee_sigma
	 * bends a gross outlier's penalty to linear; cost_weight commensurates it with the reprojection error.
	 * See pose_metrics_prior_orient_cost. */
	struct xrt_vec3 up_vector;
	float sigma_tilt_rad;
	float sigma_yaw_rad;
	float huber_knee_sigma;
	float cost_weight;
};

struct correspondence_search_diagnostics
{
	uint32_t input_blobs;
	uint32_t searchable_anchors;
	uint32_t filtered_anchors;
	uint32_t anchors_with_3_neighbours;
	uint32_t neighbour_links;
	uint32_t num_trials;
	uint32_t num_pose_checks;
	uint32_t num_pose_checks_pruned;
	int32_t min_led_depth;
	int32_t max_led_depth;
	int32_t max_blob_depth;
	int32_t best_pose_blob_depth;
	int32_t best_pose_led_depth;
	int32_t best_any_pose_blob_depth;
	int32_t best_any_pose_led_depth;
	struct pose_metrics best_any_score;
	struct xrt_pose best_any_pose;
	uint32_t best_any_match_flags;
	uint32_t best_any_leds_visible;
	uint32_t best_any_blobs_matched;
	uint32_t best_any_unmatched_blobs;
	float best_any_reproj_err_px;
	uint32_t work_spent;     /* work units charged by this pass (trials + 5x pose checks) */
	uint8_t budget_exhausted; /* 1 if a charge was denied: the pass hit its pre-assigned allowance */
};

struct correspondence_search
{
	int num_points;
	int points_capacity;
	struct cs_image_point *points;
	struct blob *blobs; /* Original blobs structs [num_points] */

	unsigned int num_trials;
	unsigned int num_pose_checks;
	unsigned int num_pose_checks_pruned; /* candidates skipped by the admissible-bound prior prune */
	uint32_t work_spent;                 /* work units charged in the current pass */
	bool budget_exhausted;               /* a charge was denied: the pass hit its allowance */

	struct camera_model *calib;

	/* List of the nearest blobs for each blob */
	struct cs_image_point *blob_neighbours[MAX_BLOBS_PER_FRAME][MAX_BLOBS_PER_FRAME];

	struct correspondence_search_diagnostics last_diag;
};

struct correspondence_search *
correspondence_search_new(struct camera_model *camera_calib);
void
correspondence_search_set_blobs(struct correspondence_search *cs, struct blob *blobs, int num_blobs);

void
correspondence_search_free(struct correspondence_search *cs);

int
correspondence_search_find_pose_candidates(struct correspondence_search *cs,
                                           struct t_constellation_search_model *model,
                                           enum correspondence_search_flags search_flags,
                                           uint32_t work_allowance,
                                           struct xrt_pose *pose,
                                           struct xrt_vec3 *pos_error_thresh,
                                           struct xrt_vec3 *rot_error_thresh,
                                           struct xrt_vec3 *up_vector,
                                           float sigma_tilt_rad,
                                           float sigma_yaw_rad,
                                           float huber_knee_sigma,
                                           float cost_weight,
                                           struct correspondence_search_result *results,
                                           int max_results);
void
correspondence_search_get_last_diagnostics(struct correspondence_search *cs,
                                           struct correspondence_search_diagnostics *out_diag);

#ifdef __cplusplus
}
#endif
