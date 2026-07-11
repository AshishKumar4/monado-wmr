// Copyright 2020 Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 */
/*!
 * @file
 * @brief  Ab-initio blob<->LED correspondence search
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup constellation
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "math/m_vec3.h"
#include "os/os_time.h"

#include "lambdatwist/lambdatwist_p3p.h"
#include "correspondence_search.h"

#define DUMP_SCENE 0
#define DUMP_BLOBS 0
#define DUMP_FULL_DEBUG 0
#define DUMP_FULL_LOG 0
#define DUMP_TIMING 0
#define CHECK_ALL_PROJECTIONS 0

#define MAX_LED_SEARCH_DEPTH 6

/* Deterministic work-unit budget currency. 1 unit = 1 P3P trial (measured 228ns/trial on the
 * band-2 clutter capture: assoc_ms = 4.22ms + 228ns/trial, R²=0.987); a full pose check
 * (projection + blob match) costs 5 units (measured 1095ns ≈ 4.8 trial-equivalents; the
 * admissible-bound pruned exits stay free). A charge is denied — never partially granted —
 * when it would exceed the pass's pre-assigned allowance, so a pass can NEVER overspend and
 * the caller's per-frame budget holds exactly. */
#define CS_WORK_UNITS_PER_POSE_CHECK 5

/* This file implements a brute-force correspondence search between LED models and observed IR LED blobs.
 *
 * The goal is to iterate over sets of LED and blob candidates, and test them as correspondences. Groups of 4
 * LEDs and Blobs are collected. The first 3 points are used with the Lambdatwist P3P to calculate a hypothesis
 * pose of the object. If the 4th point matches the hypothesis, it proceeds to a full pose check to evaluate
 * how many other LED/blob points match (inliers), and how closely (reprojection error).
 *
 * To speed up the search, the LED and blobs are iterated using nearest neighbour lists, as blobs that are nearby
 * are much more likely to match LEDs that are near each other.
 *
 * LED models provide a set of 3D LED points and their normals. The constellation_search_model helper takes the set
 * of LED points (constellation_led_model) and for each one calculates an search candidate list
 * of neighbour LEDs (LEDs with normals within 90° of the anchor LED) sorted by euclidean distance.
 *
 * Before each search, the caller provides the set of 2D blobs to match against, which are given similar
 * treatment - for each blob the set of neighbours is calculated, sorted by increasing distance.
 *
 * If available the vertical alignment of the device is used as a check on valid poses. Since the IMU can generally
 * extract the gravity vector of each device with some variance, poses that don't match the required gravity direction
 * are rejected, avoiding the full-pose check.
 *
 * Further, if CS_FLAGS_STOP_FOR_STRONG_MATCH is passed and a 'strong' pose match is found, the search is stopped
 * and the result returned immediately.
 * A strong match is defined in the pose-helper, as 7 or more LEDs matching observed blobs within 1.5 pixels/led
 * reprojection error.
 *
 * A full search of all possible LED and blob combinations is prohibitive, so the search is limited to matching
 * neighbours up to MAX_LED_SEARCH_DEPTH deep for LEDs - ie, for MAX_LED_SEARCH_DEPTH=6, we check each LED and
 * combinations of 3 from the 6 nearest LEDs. For blobs, we check MAX_BLOB_SEARCH_DEPTH nearest neighbours, but filter
 * the list to ignore blobs that are already assigned to another device (unless CS_FLAG_MATCH_ALL_BLOBS is set).
 *
 * In general, there are many more possible LEDs than there are visible blobs, and most of the LEDs will not be
 * visible in any given pose. blobs, on the other hand are (by definition) visible - if we find the right LEDs, they'll
 * match. As such, it's better to loop over some 1-2 depth combinations of blobs and test them against nearest LEDs,
 * before proceeding to any deeper search depths. If CS_FLAG_SHALLOW_SEARCH is provided, only depth 1/2 is checked.
 *
 */

struct correspondence_search *
correspondence_search_new(struct camera_model *camera_calib)
{
	struct correspondence_search *cs = calloc(1, sizeof(struct correspondence_search));
	if (cs == NULL) {
		return NULL;
	}
	cs->calib = camera_calib;
	return cs;
}

#if DUMP_FULL_DEBUG
#define DEBUG(s, ...) fprintf(stderr, s, __VA_ARGS__)
#else
#define DEBUG(s, ...)
#endif

#if DUMP_TIMING
#define DEBUG_TIMING(s, ...) printf(s, __VA_ARGS__)
#else
#define DEBUG_TIMING(s, ...)
#endif

#undef LOG
#if DUMP_FULL_LOG
#define LOG(s, ...) fprintf(stderr, s, __VA_ARGS__)
#else
#define LOG(s, ...)
#endif

/* Charge @units against the pass allowance. Denial latches cs->budget_exhausted, which the
 * search recursion seams consult to unwind; the spend depends only on the pass inputs +
 * allowance, so identical inputs charge identically (the determinism contract). */
static bool
cs_charge_work(struct correspondence_search *cs, const struct cs_model_info *mi, uint32_t units)
{
	if (units > mi->work_allowance - cs->work_spent) {
		cs->budget_exhausted = true;
		return false;
	}
	cs->work_spent += units;
	return true;
}

static void
undistort_blob_points(struct blob *blobs, int num_blobs, struct xrt_vec2 *out_points, struct camera_model *calib)
{
	for (int i = 0; i < num_blobs; i++) {
		t_camera_models_undistort(&calib->calib, blobs[i].x, blobs[i].y, &out_points[i].x, &out_points[i].y);
	}
}

static inline double
blob_distance_sq(const struct cs_image_point *point, const struct cs_image_point *anchor)
{
	const double dy = (double)point->blob->y - anchor->blob->y;
	const double dx = (double)point->blob->x - anchor->blob->x;
	return dy * dy + dx * dx;
}

static void
sort_blobs_by_distance(struct cs_image_point **points, int num_points, const struct cs_image_point *anchor)
{
	for (int i = 1; i < num_points; i++) {
		struct cs_image_point *point = points[i];
		const double distance = blob_distance_sq(point, anchor);
		int j = i - 1;

		while (j >= 0 && blob_distance_sq(points[j], anchor) > distance) {
			points[j + 1] = points[j];
			j--;
		}

		points[j + 1] = point;
	}
}

#if CHECK_ALL_PROJECTIONS
static void
project_led_point(struct xrt_vec3 *led_pos,
                  struct xrt_pose *pose,
                  struct camera_model *calib,
                  struct xrt_vec2 *out_point)
{
	struct xrt_vec3 tmp;
	math_pose_transform_point(pose, led_pos, &tmp);
	t_camera_models_project(&calib->calib, tmp.x, tmp.y, tmp.z, &out_point->x, &out_point->y);
}
#endif

void
correspondence_search_set_blobs(struct correspondence_search *cs, struct blob *blobs, int num_blobs)
{
	int i;
	struct xrt_vec2 undistorted_points[MAX_BLOBS_PER_FRAME];
	struct cs_image_point *blob_list[MAX_BLOBS_PER_FRAME];

	assert(num_blobs <= MAX_BLOBS_PER_FRAME);

	if (num_blobs > cs->points_capacity) {
		struct cs_image_point *points = realloc(cs->points, num_blobs * sizeof(struct cs_image_point));
		if (points == NULL) {
			/* OOM on the frame thread: keep the old buffer and take only the blobs that
			 * fit this frame instead of dereferencing NULL. */
			num_blobs = cs->points_capacity;
		} else {
			cs->points = points;
			cs->points_capacity = num_blobs;
		}
	}
	cs->num_points = num_blobs;
	cs->blobs = blobs;

	/* Undistort points so we can project / match properly */
	undistort_blob_points(blobs, num_blobs, undistorted_points, cs->calib);

#if DUMP_BLOBS
	printf("Building blobs search array\n");
#endif
	for (i = 0; i < num_blobs; i++) {
		struct cs_image_point *p = cs->points + i;
		struct blob *b = blobs + i;

		p->point_homog[0] = undistorted_points[i].x;
		p->point_homog[1] = undistorted_points[i].y;
		p->point_homog[2] = 1;

		p->size[0] = b->width / cs->calib->calib.fx;
		p->size[1] = b->height / cs->calib->calib.fy;
		p->max_dist = sqrt(p->size[0] * p->size[0] + p->size[1] * p->size[1]);

		p->blob = b;
		blob_list[i] = p;

#if DUMP_BLOBS
		printf("Blob %u = (%f,%f %ux%u) -> (%f, %f) %f x %f (homog (%f, %f) %f x %f) (LED id %d)\n", i, b->x,
		       b->y, b->width, b->height, p->point_homog[0] * cs->calib->calib.fx,
		       p->point_homog[1] * cs->calib->calib.fy, p->size[0] * cs->calib->calib.fx,
		       p->size[1] * cs->calib->calib.fy, p->point_homog[0], p->point_homog[1], p->size[0], p->size[1],
		       b->led_id);
#endif
	}

	/* Now the blob_list is populated,
	 * loop over the blob list and for each, prepare a list of neighbours sorted by distance */
	for (i = 0; i < cs->num_points; i++) {
		struct cs_image_point *anchor = cs->points + i;

		/* Sort the blobs by proximity to anchor blob */
		sort_blobs_by_distance(blob_list, cs->num_points, anchor);
		memcpy(cs->blob_neighbours[i], blob_list, cs->num_points * sizeof(struct cs_image_point *));
	}
}

void
correspondence_search_free(struct correspondence_search *cs)
{
	if (cs->points)
		free(cs->points);
	free(cs);
}

#if DUMP_SCENE
static void
dump_pose(struct correspondence_search *cs,
          struct t_constellation_search_model *model,
          struct xrt_pose *pose,
          struct cs_model_info *mi)
{
	int i;
	struct t_constellation_led_model *leds = model->led_model;

	printf("pose_points[%i] = [\n", mi->id);
	for (i = 0; i < leds->num_leds; i++) {
		struct xrt_vec3 pos, dir;

		/* Project HMD LED into the image (no distortion) */
		math_quat_rotate_vec3(&pose->orientation, &leds->leds[i].pos, &pos);
		math_vec3_accum(&pose->position, &pos);

		if (pos.z < 0)
			continue; // Can't be behind the camera

		math_vec3_scalar_mul(1.0 / pos.z, &pos);

		double x = pos.x;
		double y = pos.y;

		math_vec3_normalize(&pos);

		math_quat_rotate_vec3(&pose->orientation, &leds->leds[i].dir, &dir);
		math_vec3_normalize(&dir);

		double facing_dot;
		facing_dot = m_vec3_dot(pos, dir);

		if (facing_dot < 0) {
			printf("  (%f,%f),\n", x, y);
		}
	}
	printf("]\n");
}
#endif

/* |ln(pose^-1 . prior)|: the orientation-error magnitude pose_metrics_score_is_better_pose_prior uses for
 * its final tiebreak, computed from orientations alone (no reprojection) — same operations as
 * check_pose_prior so the prune metric cannot drift from the comparator's. */
static double
prior_orient_err_len(const struct xrt_pose *pose, const struct xrt_pose *prior)
{
	struct xrt_quat orient_diff;
	math_quat_unrotate(&pose->orientation, &prior->orientation, &orient_diff);
	math_quat_normalize(&orient_diff);
	struct xrt_vec3 err;
	math_quat_ln(&orient_diff, &err);
	return m_vec3_len(err);
}

/* Admissible-bound prune: skip the expensive reprojection for a candidate that provably loses every
 * pose_metrics_score_is_better_pose_prior branch against the current GOOD best. The candidate's prior
 * penalty alone lower-bounds its combined cost (reproj >= 0) and its matched-blob count is bounded by the
 * available blobs, so the cheap prior_cost + orient error decide it WITHOUT the projection. Only active once
 * a GOOD best exists (else branch 1 — old-not-GOOD — could accept any GOOD candidate). With no prior trust
 * prior_cost == 0 -> branch-4 guard fails -> never prunes (cold start untouched). */
static bool
prune_by_prior_bound(const struct cs_model_info *mi, double prior_cost, double cand_orient_err_len, int num_blobs)
{
	if (!(mi->match_flags & POSE_MATCH_GOOD) || num_blobs <= 0)
		return false;
	/* Branch 4 (equal blobs, lower combined cost): new_cost >= prior_cost, so prior_cost >= best_combined
	 * makes it impossible. */
	if (prior_cost < mi->best_combined_cost)
		return false;
	/* Branches 2,3 (more matched blobs): min new_cost_per_led = prior_cost / new_matched >= prior_cost /
	 * num_blobs; the looser 1.1x bound dominates the 1.0x one. */
	if (prior_cost / num_blobs < mi->best_cost_per_led * 1.1)
		return false;
	/* Branch 5 (orient tiebreak): only fires when the candidate's orient error is strictly smaller. */
	if (cand_orient_err_len < mi->best_orient_err_len)
		return false;
	return true;
}

static bool
diagnostic_score_is_better_any(const struct pose_metrics *old_score, const struct pose_metrics *new_score)
{
	if (old_score->visible_leds == 0 && old_score->matched_blobs == 0) {
		return true;
	}
	if (new_score->matched_blobs > old_score->matched_blobs) {
		return true;
	}
	if (new_score->matched_blobs < old_score->matched_blobs) {
		return false;
	}
	const float old_per_led =
	    old_score->matched_blobs > 0 ? old_score->reprojection_error / old_score->matched_blobs : INFINITY;
	const float new_per_led =
	    new_score->matched_blobs > 0 ? new_score->reprojection_error / new_score->matched_blobs : INFINITY;
	if (new_per_led < old_per_led) {
		return true;
	}
	if (new_per_led > old_per_led) {
		return false;
	}
	return new_score->unmatched_blobs < old_score->unmatched_blobs;
}

static bool
search_result_duplicate(const struct correspondence_search_result *a, const struct xrt_pose *pose)
{
	struct xrt_vec3 dp = a->pose.position;
	math_vec3_subtract(&pose->position, &dp);
	double d = fabs((double)a->pose.orientation.x * pose->orientation.x +
	                (double)a->pose.orientation.y * pose->orientation.y +
	                (double)a->pose.orientation.z * pose->orientation.z +
	                (double)a->pose.orientation.w * pose->orientation.w);
	d = d > 1.0 ? 1.0 : d;
	return m_vec3_len(dp) < 0.02 && (2.0 * acos(d)) < (5.0 * M_PI / 180.0);
}

static bool
search_result_better(const struct correspondence_search_result *old_result,
                     double old_prior_cost,
                     const struct pose_metrics *new_score,
                     double new_prior_cost)
{
	struct pose_metrics old_score = old_result->score;
	struct pose_metrics score = *new_score;
	return pose_metrics_score_is_better_pose_prior(&old_score, old_prior_cost, &score, new_prior_cost);
}

static bool
best_partial_result_usable(const struct pose_metrics *score)
{
	if (score == NULL || POSE_HAS_FLAGS(score, POSE_MATCH_GOOD) || score->matched_blobs < 4) {
		return false;
	}
	if (!POSE_HAS_FLAGS(score, POSE_MATCH_LED_IDS)) {
		return false;
	}
	const double reproj_per_match =
	    score->matched_blobs > 0 ? score->reprojection_error / (double)score->matched_blobs : INFINITY;
	if (!isfinite(reproj_per_match) || reproj_per_match > 3.0) {
		return false;
	}
	return true;
}

static void
append_best_partial_result(struct cs_model_info *mi)
{
	if (mi == NULL || (mi->search_flags & CS_FLAG_RETURN_BEST_PARTIAL) == 0 || mi->result_count > 0 ||
	    !best_partial_result_usable(&mi->best_any_score)) {
		return;
	}

	mi->results[0].pose = mi->best_any_pose;
	mi->results[0].score = mi->best_any_score;
	mi->result_prior_cost[0] = 0.0;
	mi->result_count = 1;
}

static void
insert_search_result(struct cs_model_info *mi,
                     const struct xrt_pose *pose,
                     const struct pose_metrics *score,
                     double prior_cost)
{
	for (int i = 0; i < mi->result_count; i++) {
		if (search_result_duplicate(&mi->results[i], pose)) {
			if (search_result_better(&mi->results[i], mi->result_prior_cost[i], score, prior_cost)) {
				mi->results[i].pose = *pose;
				mi->results[i].score = *score;
				mi->result_prior_cost[i] = prior_cost;
			}
			/* KNOWN, CALIBRATED-IN (audit 2026-07-10 R1-M1 i2): same in-place
			 * duplicate-improvement-without-re-bubble shape as
			 * association_insert_hypothesis, and results[0] is consumed downstream.
			 * The bubble fix for the sibling was REFUTED on the standing regression
			 * gate (results/w2-fold-20260711/m1-insert-order-adjudication/); the same
			 * verdict applies here — re-open only matrix-gated in the W5 rework. */
			return;
		}
	}

	if (mi->result_count < CORRESPONDENCE_SEARCH_MAX_RESULTS) {
		const int i = mi->result_count++;
		mi->results[i].pose = *pose;
		mi->results[i].score = *score;
		mi->result_prior_cost[i] = prior_cost;
	} else if (search_result_better(&mi->results[mi->result_count - 1],
	                                mi->result_prior_cost[mi->result_count - 1], score, prior_cost)) {
		const int i = mi->result_count - 1;
		mi->results[i].pose = *pose;
		mi->results[i].score = *score;
		mi->result_prior_cost[i] = prior_cost;
	}

	for (int i = mi->result_count - 1; i > 0; i--) {
		if (!search_result_better(&mi->results[i - 1], mi->result_prior_cost[i - 1], &mi->results[i].score,
		                          mi->result_prior_cost[i])) {
			break;
		}
		struct correspondence_search_result tmp = mi->results[i - 1];
		double tmp_cost = mi->result_prior_cost[i - 1];
		mi->results[i - 1] = mi->results[i];
		mi->result_prior_cost[i - 1] = mi->result_prior_cost[i];
		mi->results[i] = tmp;
		mi->result_prior_cost[i] = tmp_cost;
	}
}

static bool
correspondence_search_project_pose(struct correspondence_search *cs,
                                   struct t_constellation_search_model *model,
                                   struct xrt_pose *pose,
                                   struct cs_model_info *mi,
                                   int depth)
{
	/* Given a pose, project each 3D LED point back to 2D and assign correspondences to blobs */
	/* If enough match, print out the pose */
	struct t_constellation_led_model *leds = model->led_model;

	if (pose->position.z < 0.05 || pose->position.z > 15) { /* Invalid position, out of range */
		LOG("Pose out of range - Z @ %f metres\n", pose->position.z);
		return false;
	}

	/* Soft anisotropic prior-orientation cost (mirror-flip re-rank). Split the candidate-vs-prior orientation
	 * difference about the camera-frame world-up into TILT (driftless gravity swing) + YAW (drifting twist),
	 * form the Huber-robustified anisotropic Mahalanobis penalty, and fold it into the candidate ranking
	 * below. A tilt flip's huge driftless tilt distance out-ranks it (decided even when the yaw prior is
	 * stale); a fresh-yaw flip is out-ranked by its yaw distance; an uncertain/untracked yaw widens sigma_yaw
	 * so reprojection decides. Both P3P twins are generated as candidates, so this SELECTS the correct twin
	 * instead of dropping the frame, and never rejects a candidate outright. Keyed on
	 * CS_FLAG_TRUST_PRIOR_ORIENT (cold re-acquire => zero penalty). */
	double prior_cost = 0.0;
	if (mi->search_flags & CS_FLAG_TRUST_PRIOR_ORIENT) {
		prior_cost = pose_metrics_prior_orient_cost(&pose->orientation, &mi->pose_prior.orientation,
		                                            &mi->up_vector, mi->sigma_tilt_rad, mi->sigma_yaw_rad,
		                                            mi->huber_knee_sigma, mi->cost_weight);

		/* Early prune from the cheap prior cost + orient error: a candidate that loses every ranking
		 * branch against the current GOOD best is skipped before the projection, the accept unchanged. */
		double cand_orient_err_len = prior_orient_err_len(pose, &mi->pose_prior);
		if (prune_by_prior_bound(mi, prior_cost, cand_orient_err_len, cs->num_points)) {
			cs->num_pose_checks_pruned++;
			return false;
		}
	}

	if (!cs_charge_work(cs, mi, CS_WORK_UNITS_PER_POSE_CHECK)) {
		return false;
	}

	/* Increment stats */
	cs->num_pose_checks++;

	struct pose_metrics score;

	/* Check how many LEDs have matching blobs in this pose,
	 * if there's enough we have a good match */
	/* FIXME: It would be better to be able to pass a list of undistorted
	 * blob points */
	if (mi->search_flags & CS_FLAG_HAVE_POSE_PRIOR) {
		pose_metrics_evaluate_pose_with_prior(&score, pose, false, &mi->pose_prior, mi->pos_error_thresh,
		                                      mi->rot_error_thresh, cs->blobs, cs->num_points, leds, cs->calib,
		                                      NULL, NULL);
	} else {
		pose_metrics_evaluate_pose(&score, pose, cs->blobs, cs->num_points, leds, cs->calib, NULL);
	}
	if (diagnostic_score_is_better_any(&mi->best_any_score, &score)) {
		mi->best_any_score = score;
		mi->best_any_pose = *pose;
		mi->best_any_pose_blob_depth = depth;
		mi->best_any_pose_led_depth = mi->led_depth;
	}

	/* If this pose is any good, test it further */
	if (POSE_HAS_FLAGS(&score, POSE_MATCH_GOOD)) {
		insert_search_result(mi, pose, &score, prior_cost);
		if (pose_metrics_score_is_better_pose_prior(&mi->best_score, mi->best_prior_cost, &score,
		                                            prior_cost)) {
			mi->best_score = score;
			mi->best_prior_cost = prior_cost;
			mi->best_pose = *pose;
			/* Cache the prune bounds from the new best (only consulted under CS_FLAG_TRUST_PRIOR_ORIENT,
			 * where matched_blobs >= 3 for any GOOD pose). */
			mi->best_combined_cost = score.reprojection_error + prior_cost;
			mi->best_cost_per_led =
			    score.matched_blobs > 0 ? mi->best_combined_cost / score.matched_blobs : 0.0;
			mi->best_orient_err_len = prior_orient_err_len(pose, &mi->pose_prior);
#if DUMP_TIMING
			mi->best_pose_found_time = os_monotonic_get_ns();
#endif
			mi->best_pose_blob_depth = depth;
			mi->best_pose_led_depth = mi->led_depth;
			mi->match_flags = mi->best_score.match_flags;

			if (mi->match_flags & POSE_MATCH_STRONG) {
				DEBUG_TIMING(
				    "# Found a strong match for model %d - %d points out of %d with error %f pixels^2 "
				    "after %u trials and %u pose checks\n",
				    mi->id, mi->best_score.matched_blobs, mi->best_score.visible_leds,
				    mi->best_score.reprojection_error, cs->num_trials, cs->num_pose_checks);
				DEBUG_TIMING(
				    "# Found at LED depth %d blob depth %d after %f ms. Anchor indices: LED %d blob "
				    "%d\n",
				    mi->best_pose_led_depth, mi->best_pose_blob_depth,
				    (float)(mi->best_pose_found_time - mi->search_start_time) / 1000000.0,
				    mi->led_index, mi->blob_index);
				DEBUG_TIMING("# pose orient %f %f %f %f pos %f %f %f\n", mi->best_pose.orientation.x,
				             mi->best_pose.orientation.y, mi->best_pose.orientation.z,
				             mi->best_pose.orientation.w, mi->best_pose.position.x,
				             mi->best_pose.position.y, mi->best_pose.position.z);
			}

#if DUMP_FULL_DEBUG
			float error_per_led =
			    score.visible_leds > 0 ? score.reprojection_error / score.visible_leds : -1;
			DEBUG(
			    "model %d new best pose candidate orient %f %f %f %f pos %f %f %f has %u visible LEDs, "
			    "error %f (%f / LED) after %u trials and %u pose checks\n",
			    mi->id, pose->orientation.x, pose->orientation.y, pose->orientation.z, pose->orientation.w,
			    pose->position.x, pose->position.y, pose->position.z, score.visible_leds,
			    score.reprojection_error, error_per_led, cs->num_trials, cs->num_pose_checks);

			DEBUG("model %d matched %u blobs of %u\n", mi->id, score.matched_blobs, score.visible_leds);
#endif
#if DUMP_SCENE
			dump_pose(cs, model, pose, mi);
#endif
			return true;
		} else {
#if DUMP_FULL_DEBUG
			float error_per_led =
			    score.visible_leds > 0 ? score.reprojection_error / score.visible_leds : -1;
			DEBUG(
			    "pose candidate orient %f %f %f %f pos %f %f %f has %u visible LEDs, error %f (%f / LED)\n",
			    pose->orientation.x, pose->orientation.y, pose->orientation.z, pose->orientation.w,
			    pose->position.x, pose->position.y, pose->position.z, score.visible_leds,
			    score.reprojection_error, error_per_led);
			DEBUG("matched %u blobs of %u\n", score.matched_blobs, score.visible_leds);
#endif
		}
		LOG("Found good pose match for device %u - %u LEDs matched %u visible ones\n", mi->id,
		    score.matched_blobs, score.visible_leds);
		return true;
	}

	LOG("Failed pose match - only %u blobs matched %u visible LEDs. %u unmatched blobs. Error %f\n",
	    score.matched_blobs, score.visible_leds, score.unmatched_blobs, score.reprojection_error);

	return false;
}

static void
quat_from_rotation_matrix(struct xrt_quat *me, double R[9])
{
	float trace = R[0] + R[3 + 1] + R[6 + 2];
	if (trace > 0) {
		float s = 0.5f / sqrtf(trace + 1.0f);
		me->w = -0.25f / s;
		me->x = -(R[6 + 1] - R[3 + 2]) * s;
		me->y = -(R[2] - R[6]) * s;
		me->z = -(R[3] - R[1]) * s;
	} else {
		if (R[0] > R[4] && R[0] > R[8]) {
			float s = 2.0f * sqrtf(1.0f + R[0] - R[3 + 1] - R[6 + 2]);
			me->w = -(R[6 + 1] - R[3 + 2]) / s;
			me->x = -0.25f * s;
			me->y = -(R[1] + R[3]) / s;
			me->z = -(R[2] + R[6]) / s;
		} else if (R[3 + 1] > R[6 + 2]) {
			float s = 2.0f * sqrtf(1.0f + R[3 + 1] - R[0] - R[6 + 2]);
			me->w = -(R[2] - R[6]) / s;
			me->x = -(R[1] + R[3]) / s;
			me->y = -0.25f * s;
			me->z = -(R[3 + 2] + R[6 + 1]) / s;
		} else {
			float s = 2.0f * sqrtf(1.0f + R[6 + 2] - R[0] - R[3 + 1]);
			me->w = -(R[3] - R[1]) / s;
			me->x = -(R[2] + R[6]) / s;
			me->y = -(R[3 + 2] + R[6 + 1]) / s;
			me->z = -0.25f * s;
		}
	}
}

static void
check_led_against_model_subset(struct correspondence_search *cs,
                               struct cs_model_info *mi,
                               struct cs_image_point **blobs,
                               struct t_constellation_led *model_leds[4],
                               int depth)
{
	struct t_constellation_search_model *model = mi->model;
	double x[3][3];
	struct xrt_vec3 *xcheck;
	int i;
	double *y1, *y2, *y3;
	double Rs[4][9], Ts[4][3];

	y1 = blobs[0]->point_homog;
	y2 = blobs[1]->point_homog;
	y3 = blobs[2]->point_homog;

	if (!cs_charge_work(cs, mi, 1)) {
		return;
	}
	cs->num_trials++;

	for (i = 0; i < 3; i++) {
		x[i][0] = model_leds[i]->pos.x;
		x[i][1] = model_leds[i]->pos.y;
		x[i][2] = model_leds[i]->pos.z;
	}
	xcheck = &model_leds[3]->pos;

	/* FIXME: It would be better if this spat out quaternions,
	 * then we wouldn't need to convert below */
	int valid = lambdatwist_p3p(y1, y2, y3, x[0], x[1], x[2], Rs, Ts);

	if (!valid) {
		return;
	}
	for (i = 0; i < valid; i++) {
		struct xrt_pose pose;
		struct xrt_vec3 checkpos, checkdir;
		float l;

		/* Construct quat and trans for this pose */
		quat_from_rotation_matrix(&pose.orientation, Rs[i]);

		pose.position.x = Ts[i][0];
		pose.position.y = Ts[i][1];
		pose.position.z = Ts[i][2];

		if (pose.position.z < 0.05 || pose.position.z > 15) {
			/* The object is unlikely to be < 5cm or > 15m from the camera */
			continue;
		}

		/* The quaternion produced should already be normalised, but sometimes it's not! */
#if 1
		math_quat_normalize(&pose.orientation);
#else
		l = math_quat_len(&pose.orient);
		assert(l > 0.9999 && l < 1.0001);
#endif

		bool checks_failed = false;
		struct xrt_vec3 tmp;

		for (int p = 0; p < 3; p++) {
			struct xrt_vec3 tmpblob;

			/* This pose must yield a projection of the anchor
			 * point, or something is really wrong */
			math_quat_rotate_vec3(&pose.orientation, &model_leds[p]->pos, &checkpos);
			math_vec3_accum(&pose.position, &checkpos);

			/* And should be camera facing in this pose */
			math_quat_rotate_vec3(&pose.orientation, &model_leds[p]->dir, &checkdir);
			math_vec3_normalize(&checkpos);

			double facing_dot = m_vec3_dot(checkpos, checkdir);

			/* Require only that the anchor LED not be actively facing away from
			 * the camera (that it's at worst perpendicular). Controller LEDs
			 * can be visible at that angle */
			if (facing_dot > 0.0) {
				// LED not facing the camera -> invalid pose
				checks_failed = true;
				break;
			}

			tmpblob.x = blobs[p]->point_homog[0];
			tmpblob.y = blobs[p]->point_homog[1];
			tmpblob.z = blobs[p]->point_homog[2];

			/* Calculate the image plane projection of the anchor
			 * LED position and check it's within 2.5mm of where it
			 * should be, to catch spurious failures in lambdatwist */
			math_vec3_scalar_mul(1.0 / checkpos.z, &checkpos);
			tmp = m_vec3_sub(checkpos, tmpblob);
			l = m_vec3_len(tmp);
			if (!(l <= 0.0025)) {
				LOG("Error pose candidate orient %f %f %f %f pos %f %f %f "
				    "LED %d @ %f %f %f projected to %f %f %f (err %f)\n",
				    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
				    pose.position.x, pose.position.y, pose.position.z, p, tmpblob.x, tmpblob.y,
				    tmpblob.z, checkpos.x, checkpos.y, checkpos.z, l);
				checks_failed = true;
				break; /* FIXME: Figure out why this happened */
			}
#if !CHECK_ALL_PROJECTIONS
			break;
#else
			struct xrt_vec2 reprojected;

			project_led_point(&model_leds[p]->pos, &pose, cs->calib, &reprojected);
			double xdiff = reprojected.x - blobs[p]->blob->x;
			double ydiff = reprojected.y - blobs[p]->blob->y;

			printf("Blob %d @ %f,%f should match LED %d projection %f %f (err %f)\n", p, blobs[p]->blob->x,
			       blobs[p]->blob->y, model_leds[p]->id, reprojected.x, reprojected.y,
			       sqrt(xdiff * xdiff + ydiff * ydiff));
#endif
		}

		if (checks_failed) {
			continue;
		}

		/* check against the 4th point to check the proposed P3P solution */
		math_quat_rotate_vec3(&pose.orientation, xcheck, &checkpos);
		math_vec3_accum(&pose.position, &checkpos);
		math_vec3_scalar_mul(1.0 / checkpos.z, &checkpos);

		/* Subtract projected point from undistorted 4th reference point to check error */
		struct xrt_vec3 checkblob;
		checkblob.x = blobs[3]->point_homog[0];
		checkblob.y = blobs[3]->point_homog[1];
		checkblob.z = blobs[3]->point_homog[2];

		tmp = m_vec3_sub(checkpos, checkblob);
		float distance = m_vec3_len(tmp);

		/* Check that the 4th point projected to within its blob */
		if (distance <= blobs[3]->max_dist) {
			correspondence_search_project_pose(cs, model, &pose, mi, depth);
		}
	}
}

/* Select k entries from the n provided in candidate_list into
 * output_list, then call check_led_match() with the result_list */
static void
select_k_blobs_from_n(struct correspondence_search *cs,
                      struct cs_model_info *mi,
                      struct t_constellation_led **model_leds,
                      struct cs_image_point **result_list,
                      struct cs_image_point **output_list,
                      struct cs_image_point **candidate_list,
                      int k,
                      int n,
                      int depth)
{
	if (k == 1) {
		output_list[0] = candidate_list[0];
		check_led_against_model_subset(cs, mi, result_list, model_leds, depth);
		return;
	}

	/*
	 * 2 branches here:
	 *   take the first entry, then select k-1 from n-1
	 *   don't take the first entry, select k from n-1 if n > k
	 */
	assert(k > 1);
	assert(n > 1);
	output_list[0] = candidate_list[0];
	select_k_blobs_from_n(cs, mi, model_leds, result_list, output_list + 1, candidate_list + 1, k - 1, n - 1,
	                      depth);

	/* Short circuit if we found a strong pose match already */
	if ((mi->match_flags & POSE_MATCH_STRONG) && (mi->search_flags & CS_FLAG_STOP_FOR_STRONG_MATCH))
		return;
	if (cs->budget_exhausted)
		return;

	if (n > k)
		select_k_blobs_from_n(cs, mi, model_leds, result_list, output_list, candidate_list + 1, k, n - 1,
		                      depth + 1);
}

/* Generate quadruples of neighbouring blobs and pass to check_led_match().
 * We want to
 *   a) preserve the first blob in the list each iteration
 *   b) Select combinations of 3 from the next 'max_search_depth' points.
 *
 * Permutation checking is done in the model LED inner loop, so this inner selection
 * only needs to try combos.
 *
 * We can work in a tmp array so as not to destroy the master list.
 *
 * The caller guarantees that the blob_list has at least max_search_depth+1
 * entries.
 */
static void
check_leds_against_anchor(struct correspondence_search *cs,
                          struct cs_model_info *mi,
                          struct t_constellation_led **model_leds,
                          struct cs_image_point *anchor)
{
	struct cs_image_point *work_list[MAX_BLOB_SEARCH_DEPTH + 1];
	int max_blob_search_depth = MIN(anchor->num_neighbours, mi->max_blob_depth);

	if (max_blob_search_depth < 3)
		return; // Not enough blobs to compare against

	work_list[0] = anchor;
	select_k_blobs_from_n(cs, mi, model_leds, work_list, work_list + 1, anchor->neighbours, 3,
	                      max_blob_search_depth, 1);
}

/* Called with 4 model_leds that need checking against blob combos */
static void
check_led_match(struct correspondence_search *cs,
                struct cs_model_info *mi,
                struct t_constellation_led **model_leds,
                int depth)
{
	int b;

	mi->led_depth = depth;

	for (b = 0; b < cs->num_points; b++) {
		if (cs->budget_exhausted) {
			return;
		}
		struct cs_image_point *anchor = cs->points + b;
		mi->blob_index = b;
		check_leds_against_anchor(cs, mi, model_leds, anchor);
	}
}

/* Select k constellation_led entries from the n provided in candidate_list into
 * output_list, then call check_led_match() with the result_list */
static void
select_k_leds_from_n(struct correspondence_search *cs,
                     struct cs_model_info *mi,
                     struct t_constellation_led **result_list,
                     struct t_constellation_led **output_list,
                     struct t_constellation_led **candidate_list,
                     int k,
                     int n,
                     int depth)
{
	if (k == 1) {
		struct t_constellation_led *swap_list[4];

		output_list[0] = candidate_list[0];
		check_led_match(cs, mi, result_list, depth);

		/* Short circuit if we found a strong pose match already */
		if ((mi->match_flags & POSE_MATCH_STRONG) && (mi->search_flags & CS_FLAG_STOP_FOR_STRONG_MATCH))
			return;
		if (cs->budget_exhausted)
			return;

		/* Check the other orientation of blob 2/3, without
		 * affecting result_list that needs to stay intact
		 * for other recursive calls */
		swap_list[0] = result_list[0];
		swap_list[1] = result_list[2];
		swap_list[2] = result_list[1];
		swap_list[3] = result_list[3];

		check_led_match(cs, mi, swap_list, depth);

		return;
	}

	/*
	 * 2 branches here:
	 *   take the first entry, then select k-1 from n-1
	 *   don't take the first entry, select k from n-1 if n > k
	 */
	assert(k > 1);
	assert(n > 1);
	output_list[0] = candidate_list[0];
	select_k_leds_from_n(cs, mi, result_list, output_list + 1, candidate_list + 1, k - 1, n - 1, depth);

	/* Short circuit if we found a strong pose match already */
	if ((mi->match_flags & POSE_MATCH_STRONG) && (mi->search_flags & CS_FLAG_STOP_FOR_STRONG_MATCH))
		return;
	if (cs->budget_exhausted)
		return;

	if (n > k)
		select_k_leds_from_n(cs, mi, result_list, output_list, candidate_list + 1, k, n - 1, depth + 1);
}

/* Generate quadruples of neighbouring leds and pass to check_led_match().
 * We want to
 *   a) preserve the first led in the list each iteration
 *   b) Select combinations of 3 from the 'max_search_depth' neighbouring LEDs.
 *   c) Try 2 permutations, [0,1,2,3] and [0,2,1,3]. In each case index 0 is
 *   the anchor and index 3 is a disambiguation check on the P3P result
 *
 * We can work in a tmp array so as not to destroy the master list.
 *
 * The caller guarantees that the LED neighbours list has at least max_search_depth
 * entries.
 */
static void
generate_led_match_candidates(struct correspondence_search *cs,
                              struct cs_model_info *mi,
                              struct t_constellation_search_led_candidate *c)
{
	struct t_constellation_led *work_list[MAX_LED_SEARCH_DEPTH + 1];

	int max_search_depth = MIN(mi->max_led_depth, c->num_neighbours) - mi->min_led_depth + 1;
	if (max_search_depth < 3)
		return; // Not enough LEDs to compare against

	work_list[0] = c->led;
	select_k_leds_from_n(cs, mi, work_list, work_list + 1, c->neighbours + mi->min_led_depth - 1, 3,
	                     max_search_depth, mi->min_led_depth);
}

static bool
search_pose_for_model(struct correspondence_search *cs, struct cs_model_info *mi)
{
	struct t_constellation_search_model *model = mi->model;
	int b, l;

	/* clear the info for this model (poses/depths too: on a no-GOOD-pose search these are
	 * copied into the failure diagnostics and were previously read uninitialized) */
	memset(&mi->best_score, 0, sizeof(struct pose_metrics));
	memset(&mi->best_any_score, 0, sizeof(struct pose_metrics));
	memset(&mi->best_pose, 0, sizeof(mi->best_pose));
	memset(&mi->best_any_pose, 0, sizeof(mi->best_any_pose));
	mi->best_pose_blob_depth = -1;
	mi->best_pose_led_depth = -1;
	mi->best_any_pose_blob_depth = -1;
	mi->best_any_pose_led_depth = -1;
	mi->best_prior_cost = 0.0;
	mi->best_combined_cost = 0.0;
	mi->best_cost_per_led = 0.0;
	mi->best_orient_err_len = 0.0;
	mi->match_flags = 0;
	mi->result_count = 0;

	/* Clear stats */
	cs->num_trials = cs->num_pose_checks = cs->num_pose_checks_pruned = 0;
	cs->work_spent = 0;
	cs->budget_exhausted = false;
	memset(&cs->last_diag, 0, sizeof(cs->last_diag));
	cs->last_diag.input_blobs = (uint32_t)cs->num_points;
	cs->last_diag.best_pose_blob_depth = -1;
	cs->last_diag.best_pose_led_depth = -1;
	cs->last_diag.best_any_pose_blob_depth = -1;
	cs->last_diag.best_any_pose_led_depth = -1;

	/* Configure search params from the flags */
	if (mi->search_flags & CS_FLAG_SHALLOW_SEARCH) {
		mi->max_blob_depth = 4; /* Only test nearest neighbours of blobs +1 */
		mi->max_led_depth = 4;  /* Test LEDs plus 1 neighbour removed */
		mi->min_led_depth = 1;  /* Start with the nearest LED neighbour */
	} else {
		mi->max_blob_depth = MAX_BLOB_SEARCH_DEPTH;
		mi->max_led_depth = MAX_LED_SEARCH_DEPTH;
		mi->min_led_depth = 3; /* Start with next nearest LED neighbour that shallow search stopped at */
	}

	if (mi->search_flags & CS_FLAG_DEEP_SEARCH) {
		mi->max_blob_depth = MAX_BLOB_SEARCH_DEPTH;
		mi->max_led_depth = MAX_LED_SEARCH_DEPTH;
	}
	cs->last_diag.min_led_depth = mi->min_led_depth;
	cs->last_diag.max_led_depth = mi->max_led_depth;
	cs->last_diag.max_blob_depth = mi->max_blob_depth;

#if DUMP_TIMING
	/* Set the search start time */
	mi->search_start_time = os_monotonic_get_ns();
#endif

	/* filter the list of blobs to unknown, or belonging to this model */
	for (b = 0; b < cs->num_points; b++) {
		struct cs_image_point **all_neighbours = cs->blob_neighbours[b];
		struct cs_image_point *anchor = cs->points + b;
		int out_index = 0, in_index;
		uint16_t led_id = anchor->blob->led_id;

		if ((mi->search_flags & CS_FLAG_MATCH_ALL_BLOBS) || led_id == LED_INVALID_ID ||
		    LED_OBJECT_ID(led_id) == mi->id) {
			cs->last_diag.searchable_anchors++;
			for (in_index = 0; in_index < cs->num_points && out_index < MAX_BLOB_SEARCH_DEPTH; in_index++) {
				struct cs_image_point *p = all_neighbours[in_index];

				/* Don't include the blob in its own neighbours */
				if (anchor->blob == p->blob)
					continue;

				led_id = p->blob->led_id;
				if ((mi->search_flags & CS_FLAG_MATCH_ALL_BLOBS) || led_id == LED_INVALID_ID ||
				    LED_OBJECT_ID(led_id) == mi->id)
					anchor->neighbours[out_index++] = p;
			}
		} else {
			cs->last_diag.filtered_anchors++;
		}
		anchor->num_neighbours = out_index;
		cs->last_diag.neighbour_links += (uint32_t)out_index;
		if (out_index >= 3) {
			cs->last_diag.anchors_with_3_neighbours++;
		}

#if DUMP_FULL_LOG
		printf("Model %d, blob %d @ %f,%f neighbours %d Search list:\n", mi->id, b, anchor->blob->x,
		       anchor->blob->y, anchor->num_neighbours);
		for (int i = 0; i < anchor->num_neighbours; i++) {
			struct cs_image_point *p1 = anchor->neighbours[i];
			double dist = (p1->blob->y - anchor->blob->y) * (p1->blob->y - anchor->blob->y) +
			              (p1->blob->x - anchor->blob->x) * (p1->blob->x - anchor->blob->x);
			printf("  LED ID %u (%f,%f) @ %f,%f. Dist %f\n", p1->blob->led_id, p1->point_homog[0],
			       p1->point_homog[1], p1->blob->x, p1->blob->y, sqrt(dist));
		}
#endif
	}

	/* Start correspondence search for this model */
	/* At this stage, each image point has a list of the nearest neighbours filtered for this model */
	for (l = 0; l < model->num_points; l++) {
		struct t_constellation_search_led_candidate *c = model->points[l];
		mi->led_index = l;

		generate_led_match_candidates(cs, mi, c);

		if ((mi->match_flags & POSE_MATCH_STRONG) && (mi->search_flags & CS_FLAG_STOP_FOR_STRONG_MATCH))
			return true;
	}

	if (mi->match_flags & POSE_MATCH_GOOD)
		return true;

	return false;
}

static void
finish_search_diagnostics(struct correspondence_search *cs, const struct cs_model_info *mi)
{
	cs->last_diag.num_trials = cs->num_trials;
	cs->last_diag.num_pose_checks = cs->num_pose_checks;
	cs->last_diag.num_pose_checks_pruned = cs->num_pose_checks_pruned;
	cs->last_diag.best_pose_blob_depth = mi->best_pose_blob_depth;
	cs->last_diag.best_pose_led_depth = mi->best_pose_led_depth;
	cs->last_diag.best_any_pose_blob_depth = mi->best_any_pose_blob_depth;
	cs->last_diag.best_any_pose_led_depth = mi->best_any_pose_led_depth;
	cs->last_diag.best_any_score = mi->best_any_score;
	cs->last_diag.best_any_pose = mi->best_any_pose;
	cs->last_diag.best_any_match_flags = mi->best_any_score.match_flags;
	cs->last_diag.best_any_leds_visible = mi->best_any_score.visible_leds;
	cs->last_diag.best_any_blobs_matched = mi->best_any_score.matched_blobs;
	cs->last_diag.best_any_unmatched_blobs = mi->best_any_score.unmatched_blobs;
	cs->last_diag.best_any_reproj_err_px = mi->best_any_score.reprojection_error;
	cs->last_diag.work_spent = cs->work_spent;
	cs->last_diag.budget_exhausted = cs->budget_exhausted ? 1 : 0;
}

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
                                           int max_results)
{
	assert(pose != NULL);
	assert(results != NULL || max_results == 0);

	if (max_results <= 0) {
		return 0;
	}
	if (max_results > CORRESPONDENCE_SEARCH_MAX_RESULTS) {
		max_results = CORRESPONDENCE_SEARCH_MAX_RESULTS;
	}
	if ((search_flags & (CS_FLAG_SHALLOW_SEARCH | CS_FLAG_DEEP_SEARCH)) == 0) {
		search_flags |= CS_FLAG_SHALLOW_SEARCH | CS_FLAG_DEEP_SEARCH;
	}

	struct cs_model_info mi;
	mi.id = model->id;
	mi.model = model;
	mi.search_flags = search_flags;
	mi.match_flags = 0;
	mi.work_allowance = work_allowance;

	if (search_flags & CS_FLAG_HAVE_POSE_PRIOR) {
		assert(pos_error_thresh != NULL);
		assert(rot_error_thresh != NULL);
		mi.pose_prior = *pose;
		mi.pos_error_thresh = pos_error_thresh;
		mi.rot_error_thresh = rot_error_thresh;
	}

	if (search_flags & CS_FLAG_TRUST_PRIOR_ORIENT) {
		assert((search_flags & CS_FLAG_HAVE_POSE_PRIOR) != 0);
		assert(up_vector != NULL);
		mi.up_vector = *up_vector;
		mi.sigma_tilt_rad = sigma_tilt_rad;
		mi.sigma_yaw_rad = sigma_yaw_rad;
		mi.huber_knee_sigma = huber_knee_sigma;
		mi.cost_weight = cost_weight;
	}

	search_pose_for_model(cs, &mi);
	append_best_partial_result(&mi);
	finish_search_diagnostics(cs, &mi);
	const int n = mi.result_count < max_results ? mi.result_count : max_results;
	for (int i = 0; i < n; i++) {
		results[i] = mi.results[i];
	}
	if (n > 0) {
		*pose = results[0].pose;
	}
	return n;
}

void
correspondence_search_get_last_diagnostics(struct correspondence_search *cs,
                                           struct correspondence_search_diagnostics *out_diag)
{
	assert(cs != NULL);
	assert(out_diag != NULL);
	*out_diag = cs->last_diag;
}
