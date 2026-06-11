// Copyright 2020 Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Constellation tracking metrics for assessing pose matches
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup constellation
 */
#include <assert.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#include "math/m_api.h"
#include "math/m_vec3.h"
#include "util/u_logging.h"

#include "pose_metrics.h"

#include <math.h>

void
pose_metrics_prior_orient_split(const struct xrt_quat *q_cand,
                                const struct xrt_quat *q_prior,
                                const struct xrt_vec3 *up,
                                double *out_tilt_rad,
                                double *out_yaw_rad)
{
	/* NaN guard: a non-finite-component input quat would propagate NaN through the cw>1?1:cw clamps below
	 * (NaN>1 is false, so the clamp doesn't catch it). Unreachable from the live call sites (always
	 * normalized finite PnP/filter quats); pure hardening so a garbage input yields tilt=yaw=0, not NaN. */
	if (!isfinite(q_cand->x) || !isfinite(q_cand->y) || !isfinite(q_cand->z) || !isfinite(q_cand->w) ||
	    !isfinite(q_prior->x) || !isfinite(q_prior->y) || !isfinite(q_prior->z) || !isfinite(q_prior->w)) {
		if (out_tilt_rad)
			*out_tilt_rad = 0.0;
		if (out_yaw_rad)
			*out_yaw_rad = 0.0;
		return;
	}

	/* Camera-frame relative rotation from prior to candidate: q_rel = q_cand . q_prior^-1 (left-multiply).
	 * It MUST be the camera-frame (left) relative, not the object-frame q_prior^-1 . q_cand, because the
	 * swing/twist split below is about a CAMERA-frame axis (up): the rotation and the axis must live in the
	 * same frame. The object-frame form leaks ~sin(tilt)*yaw of a pure world-yaw on a tilted prior into the
	 * tilt term (e.g. 33 deg tilt + 90 deg yaw -> 45.3 deg false tilt), over-rejecting tilted+yaw-drifted
	 * frames; the camera-frame form correctly attributes a pure world-yaw entirely to yaw (tilt = 0). */
	struct xrt_quat q_prior_inv;
	math_quat_invert(q_prior, &q_prior_inv);
	struct xrt_quat q_rel;
	math_quat_rotate(q_cand, &q_prior_inv, &q_rel); /* q_rel = q_cand . q_prior^-1 */
	math_quat_normalize(&q_rel);

	/* A near-zero up axis can't define a twist -> all difference is "yaw" (defer to the yaw bound). Report
	 * the FULL geodesic angle (2*acos|w|), consistent with the swing/twist angles below. */
	struct xrt_vec3 axis = *up;
	const double up_len = sqrt((double)axis.x * axis.x + (double)axis.y * axis.y + (double)axis.z * axis.z);
	if (!(up_len > 1e-6)) {
		if (out_tilt_rad)
			*out_tilt_rad = 0.0;
		if (out_yaw_rad) {
			double cw = fabs((double)q_rel.w);
			*out_yaw_rad = 2.0 * acos(cw > 1.0 ? 1.0 : cw);
		}
		return;
	}
	axis.x /= (float)up_len;
	axis.y /= (float)up_len;
	axis.z /= (float)up_len;

	/* Split into swing (off-axis tilt) and twist (about-axis yaw). swing.twist == q_rel. */
	struct xrt_quat swing, twist;
	math_quat_decompose_swing_twist(&q_rel, &axis, &swing, &twist);
	if (out_tilt_rad) {
		double cw = fabs((double)swing.w);
		*out_tilt_rad = 2.0 * acos(cw > 1.0 ? 1.0 : cw); /* swing angle (tilt) */
	}
	if (out_yaw_rad) {
		double cw = fabs((double)twist.w);
		*out_yaw_rad = 2.0 * acos(cw > 1.0 ? 1.0 : cw); /* twist angle (yaw) */
	}
}

double
pose_metrics_prior_orient_cost(const struct xrt_quat *q_cand,
                               const struct xrt_quat *q_prior,
                               const struct xrt_vec3 *up,
                               double sigma_tilt_rad,
                               double sigma_yaw_rad,
                               double huber_knee_sigma,
                               double weight)
{
	double tilt_rad = 0.0, yaw_rad = 0.0;
	pose_metrics_prior_orient_split(q_cand, q_prior, up, &tilt_rad, &yaw_rad);

	/* Anisotropic squared Mahalanobis distance. A non-positive sigma drops that axis (treated as +inf). */
	double d2 = 0.0;
	if (sigma_tilt_rad > 0.0) {
		const double s = tilt_rad / sigma_tilt_rad;
		d2 += s * s;
	}
	if (sigma_yaw_rad > 0.0) {
		const double s = yaw_rad / sigma_yaw_rad;
		d2 += s * s;
	}

	/* Huber loss on the standardized residual s = sqrt(d2): quadratic within the knee, linear beyond, so a
	 * gross outlier's penalty grows only linearly (bend, don't cut) and cannot dominate the cost. */
	const double s = sqrt(d2);
	const double k = huber_knee_sigma > 0.0 ? huber_knee_sigma : 0.0;
	const double rho = (k <= 0.0 || s <= k) ? d2 : (k * (2.0 * s - k));

	return weight * rho;
}

static void
expand_rect(struct pose_rect *bounds, double x, double y, double w, double h)
{
	if (x < bounds->left)
		bounds->left = x;
	if (y < bounds->top)
		bounds->top = y;
	if (x + w > bounds->right)
		bounds->right = x + w;
	if (y + h > bounds->bottom)
		bounds->bottom = y + h;
}

/* True if @p blob falls inside this LED's covariance-scaled gate ellipse (and isn't grossly oversized).
 * Returns the raw reprojection cost (px^2) in @p out_sqerror — accumulated into reprojection_error and
 * compared against the px^2-calibrated GOOD/STRONG thresholds — AND the per-blob measurement-noise-weighted
 * (Mahalanobis) cost (dx^2+dy^2)/pos_var_px2 in @p out_rank_cost, which RANKS the global one-to-one
 * assignment: a fat / clipped / edge / faint blob (large pos_var_px2, an uncertain LED centre) yields a
 * larger weighted cost, so when blobs compete for an LED the tighter, more certain centre is preferred. The
 * accept/reject error stays in raw px^2; only the assignment ordering is uncertainty-weighted. The
 * gate half-axes gate_ax/ay come from get_visible_leds_and_bounds; they equal led_radius_px. */
static bool
led_blob_match_cost(const struct pose_metrics_visible_led_info *led_info,
                    const struct blob *blob,
                    double *out_sqerror,
                    double *out_rank_cost)
{
	if (blob->width > led_info->led_radius_px * 4 || blob->height > led_info->led_radius_px * 4)
		return false; /* blob far larger than the LED -> not this LED */
	const double dx = led_info->pos_px.x - blob->x;
	const double dy = led_info->pos_px.y - blob->y;
	const double ax = led_info->gate_ax_px, ay = led_info->gate_ay_px;
	if ((dx * dx) / (ax * ax) + (dy * dy) / (ay * ay) > 1.0)
		return false; /* outside the gate ellipse */
	const double sqerror = dx * dx + dy * dy;
	*out_sqerror = sqerror;
	/* pos_var_px2 is floored >= 0.25 px^2 at the source (blobwatch), so this division is well-posed. */
	*out_rank_cost = sqerror / (double)blob->pos_var_px2;
	return true;
}

/* Upper bound on admissible (blob,LED) pairings collected per pose check = blobs*visible_leds, so it can
 * never truncate a valid pairing (derived from the array maxima, not a magic cap). ~50 KB on the stack. */
#define MAX_GATE_CANDIDATES (MAX_BLOBS_PER_FRAME * MAX_OBJECT_LEDS)

/* One admissible blob<->LED pairing for the global assignment. Ranked by rank_cost (the per-blob
 * measurement-noise-weighted reprojection cost); sqerror is the raw px^2 error folded into reprojection_error
 * once the pairing is committed. */
struct gate_candidate
{
	float rank_cost;
	float sqerror;
	uint16_t blob_idx;
	uint16_t led_idx;
};

static int
gate_candidate_cmp(const void *a, const void *b)
{
	const struct gate_candidate *ga = a, *gb = b;
	if (ga->rank_cost < gb->rank_cost)
		return -1;
	if (ga->rank_cost > gb->rank_cost)
		return 1;
	/* Deterministic tiebreak (qsort isn't stable) so an exact-cost tie resolves identically every run
	 * -> reproducible offline replay. */
	if (ga->blob_idx != gb->blob_idx)
		return (ga->blob_idx < gb->blob_idx) ? -1 : 1;
	return (ga->led_idx < gb->led_idx) ? -1 : (ga->led_idx > gb->led_idx) ? 1 : 0;
}

static void
sort_gate_candidates(struct gate_candidate *cands, int ncand)
{
	if (ncand <= 1) {
		return;
	}
	if (ncand <= 32) {
		for (int i = 1; i < ncand; i++) {
			struct gate_candidate cur = cands[i];
			int j = i;
			while (j > 0 && gate_candidate_cmp(cands + j - 1, &cur) > 0) {
				cands[j] = cands[j - 1];
				j--;
			}
			cands[j] = cur;
		}
		return;
	}
	qsort(cands, ncand, sizeof(cands[0]), gate_candidate_cmp);
}

/* Bound LED detection odds: even a straight-on LED can be missed, and a grazing LED can still be detected. */
#define DATA_NLL_P_MIN 0.05
#define ASSOC_CLUTTER_MATCH_CAP_NLL 14.3

static double
led_visibility_weight(double facing_dot)
{
	const double edge = cos(DEG_TO_RAD(180.0 - LED_ANGLE)); /* the visibility cutoff, ~cos(98 deg) < 0 */
	double w = (edge - facing_dot) / (edge + 1.0);
	if (w < 0.0)
		w = 0.0;
	if (w > 1.0)
		w = 1.0;
	return w;
}

double
pose_metrics_pkf_detection_prob(double facing_dot)
{
	double p = DATA_NLL_P_MIN + (1.0 - DATA_NLL_P_MIN) * led_visibility_weight(facing_dot);
	if (p > 1.0 - DATA_NLL_P_MIN) {
		p = 1.0 - DATA_NLL_P_MIN;
	}
	if (p < DATA_NLL_P_MIN) {
		p = DATA_NLL_P_MIN;
	}
	return p;
}

double
pose_metrics_pkf_pair_nll(double sqerror_px2, double p_i)
{
	if (p_i < DATA_NLL_P_MIN) {
		p_i = DATA_NLL_P_MIN;
	}
	if (p_i > 1.0 - DATA_NLL_P_MIN) {
		p_i = 1.0 - DATA_NLL_P_MIN;
	}
	return sqerror_px2 + (-log(p_i) + log(1.0 - p_i));
}

double
pose_metrics_pkf_clutter_likelihood(void)
{
	return exp(-ASSOC_CLUTTER_MATCH_CAP_NLL);
}

double
pose_metrics_pkf_permanent(const double *Q, int m, int n)
{
	assert(m >= 0 && m <= PKF_MAX_CLUSTER);
	assert(n >= 0 && n <= PKF_MAX_CLUSTER);
	assert(m <= n);
	if (m == 0) {
		return 1.0;
	}

	double dp[1 << PKF_MAX_CLUSTER] = {0};
	double next[1 << PKF_MAX_CLUSTER] = {0};
	const int nmask = 1 << n;
	dp[0] = 1.0;

	for (int i = 0; i < m; i++) {
		for (int mask = 0; mask < nmask; mask++) {
			next[mask] = 0.0;
		}
		for (int mask = 0; mask < nmask; mask++) {
			if (dp[mask] == 0.0) {
				continue;
			}
			for (int j = 0; j < n; j++) {
				if ((mask & (1 << j)) != 0) {
					continue;
				}
				const double q = Q[i * n + j];
				if (q != 0.0) {
					next[mask | (1 << j)] += dp[mask] * q;
				}
			}
		}
		for (int mask = 0; mask < nmask; mask++) {
			dp[mask] = next[mask];
		}
	}

	double per = 0.0;
	for (int mask = 0; mask < nmask; mask++) {
		per += dp[mask];
	}
	return per;
}

double
pose_metrics_pkf_permanent_augmented(const double *L, const double *L_clutter, int m, int n)
{
	assert(m >= 1 && m <= PKF_MAX_CLUSTER);
	assert(n >= 0 && n <= PKF_MAX_CLUSTER);

	double dp[1 << PKF_MAX_CLUSTER] = {0};
	double next[1 << PKF_MAX_CLUSTER] = {0};
	const int nmask = 1 << n;
	dp[0] = 1.0;

	for (int i = 0; i < m; i++) {
		for (int mask = 0; mask < nmask; mask++) {
			next[mask] = 0.0;
		}
		for (int mask = 0; mask < nmask; mask++) {
			if (dp[mask] == 0.0) {
				continue;
			}
			next[mask] += dp[mask] * L_clutter[i];
			for (int j = 0; j < n; j++) {
				if ((mask & (1 << j)) != 0) {
					continue;
				}
				const double l = L[i * n + j];
				if (l != 0.0) {
					next[mask | (1 << j)] += dp[mask] * l;
				}
			}
		}
		for (int mask = 0; mask < nmask; mask++) {
			dp[mask] = next[mask];
		}
	}

	double per = 0.0;
	for (int mask = 0; mask < nmask; mask++) {
		per += dp[mask];
	}
	return per;
}

/* Detection NLL for a finished assignment, plus the all-missed reference for matched LEDs. */
static void
compute_data_nll(struct pose_metrics_blob_match_info *match_info)
{
	double nll_detection = 0.0;
	double nll_missed_if_matched = 0.0;
	for (int i = 0; i < match_info->num_visible_leds; i++) {
		const struct pose_metrics_visible_led_info *led_info = match_info->visible_leds + i;
		const double w = led_visibility_weight(led_info->facing_dot);
		double p = DATA_NLL_P_MIN + (1.0 - DATA_NLL_P_MIN) * w;
		if (p > 1.0 - DATA_NLL_P_MIN)
			p = 1.0 - DATA_NLL_P_MIN; /* ceiling: keep -log(1-p) finite for a straight-on LED */
		if (led_info->matched_blob != NULL) {
			nll_detection += -log(p);
			nll_missed_if_matched += -log(1.0 - p);
		} else {
			nll_detection += -log(1.0 - p); /* visible LED left unmatched: miss cost */
		}
	}
	match_info->data_nll_detection = nll_detection;
	match_info->data_nll_missed_if_matched = nll_missed_if_matched;
}

static void
check_pose_prior(struct pose_metrics *score,
                 struct xrt_pose *pose,
                 struct xrt_pose *pose_prior,
                 const struct xrt_vec3 *pos_error_thresh,
                 const struct xrt_vec3 *rot_error_thresh)
{
	struct xrt_quat orient_diff;

	score->match_flags |= POSE_HAD_PRIOR;

	score->pos_error = m_vec3_sub(pose->position, pose_prior->position);

	math_quat_unrotate(&pose->orientation, &pose_prior->orientation, &orient_diff);
	math_quat_normalize(&orient_diff);
	math_quat_ln(&orient_diff, &score->orient_error);

	/* Check each component of position and rotation are within the passed error bound and
	 * clear any return flag that's not set */
	if (pos_error_thresh) {
		score->match_flags |= POSE_MATCH_POSITION;

		if (fabs(score->pos_error.x) > pos_error_thresh->x || fabs(score->pos_error.y) > pos_error_thresh->y ||
		    fabs(score->pos_error.z) > pos_error_thresh->z) {
			score->match_flags &= ~POSE_MATCH_POSITION;
		}
	}

	if (rot_error_thresh) {
		score->match_flags |= POSE_MATCH_ORIENT;

		if (fabs(score->orient_error.x) > rot_error_thresh->x ||
		    fabs(score->orient_error.y) > rot_error_thresh->y ||
		    fabs(score->orient_error.z) > rot_error_thresh->z) {
			score->match_flags &= ~POSE_MATCH_ORIENT;
		}
	}
}

static bool
project_led_points(struct t_constellation_led_model *led_model,
                   struct camera_model *calib,
                   struct xrt_pose *pose,
                   struct xrt_vec3 *out_positions,
                   struct xrt_vec2 *out_points)
{
	for (int i = 0; i < led_model->num_leds; i++) {
		struct xrt_vec3 *tmp = out_positions + i;
		math_pose_transform_point(pose, &led_model->leds[i].pos, tmp);
		if (!t_camera_models_project(&calib->calib, tmp->x, tmp->y, tmp->z, &out_points[i].x, &out_points[i].y))
			return false;
	}
	return true;
}

static void
get_visible_leds_and_bounds(struct xrt_pose *pose,
                            struct t_constellation_led_model *led_model,
                            struct camera_model *calib,
                            struct pose_metrics_visible_led_info *visible_led_points,
                            int *num_visible_leds,
                            struct pose_rect *bounds)
{
	struct xrt_vec3 led_out_positions[MAX_OBJECT_LEDS];
	struct xrt_vec2 led_out_points[MAX_OBJECT_LEDS];
	bool first_visible_led = true;
	int i;
	struct t_constellation_led *leds = led_model->leds;
	const int num_leds = led_model->num_leds;

	/* Project LEDs into the distorted image space */
	if (!project_led_points(led_model, calib, pose, led_out_positions, led_out_points)) {
		*num_visible_leds = 0;
		return;
	}

	/* Compute LED pixel size based on model distance below
	 * using the larger X/Y focal length and LED's Z value */
	double focal_length = MAX(calib->calib.fx, calib->calib.fy);

	// pose is camera->device, I_pose is device->camera
	struct xrt_pose P_obj_cam;
	math_pose_invert(pose, &P_obj_cam);

	/* Calculate the bounding box and visible LEDs */
	*num_visible_leds = 0;
	for (i = 0; i < num_leds; i++) {
		struct xrt_vec2 *led_pos_px = led_out_points + i;
		struct xrt_vec3 *led_pos_m = led_out_positions + i;
		struct t_constellation_led *led = &leds[i];

		/* LEDs behind the camera are not visible */
		if (led_pos_m->z <= 0.0) {
			continue;
		}

		if (led_pos_px->x < 0 || led_pos_px->y < 0 || led_pos_px->x >= calib->width ||
		    led_pos_px->y >= calib->height)
			continue; // Outside the visible screen space

		/* Calculate the expected size of an LED at this distance */
		double led_radius_px = 4.0;
		const double led_radius_mm = led->radius_mm;
		led_radius_px = focal_length * led_radius_mm / led_pos_m->z / 1000.0;

		/* Convert the position to a unit vector for dot product comparison */
		struct xrt_vec3 view_vec = *led_pos_m;
		struct xrt_vec3 normal;

		math_vec3_normalize(&view_vec);
		math_quat_rotate_vec3(&pose->orientation, &leds[i].dir, &normal);

		double facing_dot = m_vec3_dot(view_vec, normal);

		/* The vector to the LED position points out from the camera
		 * to the LED, but the normal points toward the camera, so
		 * we need to compare against 180 - LED_ANGLE here */
		if (facing_dot > cos(DEG_TO_RAD(180.0 - LED_ANGLE))) {
			continue;
		}

		if (led_model->check_led_visibility &&
		    !led_model->check_led_visibility(led_model, i, P_obj_cam.position)) {
			continue;
		}

		/* Blob<->LED gate half-axes: the isotropic fixed LED-radius circle. The mutual-exclusion win
		 * comes from the GLOBAL one-to-one assignment, not a wider gate, so the gate stays at
		 * led_radius_px; widening it from a prior covariance stays a body-only change. */
		const double gate_ax_px = led_radius_px;
		const double gate_ay_px = led_radius_px;

		struct pose_metrics_visible_led_info *led_info = visible_led_points + (*num_visible_leds);
		led_info->led = leds + i;
		led_info->pos_px = *led_pos_px;
		led_info->pos_m = *led_pos_m;
		led_info->led_radius_px = led_radius_px;
		led_info->gate_ax_px = gate_ax_px;
		led_info->gate_ay_px = gate_ay_px;
		led_info->matched_blob = NULL;
		led_info->facing_dot = facing_dot;
		(*num_visible_leds)++;

		/* Expand the bounding box by the gate so blobs in the gate aren't
		 * dropped by the bbox pre-filter. Margins are asymmetric (-axis .. +2*axis). */
		if (first_visible_led) {
			bounds->left = led_pos_px->x - gate_ax_px;
			bounds->top = led_pos_px->y - gate_ay_px;
			bounds->right = led_pos_px->x + 2 * gate_ax_px;
			bounds->bottom = led_pos_px->y + 2 * gate_ay_px;
			first_visible_led = false;
		} else {
			expand_rect(bounds, led_pos_px->x - gate_ax_px, led_pos_px->y - gate_ay_px, 2 * gate_ax_px,
			            2 * gate_ay_px);
		}
	}
}

static bool
project_bounding_points(struct t_constellation_led_model *led_model,
                        struct camera_model *calib,
                        struct xrt_pose *P_cam_obj,
                        struct xrt_vec3 *out_positions,
                        struct xrt_vec2 *out_points)
{
	for (int i = 0; i < led_model->num_bounding_points; i++) {
		struct xrt_vec3 *tmp = out_positions + i;
		math_pose_transform_point(P_cam_obj, &led_model->bounding_points[i].pos, tmp);
		if (!t_camera_models_project(&calib->calib, tmp->x, tmp->y, tmp->z, &out_points[i].x,
		                             &out_points[i].y)) {
			return false;
		}
	}
	return true;
}

void
pose_metrics_get_device_bounds(struct xrt_pose *P_cam_obj,
                               struct t_constellation_led_model *led_model,
                               struct camera_model *calib,
                               struct pose_rect *device_bounds,
                               struct xrt_vec2 *visible_points,
                               size_t *num_visible_points)
{
	bool first_visible_bounding_point = true;

	if (num_visible_points)
		*num_visible_points = 0;

	struct xrt_vec3 bounding_out_positions[MAX_OBJECT_LEDS];
	struct xrt_vec2 bounding_out_points[MAX_OBJECT_LEDS];

	const int num_bounding_points = led_model->num_bounding_points;
	/* Project device bounding points into the distorted image space */
	bool bounding_points_valid =
	    project_bounding_points(led_model, calib, P_cam_obj, bounding_out_positions, bounding_out_points);

	*device_bounds = (struct pose_rect){0, 0, 0, 0};

	for (int i = 0; i < num_bounding_points && bounding_points_valid; i++) {
		struct xrt_vec2 *point_pos_px = bounding_out_points + i;
		struct xrt_vec3 *point_pos_m = bounding_out_positions + i;

		// skip points behind the camera
		if (point_pos_m->z <= 0) {
			continue;
		}

		if (visible_points && num_visible_points)
			visible_points[(*num_visible_points)++] = *point_pos_px;

		struct xrt_vec2 clamped_pos = {CLAMP(point_pos_px->x, 0, calib->width - 1),
		                               CLAMP(point_pos_px->y, 0, calib->height - 1)};

		if (first_visible_bounding_point) {
			*device_bounds = (struct pose_rect){
			    .top = clamped_pos.y,
			    .bottom = clamped_pos.y,
			    .left = clamped_pos.x,
			    .right = clamped_pos.x,
			};
		} else
			expand_rect(device_bounds, clamped_pos.x, clamped_pos.y, 0, 0);

		first_visible_bounding_point = false;
	}
}

void
pose_metrics_match_pose_to_blobs(struct xrt_pose *pose,
                                 struct blob *blobs,
                                 int num_blobs,
                                 struct t_constellation_led_model *led_model,
                                 struct camera_model *calib,
                                 struct pose_metrics_blob_match_info *match_info)
{
	struct pose_rect *bounds = &match_info->bounds;

	match_info->reprojection_error = 0.0;
	match_info->matched_blobs = 0;
	match_info->unmatched_blobs = 0;
	match_info->data_nll_detection = 0.0;
	match_info->data_nll_missed_if_matched = 0.0;

	get_visible_leds_and_bounds(pose, led_model, calib, match_info->visible_leds, &match_info->num_visible_leds,
	                            &match_info->bounds);

	/* Global ONE-TO-ONE assignment (Global Nearest Neighbour): collect every admissible (blob,LED) pair
	 * within its gate ellipse, rank by reprojection cost, then assign greedily with mutual exclusivity —
	 * each blob and each LED used at most once. Unlike per-blob nearest matching, this cannot mislabel
	 * when the gate is wider than the inter-LED spacing (the case a covariance-adaptive gate creates): it
	 * yields the lowest-total-reprojection consistent matching, so a wide prior widens the SEARCH without
	 * corrupting the ASSIGNMENT. */
	bool all_led_ids_matched = true;
	const int nleds = match_info->num_visible_leds;

	struct gate_candidate cands[MAX_GATE_CANDIDATES];
	int ncand = 0;
	int considered = 0; /* in-bounds blobs belonging to this device -> the matchable set */

	for (int i = 0; i < num_blobs; i++) {
		struct blob *b = blobs + i;
		uint32_t led_object_id = LED_OBJECT_ID(b->led_id);

		/* Skip blobs already labelled for another device, or outside the pose bounding box. */
		if (led_object_id != LED_INVALID_ID && led_object_id != led_model->id)
			continue;
		if (b->x < bounds->left || b->y < bounds->top || b->x > bounds->right || b->y > bounds->bottom)
			continue;

		considered++;
		for (int j = 0; j < nleds && ncand < MAX_GATE_CANDIDATES; j++) {
			double sqerror, rank_cost;
			if (led_blob_match_cost(match_info->visible_leds + j, b, &sqerror, &rank_cost)) {
				cands[ncand].sqerror = (float)sqerror;
				cands[ncand].rank_cost = (float)rank_cost;
				cands[ncand].blob_idx = (uint16_t)i;
				cands[ncand].led_idx = (uint16_t)j;
				ncand++;
			}
		}
	}

	sort_gate_candidates(cands, ncand);

	bool blob_used[MAX_BLOBS_PER_FRAME] = {false};
	bool led_used[MAX_OBJECT_LEDS] = {false};
	for (int c = 0; c < ncand; c++) {
		const int bi = cands[c].blob_idx, li = cands[c].led_idx;
		if (blob_used[bi] || led_used[li])
			continue; /* blob or LED already taken by a lower-cost pairing */
		blob_used[bi] = led_used[li] = true;

		struct blob *b = blobs + bi;
		struct pose_metrics_visible_led_info *led_info = match_info->visible_leds + li;
		led_info->matched_blob = b;
		match_info->reprojection_error += cands[c].sqerror;
		match_info->matched_blobs++;

		if (b->led_id != LED_INVALID_ID && b->led_id != LED_MAKE_ID(led_model->id, led_info->led->id)) {
			all_led_ids_matched = false; /* prior label disagrees with this assignment */
		}
	}

	match_info->unmatched_blobs = considered - match_info->matched_blobs;
	match_info->all_led_ids_matched = all_led_ids_matched;

	/* Build the per-LED detection log-likelihood sums (half the matcher's nats objective) from the
	 * finished assignment; the reprojection-fit half is the per-LED mean of reprojection_error. */
	compute_data_nll(match_info);
}

void
pose_metrics_evaluate_pose_with_prior(struct pose_metrics *score,
                                      struct xrt_pose *pose,
                                      bool prior_must_match,
                                      struct xrt_pose *pose_prior,
                                      const struct xrt_vec3 *pos_error_thresh,
                                      const struct xrt_vec3 *rot_error_thresh,
                                      struct blob *blobs,
                                      int num_blobs,
                                      struct t_constellation_led_model *led_model,
                                      struct camera_model *calib,
                                      struct pose_rect *out_bounds,
                                      struct pose_metrics_blob_match_info *out_match_info)
{
	/*
	 * 1. Project the LED points with the provided pose
	 * 2. Build a bounding box for the points
	 * 3. For blobs within the bounding box, see if they match a LED
	 * 4. Count up the matched LED<->Blob correspondences, and the reprojection error
	 */
	struct pose_metrics_blob_match_info blob_match_info;

	pose_metrics_match_pose_to_blobs(pose, blobs, num_blobs, led_model, calib, &blob_match_info);

	assert(led_model->num_leds > 0);
	assert(num_blobs > 0);
	assert(score != NULL);

	score->match_flags = 0;
	score->reprojection_error = blob_match_info.reprojection_error;
	score->matched_blobs = blob_match_info.matched_blobs;
	score->unmatched_blobs = blob_match_info.unmatched_blobs;
	score->visible_leds = blob_match_info.num_visible_leds;

	if (blob_match_info.all_led_ids_matched) {
		score->match_flags |= POSE_MATCH_LED_IDS;
	}

	/* If we have a pose prior, calculate the rotation and translation error and match flags as needed */
	if (pose_prior) {
		/* We can't validate a prior without error bounds */
		assert(pos_error_thresh != NULL);
		assert(rot_error_thresh != NULL);

		check_pose_prior(score, pose, pose_prior, pos_error_thresh, rot_error_thresh);
	}

	/* Don't add GOOD/STRONG flags if matched fewer than 3 blobs */
	if (score->matched_blobs < 3) {
		goto done;
	}

	double error_per_led = score->reprojection_error / score->matched_blobs;

	/* At this point, we have at least 3 LEDs and their blobs matching */
	if (POSE_HAS_FLAGS(score, POSE_MATCH_POSITION | POSE_MATCH_ORIENT)) {
		/* Three routes to GOOD:
		 * (A) "cluster mostly explained" — require matched >= 5 for priorless texture safety;
		 * (B) "covers >= 2/3 of visible LEDs";
		 * (C) a minimal 4-LED pose that is prior-consistent, blob-label-compatible, and clutter-free.
		 * Route C is not a looser texture gate: blob labels are tracker state, not physical LED IDs, so
		 * they are only a weak contradiction check. The hard evidence is still prior agreement, no clutter,
		 * and low reprojection error. This keeps side/edge-FOV frames from being discarded solely because
		 * the geometric model projects many more LEDs than the camera actually extracts at high speed or
		 * oblique angles. */
		const bool ratio_a_clean_cluster = score->unmatched_blobs * 4 <= score->matched_blobs &&
		                                   score->matched_blobs >= 5;
		const bool ratio_b_covers_visible = 2 * score->visible_leds <= 3 * score->matched_blobs;
		const bool ratio_c_prior_supported_minimal = score->matched_blobs >= 4 &&
		                                             score->unmatched_blobs == 0 &&
		                                             POSE_HAS_FLAGS(score, POSE_MATCH_LED_IDS);
		if (error_per_led < 2.0 &&
		    (ratio_a_clean_cluster || ratio_b_covers_visible || ratio_c_prior_supported_minimal)) {
			score->match_flags |= POSE_MATCH_GOOD;
			if (ratio_c_prior_supported_minimal && !ratio_a_clean_cluster && !ratio_b_covers_visible) {
				score->match_flags |= POSE_MATCH_PRIOR_SUPPORTED_PARTIAL;
			}

			if (error_per_led < 1.5)
				score->match_flags |= POSE_MATCH_STRONG;
		}
	} else if (prior_must_match) {
		/* If we must match the prior and failed, bail out */
		goto done;
	} else if (score->visible_leds > 6 && score->matched_blobs > 6 && error_per_led < 3.0 &&
	           (score->unmatched_blobs * 4 <= score->matched_blobs ||
	            (2 * score->visible_leds <= 3 * score->matched_blobs))) {
		/* If we matched all the blobs in the pose bounding box (allowing 25% noise / overlapping blobs)
		 * or if we matched a large proportion (2/3) of the LEDs we expect to be visible, then consider this a
		 * good pose match */
		score->match_flags |= POSE_MATCH_GOOD;

		/* If we had no pose prior, but a close reprojection error, allow a STRONG match */
		/* If we had a pose prior and got here, the pose is out of tolerance, so only permit a "GOOD" match */
		if (pose_prior == NULL && error_per_led < 1.5)
			score->match_flags |= POSE_MATCH_STRONG;
	}

done:
	if (out_bounds)
		*out_bounds = blob_match_info.bounds;
	if (out_match_info)
		*out_match_info = blob_match_info;
}

void
pose_metrics_evaluate_pose(struct pose_metrics *score,
                           struct xrt_pose *pose,
                           struct blob *blobs,
                           int num_blobs,
                           struct t_constellation_led_model *led_model,
                           struct camera_model *calib,
                           struct pose_rect *out_bounds)
{
	pose_metrics_evaluate_pose_with_prior(score, pose, false, NULL, NULL, NULL, blobs, num_blobs, led_model, calib,
	                                      out_bounds, NULL);
}

/* Return true if new_score is for a more likely pose than old_score */
bool
pose_metrics_score_is_better_pose_prior(struct pose_metrics *old_score,
                                        double old_prior_cost,
                                        struct pose_metrics *new_score,
                                        double new_prior_cost)
{
	/* if our previous best pose was "strong", only take better "strong" poses */
	if (POSE_HAS_FLAGS(old_score, POSE_MATCH_STRONG) && !POSE_HAS_FLAGS(new_score, POSE_MATCH_STRONG))
		return false;

	/* If the old score wasn't any good, but the new one is - take the new one */
	if (!POSE_HAS_FLAGS(old_score, POSE_MATCH_GOOD) && POSE_HAS_FLAGS(new_score, POSE_MATCH_GOOD))
		return true;

	/* Fold the soft prior penalty into the reprojection cost (both summed px^2): the lowest combined
	 * cost = reprojection_error + prior penalty wins. With zero prior costs this is the plain
	 * reprojection ordering. */
	const double new_cost = new_score->reprojection_error + new_prior_cost;
	const double old_cost = old_score->reprojection_error + old_prior_cost;
	double new_cost_per_led = new_cost / new_score->matched_blobs;
	double best_cost_per_led = 10.0;

	if (old_score->matched_blobs > 0)
		best_cost_per_led = old_cost / old_score->matched_blobs;

	if (old_score->matched_blobs < new_score->matched_blobs && (new_cost_per_led < best_cost_per_led))
		return true; /* prefer more matched blobs with tighter cost/LED */

	if (old_score->matched_blobs + 1 < new_score->matched_blobs && (new_cost_per_led < best_cost_per_led * 1.1))
		return true; /* prefer at least 2 more matched blobs with slightly worse cost/LED */

	if (old_score->matched_blobs == new_score->matched_blobs && new_cost < old_cost)
		return true; /* equal matches: prefer the lower combined cost (resolves the mirror-twin tie) */

	/* Final tiebreak when the combined costs don't decide (e.g. no prior penalty supplied): prefer the
	 * pose whose orientation better matches the prior. The anisotropic prior penalty above subsumes this
	 * when supplied; this is the ordering for the zero-prior-cost callers. */
	if (POSE_HAS_FLAGS(old_score, POSE_HAD_PRIOR) && POSE_HAS_FLAGS(new_score, POSE_HAD_PRIOR)) {
		if (m_vec3_len(new_score->orient_error) < m_vec3_len(old_score->orient_error)) {
			return true;
		}
	}

	return false;
}
