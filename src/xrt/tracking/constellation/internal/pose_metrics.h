// Copyright 2020-2023 Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
/*
 * Ported from OpenHMD - Free and Open Source API and drivers for immersive technology.
 */
/*!
 * @file
 * @brief  Metrics for constellation tracking poses
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup constellation
 */
#pragma once

#include "xrt/xrt_defines.h"
#include "tracking/t_led_models.h"

#include "blobwatch.h"
#include "camera_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_OBJECT_LEDS 64

struct pose_rect
{
	double left;
	double top;
	double right;
	double bottom;
};

XRT_MAYBE_UNUSED static bool
pose_rect_has_area(struct pose_rect *rect)
{
	return rect->left != rect->right && rect->top != rect->bottom;
}

enum pose_match_flags
{
	POSE_MATCH_GOOD = 0x1,      /* A reasonable pose match - most LEDs matched to within a few pixels error */
	POSE_MATCH_STRONG = 0x2,    /* A strong pose match is a match with very low error */
	POSE_MATCH_POSITION = 0x10, /* The position of the pose matched the prior well */
	POSE_MATCH_ORIENT = 0x20,   /* The orientation of the pose matched the prior well */
	POSE_HAD_PRIOR =
	    0x100, /* If a pose prior was supplied when calculating the score, then rot/trans_error are set */
	POSE_MATCH_LED_IDS = 0x200, /* The LED IDs on the blobs all matched the LEDs we thought (or were unassigned) */
	POSE_MATCH_PRIOR_SUPPORTED_PARTIAL =
	    0x400, /* Prior-consistent sparse visual cue: valid for per-LED fusion, not a standalone 6DoF pose */
};

#define POSE_SET_FLAG(score, f) ((score)->match_flags |= (f))
#define POSE_CLEAR_FLAG(score, f) ((score)->match_flags &= ~(f))
#define POSE_HAS_FLAGS(score, f) (((score)->match_flags & (f)) == (f))

struct pose_metrics
{
	enum pose_match_flags match_flags;

	int matched_blobs;
	int unmatched_blobs;
	int visible_leds;

	double reprojection_error;

	struct xrt_vec3 orient_error; /* Rotation error (compared to a prior) */
	struct xrt_vec3 pos_error;    /* Translation error (compared to a prior) */
};

struct pose_metrics_visible_led_info
{
	struct t_constellation_led *led;
	double led_radius_px;   /* Expected max size of the LED in pixels at that distance */
	struct xrt_vec2 pos_px; /* Projected position of the LED (pixels) */
	struct xrt_vec3 pos_m;  /* Projected physical position of the LED (metres) */
	double facing_dot;      /* Dot product between LED and camera */
	/* Anisotropic blob<->LED match gate (px half-axes) derived from the prior
	 * covariance projected at this LED's depth. Floors at led_radius_px. */
	double gate_ax_px;
	double gate_ay_px;
	struct blob *matched_blob;
};

struct pose_metrics_blob_match_info
{
	struct pose_metrics_visible_led_info visible_leds[MAX_OBJECT_LEDS];
	int num_visible_leds;

	bool all_led_ids_matched;
	int matched_blobs;
	int unmatched_blobs;

	double reprojection_error;

	/* Visible-LED detection likelihood in nats. The matcher subtracts the matched-LED all-missed
	 * reference so explaining high-probability LEDs is rewarded without unbounded count hacks. */
	double data_nll_detection;
	double data_nll_missed_if_matched;

	struct pose_rect bounds;
};

/* Anisotropic prior-consistency split for mirror-flip rejection. Decompose the candidate orientation's
 * difference from the (gyro+gravity) prior, RELATIVE to the world-up axis @p up expressed in the SAME frame
 * as both orientations (camera-frame up for the ab-initio prior cost, world-frame down for the front-end
 * twin pick — either works, the split is frame-agnostic given a consistent up), into:
 *   - out_tilt_rad: the SWING (off-axis) angle == how far the candidate mis-places gravity vs the prior.
 *     DRIFTLESS (the accelerometer pins the prior's tilt even through an optical dropout) -> gate TIGHT.
 *   - out_yaw_rad : the TWIST (about-axis) angle == the yaw difference. Drifts with the gyro bias ->
 *     gate only by the covariance-sized bound, and only while the prior yaw is trusted-fresh.
 * A tilt flip -> large tilt; a pure-yaw flip -> ~0 tilt + large yaw. @p up need not be unit length
 * (normalised internally); a degenerate (zero-length) up yields tilt=0, deferring entirely to the yaw
 * bound. */
void
pose_metrics_prior_orient_split(const struct xrt_quat *q_cand,
                                const struct xrt_quat *q_prior,
                                const struct xrt_vec3 *up,
                                double *out_tilt_rad,
                                double *out_yaw_rad);

/* Soft prior-consistency penalty for ranking mirror-flip hypotheses: a robust M-estimator term to ADD to a
 * candidate's per-LED reprojection error so the lowest-cost candidate is committed and a frame is never
 * dropped for ambiguity. From the split:
 *   d2 = (tilt/@p sigma_tilt_rad)^2 + (yaw/@p sigma_yaw_rad)^2   (anisotropic squared Mahalanobis distance
 *        of the candidate orientation from the prior; tilt+yaw from pose_metrics_prior_orient_split)
 * is Huber-robustified about @p huber_knee_sigma (the standardized-residual knee, in sigmas) so a gross
 * outlier (a flip) bends to a LINEAR penalty and cannot dominate pathologically, then scaled by
 * @p weight (in the caller's reprojection-error units, px^2 per unit robustified distance) to be
 * commensurate with the reprojection error: return weight * huber(sqrt(d2)).
 * sigma_tilt is the tight DRIFTLESS gravity-anchored bound; sigma_yaw is the live fusion yaw 1-sigma:
 * confident yaw (small sigma) -> the prior term dominates and selects the prior-consistent twin (a 90 deg
 * tilt flip has huge d2 -> never selected); uncertain/untracked yaw (large sigma) -> the yaw term vanishes
 * and reprojection decides (cold-start/ab-initio bootstrap). @p up is the world-up in the same frame as both
 * orientations (see pose_metrics_prior_orient_split). A non-positive sigma is treated as +inf for that axis
 * (that axis contributes nothing). */
double
pose_metrics_prior_orient_cost(const struct xrt_quat *q_cand,
                               const struct xrt_quat *q_prior,
                               const struct xrt_vec3 *up,
                               double sigma_tilt_rad,
                               double sigma_yaw_rad,
                               double huber_knee_sigma,
                               double weight);

void
pose_metrics_get_device_bounds(struct xrt_pose *P_cam_obj,
                               struct t_constellation_led_model *led_model,
                               struct camera_model *calib,
                               struct pose_rect *device_bounds,
                               struct xrt_vec2 *visible_points,
                               size_t *num_visible_points);

void
pose_metrics_match_pose_to_blobs(struct xrt_pose *pose,
                                 struct blob *blobs,
                                 int num_blobs,
                                 struct t_constellation_led_model *led_model,
                                 struct camera_model *calib,
                                 struct pose_metrics_blob_match_info *match_info);

/* As above, but size each LED's blob-match gate from the prior's per-axis
 * position/rotation uncertainty (anisotropic Mahalanobis ellipse) instead of a
 * fixed radius. NULL thresholds => the fixed-radius (isotropic) behaviour. */
void
pose_metrics_match_pose_to_blobs_prior(struct xrt_pose *pose,
                                       struct blob *blobs,
                                       int num_blobs,
                                       const struct xrt_vec3 *pos_error_thresh,
                                       const struct xrt_vec3 *rot_error_thresh,
                                       struct t_constellation_led_model *led_model,
                                       struct camera_model *calib,
                                       struct pose_metrics_blob_match_info *match_info);

/* Pure gate math (exposed for unit testing): compute the anisotropic per-LED
 * gate half-axes (px) from the prior covariance projected at the LED depth. */
void
pose_metrics_compute_led_gate(double focal_length_px,
                              double led_depth_m,
                              double led_lever_arm_m,
                              double led_radius_px,
                              const struct xrt_vec3 *pos_error_thresh,
                              const struct xrt_vec3 *rot_error_thresh,
                              double *out_gate_ax_px,
                              double *out_gate_ay_px);

void
pose_metrics_evaluate_pose(struct pose_metrics *score,
                           struct xrt_pose *pose,
                           struct blob *blobs,
                           int num_blobs,
                           struct t_constellation_led_model *leds_model,
                           struct camera_model *calib,
                           struct pose_rect *out_bounds);

void
pose_metrics_evaluate_pose_with_prior(struct pose_metrics *score,
                                      struct xrt_pose *pose,
                                      bool prior_must_match,
                                      struct xrt_pose *pose_prior,
                                      const struct xrt_vec3 *pos_error_thresh,
                                      const struct xrt_vec3 *rot_error_thresh,
                                      struct blob *blobs,
                                      int num_blobs,
                                      struct t_constellation_led_model *leds_model,
                                      struct camera_model *calib,
                                      struct pose_rect *out_bounds);

bool
pose_metrics_score_is_better_pose(struct pose_metrics *old_score, struct pose_metrics *new_score);

/* As pose_metrics_score_is_better_pose, but the soft mirror-flip prior penalty (pose_metrics_prior_orient_cost)
 * is folded into the per-LED reprojection comparison: where two candidates have the same matched-blob count
 * (the mirror-twin case — twins reproject near-identically), the one with the lower (reproj + prior_penalty)
 * wins. This is the soft re-rank used in the ab-initio search: a flipped twin is out-ranked by its large
 * prior penalty rather than dropped, and a candidate is never rejected outright. @p old_prior_cost /
 * @p new_prior_cost are the summed prior penalties (px^2) for each candidate. */
bool
pose_metrics_score_is_better_pose_prior(struct pose_metrics *old_score,
                                        double old_prior_cost,
                                        struct pose_metrics *new_score,
                                        double new_prior_cost);

#ifdef __cplusplus
}
#endif
