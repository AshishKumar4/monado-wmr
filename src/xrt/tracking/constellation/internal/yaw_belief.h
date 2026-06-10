/*!
 * @file
 * @brief  Kept-alive two-mode yaw belief for pure-yaw mirror ambiguities.
 * @ingroup constellation
 *
 * A genuine pure-yaw tie has exactly two pose modes: the selected pose and its near-180 degree yaw twin.
 * Single-frame data and temporal costs can tie on the seed frame. This module keeps both modes alive,
 * accumulates per-frame optical evidence, and returns either "defer orientation" or "commit one mode".
 * It is pure bookkeeping: no tracker structs, no I/O, and no blob mutation.
 */
#pragma once

#include "xrt/xrt_defines.h"

#include "math/m_api.h"

#include "pose_metrics.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//! The action the belief returns for the device this frame; maps 1:1 onto the existing observation ladder.
enum yaw_belief_action
{
	YAW_BELIEF_EXPIRE = 0,      //!< the episode is over (resolved/stale/lost): use the normal association ladder
	YAW_BELIEF_COMMIT_MODE = 1, //!< one mode dominates: commit its candidate as a normal POSE_LOCK, clear the belief
	YAW_BELIEF_DEFER = 2,       //!< still ambiguous: keep both modes alive, fold the device position-only
};

//! The kept-alive 2-mode yaw belief state, stored on the persistent device next to the temporal_ref_* block.
//! Zero-initialised -> active=false (inert). The two modes are FULL committed orientations (world<-object,
//! OpenCV — the same frame as temporal_ref_*), not a yaw scalar, so the per-frame match reuses
//! pose_metrics_temporal_yaw_cost unchanged and tilt stays owned by the abs-tilt keystone. mode_q[0] is kept
//! the higher-weight (MAP) mode after every update.
struct constellation_yaw_belief
{
	bool active;
	uint64_t seed_ts;          //!< commit ts of the seed frame (episode start; the TTL anchor)
	uint32_t frames;           //!< frames the belief has stayed unresolved (the max-frames bound)
	uint64_t last_update_ts;   //!< ts of the most recent propagation
	struct xrt_quat mode_q[2]; //!< the two live yaw modes, world<-object OpenCV; mode_q[0] == current MAP
	float mode_logw[2];        //!< un-normalised log-weights; posterior == softmax(mode_logw)
	/* The gyro-propagation anchor captured at the seed (mirrors temporal_ref_prior): the fusion prior
	 * orientation and its trust flag. The tracker propagates each mode by q_prior_now . seed_prior^-1; the
	 * pure step (yaw_belief_step) receives the already-propagated modes, so it never reads these. */
	struct xrt_quat seed_prior;
	bool seed_prior_trusted;
};

//! Per-frame, per-mode evidence reduced from the work set: the best lock-eligible candidate matching this mode,
//! its world<-object orientation, and its total_nll data terms (the nats the matcher already computed). A mode
//! with no eligible candidate this frame has have_cand=false (it pays the fixed no-candidate cost).
struct yaw_belief_mode_evidence
{
	bool have_cand;          //!< a lock-eligible candidate continued this mode this frame
	struct xrt_quat cand_q;  //!< that candidate's world<-object orientation (OpenCV)
	float cand_total_nll;    //!< that candidate's total_nll (reprojection + missed-LED + abs-tilt data terms)
};

//! Tuning. The recall-safety (the firewalls in design §6) is constant-INDEPENDENT; these are quality knobs.
struct yaw_belief_params
{
	float commit_w;        //!< commit the MAP mode when its posterior weight clears this (≈0.9 ≈ 9:1 odds)
	float no_cand_nll;     //!< the per-frame evidence a mode with no eligible candidate pays
	uint64_t ttl_ns;       //!< the episode expires this long after the seed (graceful return to temporal_nll)
	uint32_t max_frames;   //!< the episode expires after this many unresolved frames
	double yaw_sigma_rad;  //!< per-mode yaw-continuity scale
	double tilt_sigma_rad; //!< tilt scale, normally wide so this belief only resolves yaw
	double huber_knee;     //!< the robust knee (== FLIP_COST_HUBER_KNEE_SIGMA)
	double weight;         //!< the continuity weight (== FLIP_COST_WEIGHT)
	double cap;            //!< the per-frame yaw-continuity cap (== ASSOC_TEMPORAL_MAX_NLL)
};

//! posterior weight of mode m from the log-weights (numerically stable two-mode softmax).
static inline float
yaw_belief_posterior(const struct constellation_yaw_belief *b, int m)
{
	const float other = b->mode_logw[m ^ 1];
	return 1.0f / (1.0f + expf(other - b->mode_logw[m]));
}

//! Seed the belief on a genuine pure-yaw tie: both modes equiprobable, the propagation clock at the seed.
//! @p chosen_q is the committed (position-only) winner, @p twin_q its near-pure-yaw mirror; @p seed_prior +
//! @p seed_prior_trusted are the fusion-prior gyro-propagation anchor (mirrors temporal_ref_prior).
static inline void
yaw_belief_seed(struct constellation_yaw_belief *b,
                uint64_t seed_ts,
                const struct xrt_quat *chosen_q,
                const struct xrt_quat *twin_q,
                const struct xrt_quat *seed_prior,
                bool seed_prior_trusted)
{
	b->active = true;
	b->seed_ts = seed_ts;
	b->last_update_ts = seed_ts;
	b->frames = 0;
	b->mode_q[0] = *chosen_q;
	b->mode_q[1] = *twin_q;
	b->mode_logw[0] = 0.0f;
	b->mode_logw[1] = 0.0f;
	b->seed_prior = *seed_prior;
	b->seed_prior_trusted = seed_prior_trusted;
}

//! One per-frame belief update (the (E)+(P)+(L)+(U)+(C)+(D) block of design §4), as a pure function of the
//! belief, this frame's timestamp, the two modes' gyro-propagated orientations @p prop, and the per-mode
//! evidence @p ev. Returns the action; on YAW_BELIEF_COMMIT_MODE, *@p out_commit_mode is the MAP mode index
//! (its candidate, ev[*out_commit_mode].cand_q, is the orientation to lock). The caller does the propagation
//! (it owns the gyro) and the candidate matching (it owns the work set); this owns the Bayesian bookkeeping,
//! the commit rule, and the bounded lifetime. mode_q is advanced to @p prop so the belief tracks the turn.
static inline enum yaw_belief_action
yaw_belief_step(struct constellation_yaw_belief *b,
                uint64_t now_ts,
                const struct xrt_quat prop[2],
                const struct yaw_belief_mode_evidence ev[2],
                const struct yaw_belief_params *p,
                int *out_commit_mode)
{
	/* Expire on a stale episode: a dropout or turn outran both modes. */
	if (now_ts < b->seed_ts || now_ts - b->seed_ts >= p->ttl_ns || b->frames >= p->max_frames) {
		b->active = false;
		return YAW_BELIEF_EXPIRE;
	}

	const struct xrt_vec3 up = {0.f, 1.f, 0.f};
	float evidence_nll[2];
	for (int m = 0; m < 2; m++) {
		if (ev[m].have_cand) {
			/* (L) per-mode evidence = the candidate's yaw-continuity NLL vs the propagated mode + its own
			 *     total_nll data terms — exactly the nats the matcher and temporal_nll already speak. */
			const double raw_yaw_nll =
			    pose_metrics_prior_orient_cost(&ev[m].cand_q, &prop[m], &up, p->tilt_sigma_rad,
			                                   p->yaw_sigma_rad, p->huber_knee, p->weight);
			const double yaw_nll = raw_yaw_nll > p->cap ? p->cap : raw_yaw_nll;
			evidence_nll[m] = (float)yaw_nll + ev[m].cand_total_nll;
		} else {
			evidence_nll[m] = p->no_cand_nll; /* the mode lost its candidate this frame (occlusion) */
		}
	}

	/* (U) Bayesian log-weight update: logw[m] += -evidence_nll[m]; renormalise by the max for stability. */
	b->mode_logw[0] += -evidence_nll[0];
	b->mode_logw[1] += -evidence_nll[1];
	const float wmax = fmaxf(b->mode_logw[0], b->mode_logw[1]);
	b->mode_logw[0] -= wmax;
	b->mode_logw[1] -= wmax;
	b->mode_q[0] = prop[0];
	b->mode_q[1] = prop[1];
	b->last_update_ts = now_ts;
	b->frames++;

	/* The MAP mode and its posterior weight, in the CURRENT (== caller's ev) index order. */
	const int map = b->mode_logw[0] >= b->mode_logw[1] ? 0 : 1;
	const float w_map = yaw_belief_posterior(b, map);

	/* (C) commit the MAP mode when it dominates and still has a candidate this frame to lock its orientation. */
	if (w_map >= p->commit_w && ev[map].have_cand) {
		*out_commit_mode = map;
		b->active = false;
		return YAW_BELIEF_COMMIT_MODE;
	}

	/* (D) still ambiguous: keep both modes alive, the caller folds the device position-only. Keep mode[0] the
	 *     MAP so the persisted state is canonical (no commit reads ev after this, so the swap is purely cosmetic
	 *     bookkeeping for the next frame's seed/inspection). */
	if (map == 1) {
		const struct xrt_quat tq = b->mode_q[0];
		const float tw = b->mode_logw[0];
		b->mode_q[0] = b->mode_q[1];
		b->mode_logw[0] = b->mode_logw[1];
		b->mode_q[1] = tq;
		b->mode_logw[1] = tw;
	}
	return YAW_BELIEF_DEFER;
}

#ifdef __cplusplus
}
#endif
