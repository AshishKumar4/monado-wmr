// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct l2_accept_evidence
{
	int matched_count;
	int distinct_view_count;
	double reproj_per_led_px2;
	double total_nll;
	double tilt_error_rad;
	bool tilt_valid;
	bool pose_match_good;
	bool already_lock_eligible;
};

struct l2_accept_params
{
	int min_matched;
	int min_distinct_views;
	double max_tilt_rad;
	double max_reproj_per_led_px2;
	double max_total_nll;
};

struct single_view_prior_gate_evidence
{
	int distinct_view_count;
	double prior_nll;
	double prior_pos_err_m;
	bool has_pose;
};

struct single_view_prior_gate_params
{
	double max_prior_nll;
	double max_prior_pos_err_m;
	double min_prior_nll_for_pos_gate;
};

static inline bool
single_view_prior_gate_disagrees(const struct single_view_prior_gate_evidence *e,
                                 const struct single_view_prior_gate_params *p)
{
	if (e == NULL || p == NULL || !e->has_pose || e->distinct_view_count >= 2) {
		return false;
	}
	if (e->prior_nll > p->max_prior_nll) {
		return true;
	}
	return e->prior_pos_err_m > p->max_prior_pos_err_m && e->prior_nll > p->min_prior_nll_for_pos_gate;
}

static inline bool
l2_accept_recoverable(const struct l2_accept_evidence *e, const struct l2_accept_params *p)
{
	if (e == NULL || p == NULL || e->already_lock_eligible) {
		return false;
	}
	if (!e->tilt_valid || e->tilt_error_rad > p->max_tilt_rad) {
		return false;
	}
	if (e->distinct_view_count < p->min_distinct_views || e->matched_count < p->min_matched) {
		return false;
	}
	if (e->reproj_per_led_px2 > p->max_reproj_per_led_px2) {
		return false;
	}
	if (e->total_nll > p->max_total_nll) {
		return false;
	}
	return e->pose_match_good;
}

#ifdef __cplusplus
}
#endif
