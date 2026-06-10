#pragma once

#include "pose_metrics.h"

#include <math.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct joint_contention_cluster
{
	int n_rows;
	int n_cols;
	int row_device[PKF_MAX_CLUSTER];
	double L[PKF_MAX_CLUSTER * PKF_MAX_CLUSTER];
	double L_clutter[PKF_MAX_CLUSTER];
	double row_rho[PKF_MAX_CLUSTER];
	double row_logodds[PKF_MAX_CLUSTER];
};

struct joint_contention_result
{
	bool valid;
	double joint_marginal;
	double dev_hard_rho[PKF_MAX_CLUSTER + 1];
	double dev_marginal[PKF_MAX_CLUSTER + 1];
};

static inline double
joint_contention_minor(const struct joint_contention_cluster *c, int drop_row, int drop_col)
{
	double L[PKF_MAX_CLUSTER * PKF_MAX_CLUSTER];
	double L_clutter[PKF_MAX_CLUSTER];
	int cmap[PKF_MAX_CLUSTER];
	int m = 0, n = 0;

	for (int j = 0; j < c->n_cols; j++) {
		if (j != drop_col) {
			cmap[n++] = j;
		}
	}
	for (int i = 0; i < c->n_rows; i++) {
		if (i == drop_row) {
			continue;
		}
		L_clutter[m] = c->L_clutter[i];
		for (int j = 0; j < n; j++) {
			L[m * n + j] = c->L[i * c->n_cols + cmap[j]];
		}
		m++;
	}

	return m == 0 ? 1.0 : pose_metrics_pkf_permanent_augmented(L, L_clutter, m, n);
}

static inline struct joint_contention_result
joint_contention_split(const struct joint_contention_cluster *c, int max_slot)
{
	struct joint_contention_result out = {false, 0.0, {0}, {0}};
	if (c->n_rows < 2 || c->n_rows > PKF_MAX_CLUSTER || c->n_cols < 1 || c->n_cols > PKF_MAX_CLUSTER ||
	    max_slot < 0 || max_slot > PKF_MAX_CLUSTER) {
		return out;
	}

	double dev_logodds[PKF_MAX_CLUSTER + 1] = {0};
	double dev_expected[PKF_MAX_CLUSTER + 1] = {0};
	double total_rho = 0.0;
	for (int i = 0; i < c->n_rows; i++) {
		const int d = c->row_device[i];
		if (d < 0 || d > max_slot) {
			return out;
		}
		out.dev_hard_rho[d] += c->row_rho[i];
		dev_logodds[d] += c->row_logodds[i];
		total_rho += c->row_rho[i];
	}
	if (!(total_rho > 0.0)) {
		return out;
	}

	const double per = pose_metrics_pkf_permanent_augmented(c->L, c->L_clutter, c->n_rows, c->n_cols);
	if (!(per > 0.0) || !isfinite(per)) {
		return out;
	}
	out.joint_marginal = -log(per);

	for (int i = 0; i < c->n_rows; i++) {
		const int d = c->row_device[i];
		const double per_no_row = joint_contention_minor(c, i, -1);
		double expected = 0.0;
		const double p_clutter = c->L_clutter[i] * per_no_row / per;
		if (p_clutter > 0.0) {
			expected += p_clutter * (-log(c->L_clutter[i]));
		}
		for (int j = 0; j < c->n_cols; j++) {
			const double lij = c->L[i * c->n_cols + j];
			if (!(lij > 0.0)) {
				continue;
			}
			const double p = lij * joint_contention_minor(c, i, j) / per;
			if (p > 0.0) {
				expected += p * (-log(lij));
			}
		}
		dev_expected[d] += expected;
	}

	for (int d = 0; d <= max_slot; d++) {
		const double marginal = dev_expected[d] - dev_logodds[d];
		out.dev_marginal[d] = marginal < out.dev_hard_rho[d] ? marginal : out.dev_hard_rho[d];
	}
	out.valid = true;
	return out;
}

static inline int
joint_contention_blob_owner(const struct joint_contention_cluster *c, int col)
{
	int owner = -1;
	double best = 0.0;
	for (int i = 0; i < c->n_rows; i++) {
		const double l = c->L[i * c->n_cols + col];
		if (l > best) {
			best = l;
			owner = c->row_device[i];
		}
	}
	return owner;
}

#ifdef __cplusplus
}
#endif
