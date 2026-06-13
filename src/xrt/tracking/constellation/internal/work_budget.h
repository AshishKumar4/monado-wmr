// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Deterministic pre-assignment of the cold-search work-unit budget.
 * @ingroup constellation
 *
 * The associator's per-frame work-unit budget (1 unit = 1 P3P trial; see
 * correspondence_search.c) is PRE-ASSIGNED to every search scope from frame content +
 * tracker state before any task runs — never a shared pool drained in completion order —
 * so identical inputs spend identical work units, live and offline. These helpers are the
 * pure integer rules that make that pre-assignment reproducible.
 */
#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define G2_WORK_BUDGET_MAX_SCOPES 32

/*!
 * Split @p total work units across @p n scopes proportionally to @p weights with
 * largest-remainder rounding: scope i first gets floor(total * w_i / W); the leftover
 * units (at most n-1) then go one each to the scopes with the largest remainder
 * (total * w_i) mod W, ties broken by LOWER scope index. An all-zero weight vector
 * splits equally (every weight treated as 1). The result always sums to exactly
 * @p total, and the rule is a pure function of its inputs.
 */
static inline void
g2_work_budget_split(uint32_t total, const uint32_t *weights, int n, uint32_t *out_allowance)
{
	assert(n <= G2_WORK_BUDGET_MAX_SCOPES);
	if (n <= 0) {
		return;
	}

	uint64_t weight_sum = 0;
	for (int i = 0; i < n; i++) {
		weight_sum += weights[i];
	}
	const bool equal_split = weight_sum == 0;
	const uint64_t W = equal_split ? (uint64_t)n : weight_sum;

	uint32_t leftover = total;
	for (int i = 0; i < n; i++) {
		const uint64_t w = equal_split ? 1 : weights[i];
		out_allowance[i] = (uint32_t)(((uint64_t)total * w) / W);
		leftover -= out_allowance[i];
	}

	bool topped_up[G2_WORK_BUDGET_MAX_SCOPES] = {false};
	for (; leftover > 0; leftover--) {
		int best = -1;
		uint64_t best_rem = 0;
		for (int i = 0; i < n; i++) {
			const uint64_t w = equal_split ? 1 : weights[i];
			const uint64_t rem = ((uint64_t)total * w) % W;
			if (!topped_up[i] && (best < 0 || rem > best_rem)) {
				best = i;
				best_rem = rem;
			}
		}
		topped_up[best] = true;
		out_allowance[best]++;
	}
}

/*!
 * Take one bounded slice from a remaining budget: min(@p cap, *@p remaining), decrementing
 * @p remaining. Callers assign slices in a fixed (device, view) order so the sequence of
 * takes — and therefore every allowance — is deterministic.
 */
static inline uint32_t
g2_work_budget_take(uint32_t *remaining, uint32_t cap)
{
	const uint32_t take = *remaining < cap ? *remaining : cap;
	*remaining -= take;
	return take;
}

#ifdef __cplusplus
}
#endif
