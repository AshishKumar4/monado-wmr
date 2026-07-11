// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Per-stream frame-timestamp guard: strictly-monotonic, plausible timelines.
 *
 * External SLAM systems (Basalt) require strictly increasing frame timestamps and turn any
 * regression into an assert/abort inside the calling process (fatal inside SteamVR's vrserver).
 * Regressions are real on live WMR hardware: the camera device->monotonic conversion offset is
 * re-estimated per IMU sample from packet arrival times, so USB churn (session launch windows)
 * inflates it by ~100 ms and the decay back regresses consecutive converted camera-group
 * timestamps by up to ~80 ms (2026-07-06 crash forensics, captures/20260706-*-s3-leg*). Rare
 * single-frame garbage timestamps (~1e19 ns) were also recorded in the same session.
 *
 * The guard admits a frame only if its timestamp advances the pushed timeline AND does not jump
 * implausibly far ahead of the most recently seen candidate, so:
 *  - a regressed frame is dropped, and the stream resumes as soon as the timeline re-exceeds the
 *    pushed maximum (the observed crash sequence loses 5 of 97 groups, then flows);
 *  - a garbage far-future timestamp is dropped WITHOUT poisoning the pushed maximum (it can never
 *    stall the stream), at a worst-case cost of one extra frame;
 *  - a legitimate long gap (submission re-enabled after a pause) costs exactly one frame, since
 *    the seen-timestamp reference catches up on the first candidate after the gap.
 *
 * State is per stream (per camera). Pure, allocation-free, unit-tested against the recorded
 * crash sequences (tests/tests_frame_ts_guard.cpp).
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! Largest plausible forward step between consecutively received candidates of one stream.
 * Frames arrive at >= 30 Hz; the longest recorded healthy gap is ~120 ms (dropped groups). A
 * candidate further than this ahead of the previously seen one is treated as implausible. */
#define U_FRAME_TS_GUARD_MAX_STEP_NS (1000LL * 1000LL * 1000LL)

enum u_frame_ts_guard_verdict
{
	U_FRAME_TS_GUARD_ACCEPT = 0,
	//! Timestamp does not advance the pushed timeline; forwarding it would abort the consumer.
	U_FRAME_TS_GUARD_DROP_REGRESSED,
	//! Timestamp jumps implausibly far ahead of the last seen candidate (garbage protection).
	U_FRAME_TS_GUARD_DROP_JUMP,
};

struct u_frame_ts_guard
{
	bool has_pushed;
	int64_t last_pushed_ns; //!< Timestamp of the last admitted frame (the consumer's timeline)
	int64_t last_seen_ns;   //!< Timestamp of the last candidate, admitted or not
};

static inline void
u_frame_ts_guard_init(struct u_frame_ts_guard *g)
{
	g->has_pushed = false;
	g->last_pushed_ns = 0;
	g->last_seen_ns = 0;
}

static inline enum u_frame_ts_guard_verdict
u_frame_ts_guard_check(struct u_frame_ts_guard *g, int64_t ts_ns)
{
	if (g->has_pushed) {
		if (ts_ns <= g->last_pushed_ns) {
			g->last_seen_ns = ts_ns;
			return U_FRAME_TS_GUARD_DROP_REGRESSED;
		}
		/* Plausible relative to EITHER recent reference: the pushed timeline (so a frame
		 * following a dropped garbage candidate is not penalized) or the last candidate
		 * (so the stream re-anchors right after a legitimate long gap). Differences are
		 * computed unsigned; with ts > reference the uint64 subtraction is exact even
		 * when the reference is a garbage negative. */
		uint64_t vs_pushed = (uint64_t)ts_ns - (uint64_t)g->last_pushed_ns;
		bool plausible = vs_pushed <= (uint64_t)U_FRAME_TS_GUARD_MAX_STEP_NS;
		if (!plausible && ts_ns > g->last_seen_ns) {
			uint64_t vs_seen = (uint64_t)ts_ns - (uint64_t)g->last_seen_ns;
			plausible = vs_seen <= (uint64_t)U_FRAME_TS_GUARD_MAX_STEP_NS;
		}
		if (!plausible) {
			g->last_seen_ns = ts_ns;
			return U_FRAME_TS_GUARD_DROP_JUMP;
		}
	}
	g->has_pushed = true;
	g->last_pushed_ns = ts_ns;
	g->last_seen_ns = ts_ns;
	return U_FRAME_TS_GUARD_ACCEPT;
}

#ifdef __cplusplus
}
#endif
