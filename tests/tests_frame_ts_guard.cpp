// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Tests for u_frame_ts_guard — the SLAM frame-timestamp guard.
 *
 * The recorded fixtures are the actual cam0 EuRoC timestamp sequences of the 2026-07-06 S3
 * session (captures/20260706-*-s3-leg*): both crashed instances (leg2 production, leg1 wake)
 * aborted vrserver/monado-service through Basalt's non-monotonic-timestamp init assert, and the
 * healthy leg1 production instance ran clean. The tests assert the guard's contract on exactly
 * that data: the crashed sequences flow through with only the offending groups dropped and a
 * strictly increasing output (no input remains that could trip the assert), and the healthy
 * sequence passes untouched (the guard is inert on healthy data).
 */
#include "catch_amalgamated.hpp"

#include "util/u_frame_ts_guard.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

std::vector<int64_t>
load_fixture(const char *name)
{
	std::string path = std::string(G2_TEST_DATA_DIR) + "/" + name;
	FILE *f = fopen(path.c_str(), "r");
	REQUIRE(f != nullptr);
	std::vector<int64_t> out;
	long long v = 0;
	while (fscanf(f, "%lld", &v) == 1) {
		out.push_back((int64_t)v);
	}
	fclose(f);
	REQUIRE(!out.empty());
	return out;
}

struct run_result
{
	std::vector<size_t> dropped_rows;
	std::vector<int64_t> pushed;
	bool jump_seen = false;
};

run_result
run_guard(const std::vector<int64_t> &seq)
{
	struct u_frame_ts_guard g;
	u_frame_ts_guard_init(&g);
	run_result r;
	for (size_t i = 0; i < seq.size(); i++) {
		enum u_frame_ts_guard_verdict v = u_frame_ts_guard_check(&g, seq[i]);
		if (v == U_FRAME_TS_GUARD_ACCEPT) {
			r.pushed.push_back(seq[i]);
		} else {
			r.dropped_rows.push_back(i);
			r.jump_seen |= (v == U_FRAME_TS_GUARD_DROP_JUMP);
		}
	}
	return r;
}

void
require_strictly_increasing(const std::vector<int64_t> &ts)
{
	for (size_t i = 1; i < ts.size(); i++) {
		REQUIRE(ts[i] > ts[i - 1]); // The Basalt invariant the guard exists to uphold.
	}
}

} // namespace

TEST_CASE("crashed leg2 production sequence flows with only the regressed groups dropped")
{
	auto seq = load_fixture("ts_guard_20260706_leg2_crash_cam0.csv");
	REQUIRE(seq.size() == 97);

	auto r = run_guard(seq);
	// The recorded -58.5/-81.5 ms regressions (rows 85-86) plus the rows still below the
	// pre-regression maximum; the stream resumes at row 90 without further loss.
	CHECK(r.dropped_rows == std::vector<size_t>{85, 86, 87, 88, 89});
	CHECK_FALSE(r.jump_seen);
	CHECK(r.pushed.size() == seq.size() - 5);
	require_strictly_increasing(r.pushed);
}

TEST_CASE("crashed leg1 wake sequence flows with only the regressed groups dropped")
{
	auto seq = load_fixture("ts_guard_20260706_leg1_wake_crash_cam0.csv");
	REQUIRE(seq.size() == 76);

	auto r = run_guard(seq);
	CHECK(r.dropped_rows == std::vector<size_t>{65, 66});
	CHECK_FALSE(r.jump_seen);
	CHECK(r.pushed.size() == seq.size() - 2);
	require_strictly_increasing(r.pushed);
}

TEST_CASE("healthy leg1 production sequence passes untouched")
{
	auto seq = load_fixture("ts_guard_20260706_leg1_healthy_cam0.csv");
	REQUIRE(seq.size() == 1531);

	auto r = run_guard(seq);
	CHECK(r.dropped_rows.empty());
	CHECK(r.pushed == seq);
}

TEST_CASE("garbage timestamps cannot poison the timeline")
{
	// A single ~1.05e19 ns garbage timestamp was recorded mid-stream in the leg2 session
	// (frame.bin rows 252-255). As int64 it is negative; also cover a huge positive one.
	const int64_t period = 33'000'000;
	const int64_t garbage_negative = (int64_t)10517582490928168604ull;
	const int64_t garbage_positive = INT64_MAX - 12345;

	struct u_frame_ts_guard g;
	u_frame_ts_guard_init(&g);
	int64_t ts = 51'944'000'000'000;
	REQUIRE(u_frame_ts_guard_check(&g, ts) == U_FRAME_TS_GUARD_ACCEPT);

	SECTION("negative garbage: dropped as regression, next real frame accepted")
	{
		REQUIRE(u_frame_ts_guard_check(&g, garbage_negative) == U_FRAME_TS_GUARD_DROP_REGRESSED);
		REQUIRE(u_frame_ts_guard_check(&g, ts + period) == U_FRAME_TS_GUARD_ACCEPT);
	}

	SECTION("far-future garbage: dropped as jump, next real frame accepted")
	{
		REQUIRE(u_frame_ts_guard_check(&g, garbage_positive) == U_FRAME_TS_GUARD_DROP_JUMP);
		// last_pushed must NOT have been poisoned by the far-future value.
		REQUIRE(u_frame_ts_guard_check(&g, ts + period) == U_FRAME_TS_GUARD_ACCEPT);
		REQUIRE(u_frame_ts_guard_check(&g, ts + 2 * period) == U_FRAME_TS_GUARD_ACCEPT);
	}
}

TEST_CASE("a legitimate long gap costs exactly one frame")
{
	const int64_t period = 33'000'000;
	struct u_frame_ts_guard g;
	u_frame_ts_guard_init(&g);
	int64_t ts = 51'944'000'000'000;
	REQUIRE(u_frame_ts_guard_check(&g, ts) == U_FRAME_TS_GUARD_ACCEPT);

	// Submission paused for a minute, then resumed: the first candidate after the gap is
	// indistinguishable from garbage and is dropped; the second re-anchors the stream.
	int64_t resumed = ts + 60'000'000'000;
	REQUIRE(u_frame_ts_guard_check(&g, resumed) == U_FRAME_TS_GUARD_DROP_JUMP);
	REQUIRE(u_frame_ts_guard_check(&g, resumed + period) == U_FRAME_TS_GUARD_ACCEPT);
	REQUIRE(u_frame_ts_guard_check(&g, resumed + 2 * period) == U_FRAME_TS_GUARD_ACCEPT);
}

TEST_CASE("recorded healthy gaps stay below the jump threshold")
{
	// The longest healthy recorded inter-group gap is ~155.6 ms (dropped groups during launch
	// churn, leg1 production); the 1 s threshold keeps a >6x margin above it.
	auto seq = load_fixture("ts_guard_20260706_leg1_healthy_cam0.csv");
	int64_t max_gap = 0;
	for (size_t i = 1; i < seq.size(); i++) {
		max_gap = std::max(max_gap, seq[i] - seq[i - 1]);
	}
	CHECK(max_gap < U_FRAME_TS_GUARD_MAX_STEP_NS / 6);
}
