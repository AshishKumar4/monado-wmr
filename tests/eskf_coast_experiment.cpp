// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Characterize the controller's INERTIAL coast drift on real recorded data, through the real
 *        ESKF, headless — AND gate it. The user-facing worry is "controllers drift away then snap back
 *        when out of view". This tool measures BOTH halves of that, decomposed by how long the
 *        controller was out of view:
 *
 *        1. DRIFT — at each incoming optical pose, read the filter's prediction for that instant
 *           BEFORE folding the pose, and compare to the (plausible) optical pose. The error is the
 *           dead-reckon coast error over the *actual* gap since the last fold; binning by that real
 *           gap avoids the confound of imposing fixed windows on a stream that already has natural
 *           dropouts. This is the same quantity as the live-telemetry imu_vs_optical_drift study, but
 *           driven reproducibly through the production filter so it is an ablation platform: rebuild
 *           the ESKF, re-run, compare the curve.
 *
 *        2. FREEZE / SNAP — past OPTICAL_FREEZE_NS (500 ms) the filter intentionally stops
 *           dead-reckoning and freezes the reported position (clears POSITION_TRACKED), so the
 *           controller never flies off; the cost is a "snap" when optical returns equal to how far the
 *           controller actually moved while frozen. We report the frozen fraction per bin and the
 *           RE-ENTRY SNAP it produces — the max single-frame jump in the filter's *reported* position
 *           across a coast->fold transition — so the freeze horizon is a measured trade-off, not a guess.
 *
 *        CORPUS + COAST HORIZONS. A single recorded session's natural gap distribution is dominated by
 *        the 20-50 ms optical cadence: its long-gap bins hold a handful of samples, far too few to gate
 *        (they were printed and ignored, so the out-of-view regime the user actually feels — "the
 *        controller disappears, holds, then snaps" — was unguarded). The run is therefore a matrix of
 *        LEGS: every fixture in the corpus x every coast horizon. A horizon > 0 withholds the optical
 *        stream for that long every recover_s, so the filter must dead-reckon a real controller through
 *        a real out-of-view window and is then measured against the real optical pose that ends it.
 *        Nothing is synthesised: the IMU, the motion and the truth pose are all recorded hardware; only
 *        the choice of which real poses to hide is ours. Each leg runs its own filter instance; the
 *        samples pool into the same gap bins (a bin is defined by the ACTUAL gap, so natural and
 *        withheld coasts of the same length are the same measurement), which is what lifts the long
 *        bins past MIN_N_TO_GATE and makes them gate.
 *
 *        Bad OPTICAL truth (degenerate PnP fly-aways the live filter itself rejects) would masquerade
 *        as huge "drift"; we reject a pose as untrustworthy truth when it implies a controller speed
 *        above MAX_SPEED_M_S over the gap, mirroring the filter's own plausibility gate, and never
 *        compare against it. Robust percentiles (median/p95) further bound any residual outliers.
 *
 *        GATE: with --check <baseline.json> the per-bin metrics are emitted machine-readable and
 *        compared against the checked-in tests/coast_baseline.json; the process EXITS NONZERO if a
 *        guarded metric regresses past its tolerance, so a change that worsens out-of-view drift /
 *        flips / the re-entry snap cannot land unnoticed. The replay is deterministic (synchronous
 *        per-pose fold), so the gate is exact, not statistical. If the fixture is absent the process
 *        exits 77 (ctest SKIP) with a LOUD reason — never a silent pass. --update (re)writes the
 *        baseline from the current tree.
 *
 * Usage: eskf_coast_experiment <file.replay>... [--withhold S]... [--recover S=2.0]
 *          [--intrinsics M_g[9] T_a[9]] [--check baseline.json | --update baseline.json]
 *
 * Every .replay is a leg source and every --withhold a coast horizon in seconds (repeat either; with no
 * --withhold only the natural gaps are measured). --recover is the optical-visible interval between two
 * imposed coast windows.
 *
 * --intrinsics seeds the optically-derived IMU intrinsics (gyro M_g then accel T_a, 18 row-major
 * doubles) through the SAME kalman_fusion_set_imu_intrinsics the driver's cache loader uses, so an
 * identity-vs-computed A/B exercises the production correction path. Omitted = identity (uncorrected).
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "xrt/xrt_defines.h"
#include "xrt/xrt_tracking.h"
#include "tracking/t_tracker_kalman_fusion_c.h"

#include <cjson/cJSON.h>

#include "replay_fixture.hpp"

namespace {

constexpr double MAX_SPEED_M_S = 4.0;     // plausible controller speed ceiling; bad-truth rejector
constexpr double SPEED_MARGIN_M = 0.30;   // slack so legit fast flicks at short gaps aren't rejected
constexpr double BOOTSTRAP_S = 3.0;       // ignore samples until the filter has locked
constexpr double FLIP_DEG = 45.0;         // an orientation error past this is a flip, not drift
// A coast->fold transition counts toward the re-entry snap only past this gap; below it the per-frame
// reported-position step is just the normal optical cadence, not a "snap back from out-of-view".
constexpr double SNAP_GAP_S = 0.10;

// ctest SKIP exit code (paired with SKIP_RETURN_CODE 77 on the registered test). Distinct from the
// hard-error codes (1 = load failure, 2 = bad args) so an absent fixture is never a silent pass.
constexpr int EXIT_SKIP = 77;

double
quat_angle_deg(const float a[4], const float b[4])
{
	double d = std::fabs((double)a[0] * b[0] + (double)a[1] * b[1] + (double)a[2] * b[2] + (double)a[3] * b[3]);
	d = std::min(1.0, std::max(0.0, d));
	return 2.0 * std::acos(d) * 180.0 / M_PI;
}

struct Sample
{
	double gap_s;
	double pos_err_m;
	double ori_err_deg;
	bool frozen; // filter had cleared POSITION_TRACKED (past freeze horizon) at this instant
};

double
pct(std::vector<double> v, double q)
{
	if (v.empty())
		return 0.0;
	std::sort(v.begin(), v.end());
	return v[std::min(v.size() - 1, (size_t)(q * v.size()))];
}

// One gap bin's reduced metrics. Distances in cm, angles in deg; flip/frozen as % of the bin.
// Position drift is measured on NON-FLIPPED samples only (ori err <= FLIP_DEG): drift against a
// flipped "truth" is not coast drift, it is the optical mirror-flip showing up in the truth stream,
// which the separate flip_pct already captures. n is the count of non-flipped (drift) samples; n_all
// is the full bin count (the flip_pct/frozen_pct denominator).
struct BinMetrics
{
	const char *label;
	int n;     // non-flipped (drift-valid) samples
	int n_all; // all samples in the bin (flip/frozen denominator)
	double frozen_pct;
	double pos_med_cm, pos_p95_cm, pos_max_cm;
	double ori_med_deg, ori_p95_deg;
	double flip_pct;
};

// Gap bin edges (ms); the last bin is open-ended. Kept in lock-step with the JSON keys below so the
// baseline is self-describing and a renamed/reordered bin can't silently mis-compare.
const double EDGES_MS[] = {20, 50, 100, 200, 350, 500, 1000, 2000, 1e12};
const char *LABELS[] = {"  0- 20ms", " 20- 50ms", " 50-100ms", "100-200ms", "200-350ms",
                        "350-500ms", "0.5-1.0s ", "1.0-2.0s ", "  >2.0s  "};
const char *KEYS[] = {"0-20ms", "20-50ms", "50-100ms", "100-200ms", "200-350ms",
                      "350-500ms", "0.5-1.0s", "1.0-2.0s", ">2.0s"};
constexpr int NB = (int)(sizeof(EDGES_MS) / sizeof(EDGES_MS[0]));

std::vector<BinMetrics>
reduce_bins(const std::vector<Sample> &samples)
{
	std::vector<BinMetrics> out;
	double lo = 0;
	for (int b = 0; b < NB; b++) {
		std::vector<double> pe; // position drift on NON-FLIPPED samples only
		std::vector<double> oe; // orientation error over ALL samples in the bin
		int n_all = 0, nfroz = 0, nflip = 0;
		for (const Sample &sm : samples) {
			const double g_ms = sm.gap_s * 1e3;
			if (g_ms > lo && g_ms <= EDGES_MS[b]) {
				n_all++;
				oe.push_back(sm.ori_err_deg);
				if (sm.frozen)
					nfroz++;
				if (sm.ori_err_deg > FLIP_DEG)
					nflip++;
				else
					pe.push_back(sm.pos_err_m); // clean coast drift (truth not flipped)
			}
		}
		lo = EDGES_MS[b];
		if (n_all == 0)
			continue;
		double pmax = 0;
		for (double v : pe)
			pmax = std::max(pmax, v);
		BinMetrics m;
		m.label = LABELS[b];
		m.n = (int)pe.size();
		m.n_all = n_all;
		m.frozen_pct = 100.0 * nfroz / n_all;
		m.pos_med_cm = pct(pe, 0.5) * 100; // 0 if a bin is entirely flipped (no clean sample)
		m.pos_p95_cm = pct(pe, 0.95) * 100;
		m.pos_max_cm = pmax * 100;
		m.ori_med_deg = pct(oe, 0.5);
		m.ori_p95_deg = pct(oe, 0.95);
		m.flip_pct = 100.0 * nflip / n_all;
		out.push_back(m);
	}
	return out;
}

// Map a label back to its stable JSON key (label/key share index).
const char *
key_for_label(const char *label)
{
	for (int b = 0; b < NB; b++)
		if (LABELS[b] == label)
			return KEYS[b];
	return label;
}

//! One (fixture, coast horizon) leg: what it replayed and what it contributed to the pool.
struct LegResult
{
	std::string fixture;
	double withhold_s;
	int folded, rejected_truth, withheld;
	size_t nsamples;
	double snap_max_m;
};

void
print_table(bool have_intrinsics, const std::vector<LegResult> &legs, size_t nsamples,
            double reentry_snap_max_m, const std::vector<BinMetrics> &bins)
{
	printf("# corpus: %zu legs  intrinsics=%s  pooled samples=%zu\n", legs.size(),
	       have_intrinsics ? "on" : "identity", nsamples);
	printf("# %-22s %8s %7s %8s %9s %8s %8s\n", "leg", "withhold", "folded", "samples", "bad-truth",
	       "withheld", "snap cm");
	for (const LegResult &l : legs) {
		printf("# %-22s %7.2fs %7d %8zu %9d %8d %8.1f\n", l.fixture.c_str(), l.withhold_s, l.folded,
		       l.nsamples, l.rejected_truth, l.withheld, l.snap_max_m * 100);
	}
	printf("# re-entry snap (max reported-pos jump across a coast->fold, gap>%.0fms) = %.1f cm\n",
	       SNAP_GAP_S * 1e3, reentry_snap_max_m * 100);
	printf("# gap         all  drift  frozen%%  clean pos drift (cm)      ori err (deg)\n");
	printf("# %-10s %4s %5s %7s   med    p95    max     med    p95   flip%%\n", "", "n", "n", "");
	for (const BinMetrics &m : bins) {
		printf("  %-10s %4d %5d %6.0f%%  %5.1f %6.1f %6.1f   %5.1f %6.1f   %5.1f%%\n", m.label, m.n_all,
		       m.n, m.frozen_pct, m.pos_med_cm, m.pos_p95_cm, m.pos_max_cm, m.ori_med_deg, m.ori_p95_deg,
		       m.flip_pct);
	}
}

// ---- baseline JSON (cJSON) ----------------------------------------------------------------------
// Shape: { "_note": "...", "reentry_snap_max_cm": X, "bins": { "<key>": {n, frozen_pct, pos_med_cm,
// pos_p95_cm, pos_max_cm, ori_med_deg, ori_p95_deg, flip_pct}, ... } }. Generated with --update on the
// real head-pose capture so it reflects current healthy behavior.

cJSON *
metrics_to_json(double reentry_snap_max_m, const std::vector<BinMetrics> &bins)
{
	cJSON *root = cJSON_CreateObject();
	cJSON_AddStringToObject(root, "_note",
	                        "edited & maintained by Claude, presented as-is. "
	                        "Regenerate: eskf_coast_experiment <fixture.replay> --update <baseline.json>");
	cJSON_AddNumberToObject(root, "reentry_snap_max_cm", reentry_snap_max_m * 100);
	cJSON *jb = cJSON_AddObjectToObject(root, "bins");
	for (const BinMetrics &m : bins) {
		cJSON *o = cJSON_AddObjectToObject(jb, key_for_label(m.label));
		cJSON_AddNumberToObject(o, "n", m.n);
		cJSON_AddNumberToObject(o, "n_all", m.n_all);
		cJSON_AddNumberToObject(o, "frozen_pct", m.frozen_pct);
		cJSON_AddNumberToObject(o, "pos_med_cm", m.pos_med_cm);
		cJSON_AddNumberToObject(o, "pos_p95_cm", m.pos_p95_cm);
		cJSON_AddNumberToObject(o, "pos_max_cm", m.pos_max_cm);
		cJSON_AddNumberToObject(o, "ori_med_deg", m.ori_med_deg);
		cJSON_AddNumberToObject(o, "ori_p95_deg", m.ori_p95_deg);
		cJSON_AddNumberToObject(o, "flip_pct", m.flip_pct);
	}
	return root;
}

bool
write_baseline(const std::string &path, double reentry_snap_max_m, const std::vector<BinMetrics> &bins)
{
	cJSON *root = metrics_to_json(reentry_snap_max_m, bins);
	char *text = cJSON_Print(root);
	cJSON_Delete(root);
	if (!text)
		return false;
	FILE *f = std::fopen(path.c_str(), "wb");
	bool ok = f && std::fputs(text, f) >= 0 && std::fputc('\n', f) != EOF;
	if (f)
		std::fclose(f);
	cJSON_free(text);
	return ok;
}

cJSON *
read_json_file(const std::string &path)
{
	FILE *f = std::fopen(path.c_str(), "rb");
	if (!f)
		return nullptr;
	std::fseek(f, 0, SEEK_END);
	long n = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	std::string buf;
	if (n > 0) {
		buf.resize((size_t)n);
		if (std::fread(&buf[0], 1, (size_t)n, f) != (size_t)n) {
			std::fclose(f);
			return nullptr;
		}
	}
	std::fclose(f);
	return cJSON_Parse(buf.c_str());
}

double
jget(const cJSON *o, const char *key, bool *found)
{
	const cJSON *v = cJSON_GetObjectItemCaseSensitive(o, key);
	*found = cJSON_IsNumber(v);
	return *found ? v->valuedouble : 0.0;
}

// One guarded metric: name, current value, baseline value, the tolerance the metric may rise by, and
// whether a min-sample-count guard suppressed it (sparse bins are reported, not gated, to stay non-flaky).
struct Guard
{
	std::string where;  // e.g. "bin 100-200ms"
	std::string metric; // e.g. "pos_p95_cm"
	double cur, base, tol;
	bool gated;       // false => reported only (bin too sparse to gate reliably)
	bool hard = false; // unconditional failure (bin vanished / gating coverage lost); tol does not apply
};

// Tolerances. The replay is deterministic so these only need to absorb legitimate filter improvements
// (which lower drift/flips/snap) and tiny numerical jitter — they are one-sided "may not RISE by".
// Position tolerances scale with the metric (a long-coast bin legitimately drifts more); flips/snap
// are absolute. A bin below MIN_N_TO_GATE is reported but not gated: a one- or two-sample percentile
// says nothing about the tracker, and a drift median over 2 samples swings wildly when a flip
// reclassifies one of them. The corpus x coast-horizon matrix exists to keep every bin above that line
// — and losing that coverage is itself a gated failure (see compare()), so the "too sparse to gate"
// escape can never quietly reopen.
constexpr int MIN_N_TO_GATE = 20;
constexpr double POS_REL_TOL = 0.50;     // a guarded pos metric may rise up to 50% over baseline
constexpr double POS_ABS_TOL_CM = 3.0;   // ...or 3 cm, whichever is larger (covers small-baseline bins)
constexpr double FLIP_ABS_TOL_PCT = 8.0; // a bin's flip% may rise up to 8 absolute points
constexpr double FROZEN_ABS_TOL_PCT = 10.0;
constexpr double SNAP_ABS_TOL_CM = 8.0;  // the global re-entry snap may rise up to 8 cm

void
guard_pos(std::vector<Guard> &g, const std::string &where, const char *metric, double cur, double base,
          int n)
{
	double tol = std::max(POS_ABS_TOL_CM, POS_REL_TOL * base);
	g.push_back({where, metric, cur, base, tol, n >= MIN_N_TO_GATE});
}

// Compare measured bins against a loaded baseline. Returns true if all GATED guards are within tol.
bool
compare(const std::vector<BinMetrics> &bins, double reentry_snap_max_m, const cJSON *baseline,
        std::vector<Guard> &guards)
{
	const cJSON *jbins = cJSON_GetObjectItemCaseSensitive(baseline, "bins");
	if (!cJSON_IsObject(jbins)) {
		fprintf(stderr, "baseline has no \"bins\" object\n");
		return false;
	}
	bool ok = true;
	for (const BinMetrics &m : bins) {
		const char *key = key_for_label(m.label);
		const cJSON *jb = cJSON_GetObjectItemCaseSensitive(jbins, key);
		if (!cJSON_IsObject(jb)) {
			// A bin present now but absent from the baseline: report, don't fail (the capture's
			// natural gap distribution can leave a bin empty at baseline-gen time).
			guards.push_back({std::string("bin ") + key, "(no baseline)", m.pos_p95_cm, 0, 0, false});
			continue;
		}
		bool f;
		const std::string where = std::string("bin ") + key;
		// Gating coverage may not shrink. The replay is deterministic, so a bin that carried enough
		// samples to gate when the baseline was pinned and no longer does has lost a guard — that is a
		// regression to report, never a reason to stop guarding it.
		const double base_n = jget(jb, "n", &f);
		const double base_n_all = jget(jb, "n_all", &f);
		if (base_n >= MIN_N_TO_GATE && m.n < MIN_N_TO_GATE) {
			guards.push_back({where, "(drift coverage)", (double)m.n, base_n, 0, true, true});
		}
		if (base_n_all >= MIN_N_TO_GATE && m.n_all < MIN_N_TO_GATE) {
			guards.push_back({where, "(bin coverage)", (double)m.n_all, base_n_all, 0, true, true});
		}
		// Drift metrics gate on the clean (non-flipped) sample count; flip/frozen on the full count.
		guard_pos(guards, where, "pos_med_cm", m.pos_med_cm, jget(jb, "pos_med_cm", &f), m.n);
		guard_pos(guards, where, "pos_p95_cm", m.pos_p95_cm, jget(jb, "pos_p95_cm", &f), m.n);
		guard_pos(guards, where, "pos_max_cm", m.pos_max_cm, jget(jb, "pos_max_cm", &f), m.n);
		guards.push_back({where, "flip_pct", m.flip_pct, jget(jb, "flip_pct", &f), FLIP_ABS_TOL_PCT,
		                  m.n_all >= MIN_N_TO_GATE});
		guards.push_back({where, "frozen_pct", m.frozen_pct, jget(jb, "frozen_pct", &f),
		                  FROZEN_ABS_TOL_PCT, m.n_all >= MIN_N_TO_GATE});
	}
	// A baseline bin with NO samples in the current run is a regression, never a free pass: the
	// replay is deterministic, so a vanished bin means the coast-gap/fold distribution itself
	// changed (e.g. the filter stopped reporting valid positions in that regime). A legitimate
	// re-distribution goes through --update, visibly.
	for (const cJSON *jb = jbins->child; jb != NULL; jb = jb->next) {
		bool present = false;
		for (const BinMetrics &m : bins) {
			if (strcmp(key_for_label(m.label), jb->string) == 0) {
				present = true;
				break;
			}
		}
		if (!present) {
			bool f;
			guards.push_back({std::string("bin ") + jb->string, "(vanished)", 0.0,
			                  jget(jb, "n_all", &f), 0.0, true, true});
		}
	}
	bool snap_found;
	double snap_base = jget(baseline, "reentry_snap_max_cm", &snap_found);
	guards.push_back({"global", "reentry_snap_max_cm", reentry_snap_max_m * 100, snap_base,
	                  SNAP_ABS_TOL_CM, snap_found});
	for (Guard &gd : guards) {
		if (gd.gated && (gd.hard || gd.cur - gd.base > gd.tol))
			ok = false;
	}
	return ok;
}

void
print_guard_table(const std::vector<Guard> &guards)
{
	printf("%-16s %-14s %9s %9s %9s  %s\n", "where", "metric", "current", "baseline", "tol", "status");
	printf("%s\n", std::string(72, '-').c_str());
	for (const Guard &g : guards) {
		const bool fail = g.gated && (g.hard || g.cur - g.base > g.tol);
		const char *status = !g.gated ? "report" : (fail ? "FAIL" : "ok");
		printf("%-16s %-14s %9.2f %9.2f %9.2f  %s%s\n", g.where.c_str(), g.metric.c_str(), g.cur, g.base,
		       g.tol, status, fail ? "  <== REGRESSION" : "");
	}
}

//! Replay ONE leg: a fixture driven through a fresh filter with @p withhold_ns of optical hidden every
//! @p recover_ns. Appends its coast samples to @p samples and returns what the leg replayed. Every leg
//! is independent (own filter, own bootstrap), so a corpus is just a loop.
LegResult
run_leg(const g2replay::Dataset &ds, const std::string &fixture, int64_t withhold_ns, int64_t recover_ns,
        const double *mg, const double *ta, std::vector<Sample> &samples)
{
	LegResult leg{fixture, withhold_ns * 1e-9, 0, 0, 0, 0, 0.0};
	const int64_t t0 = ds.imu.front().t_ns;
	const int64_t bootstrap_ns = t0 + (int64_t)(BOOTSTRAP_S * 1e9);
	const size_t samples_before = samples.size();

	struct KalmanFusionInterfaceWrapper *kf = kalman_fusion_create();
	if (mg != nullptr) {
		kalman_fusion_set_imu_intrinsics(kf, mg, ta); // production cache-load path; identity = no-op
	}

	int64_t last_fold_ns = 0;           // last optical actually folded (gap reference)
	float last_fold_pos[3] = {0, 0, 0}; // its position, for the bad-truth speed gate
	bool have_fold = false;

	// Re-entry snap: the visible jump in the filter's REPORTED position across a coast->fold. We hold
	// the last reported position; when a fold ends a gap > SNAP_GAP_S we read the prediction again right
	// after process_pose and take the largest such single-frame jump.
	bool have_reported = false;
	float last_reported_pos[3] = {0, 0, 0};

	// Artificial-withhold bookkeeping (only when withhold_ns > 0).
	bool in_coast = false;
	int64_t coast_end = 0, next_coast = t0 + recover_ns;

	size_t ip = 0;
	for (const g2replay::Imu &s : ds.imu) {
		while (ip < ds.pose.size() && ds.pose[ip].t_ns <= s.t_ns) {
			const g2replay::Pose &p = ds.pose[ip];

			// Imposed coast window: pretend the controller left view for this long.
			if (withhold_ns > 0) {
				if (!in_coast && p.t_ns >= next_coast) {
					in_coast = true;
					coast_end = p.t_ns + withhold_ns;
				}
				if (in_coast && p.t_ns < coast_end) {
					leg.withheld++;
					ip++;
					continue;
				}
				if (in_coast) {
					in_coast = false;
					next_coast = p.t_ns + recover_ns;
				}
			}

			const float tp[3] = {p.px, p.py, p.pz};
			const float tq[4] = {p.qx, p.qy, p.qz, p.qw};

			// Measure the prediction for this instant BEFORE folding p. (Withheld / bad-truth poses
			// have already been skipped above, so every pose reaching here is folded.)
			double gap_s = 0;
			if (have_fold && p.t_ns > bootstrap_ns) {
				gap_s = (p.t_ns - last_fold_ns) * 1e-9;
				const double moved = std::sqrt((tp[0] - last_fold_pos[0]) * (tp[0] - last_fold_pos[0]) +
				                               (tp[1] - last_fold_pos[1]) * (tp[1] - last_fold_pos[1]) +
				                               (tp[2] - last_fold_pos[2]) * (tp[2] - last_fold_pos[2]));
				const bool plausible_truth = moved <= MAX_SPEED_M_S * gap_s + SPEED_MARGIN_M;
				if (!plausible_truth) {
					leg.rejected_truth++; // degenerate optical; don't fold, don't compare
					ip++;
					continue;
				}
				struct xrt_space_relation rel = {};
				kalman_fusion_get_prediction(kf, p.t_ns, &rel, nullptr);
				if (rel.relation_flags & XRT_SPACE_RELATION_POSITION_VALID_BIT) {
					const float pr[3] = {rel.pose.position.x, rel.pose.position.y, rel.pose.position.z};
					const float pq[4] = {rel.pose.orientation.x, rel.pose.orientation.y,
					                     rel.pose.orientation.z, rel.pose.orientation.w};
					Sample sm;
					sm.gap_s = gap_s;
					sm.pos_err_m = std::sqrt((pr[0] - tp[0]) * (pr[0] - tp[0]) + (pr[1] - tp[1]) * (pr[1] - tp[1]) +
					                         (pr[2] - tp[2]) * (pr[2] - tp[2]));
					sm.ori_err_deg = quat_angle_deg(pq, tq);
					sm.frozen = !(rel.relation_flags & XRT_SPACE_RELATION_POSITION_TRACKED_BIT);
					samples.push_back(sm);
				}
			}

			struct xrt_pose_sample ps = {};
			ps.pose.position = {p.px, p.py, p.pz};
			ps.pose.orientation = {p.qx, p.qy, p.qz, p.qw};
			ps.timestamp_ns = p.t_ns;
			kalman_fusion_process_pose(kf, &ps, nullptr, nullptr, 15, nullptr);

			// Reported-position step across this fold (the user-visible motion). A step after a real
			// coast (gap > SNAP_GAP_S) is the re-entry SNAP; track its max.
			struct xrt_space_relation after = {};
			kalman_fusion_get_prediction(kf, p.t_ns, &after, nullptr);
			if (after.relation_flags & XRT_SPACE_RELATION_POSITION_VALID_BIT) {
				const float rp[3] = {after.pose.position.x, after.pose.position.y, after.pose.position.z};
				if (have_reported && have_fold && p.t_ns > bootstrap_ns && gap_s > SNAP_GAP_S) {
					const double jump = std::sqrt((rp[0] - last_reported_pos[0]) * (rp[0] - last_reported_pos[0]) +
					                              (rp[1] - last_reported_pos[1]) * (rp[1] - last_reported_pos[1]) +
					                              (rp[2] - last_reported_pos[2]) * (rp[2] - last_reported_pos[2]));
					leg.snap_max_m = std::max(leg.snap_max_m, jump);
				}
				last_reported_pos[0] = rp[0];
				last_reported_pos[1] = rp[1];
				last_reported_pos[2] = rp[2];
				have_reported = true;
			}

			last_fold_ns = p.t_ns;
			last_fold_pos[0] = tp[0];
			last_fold_pos[1] = tp[1];
			last_fold_pos[2] = tp[2];
			have_fold = true;
			leg.folded++;
			ip++;
		}
		struct xrt_imu_sample is = {};
		is.timestamp_ns = s.t_ns;
		is.accel_m_s2 = {s.ax, s.ay, s.az};
		is.gyro_rad_secs = {s.gx, s.gy, s.gz};
		kalman_fusion_process_imu_data(kf, &is, nullptr, nullptr);
	}
	kalman_fusion_destroy(kf);
	leg.nsamples = samples.size() - samples_before;
	return leg;
}

} // namespace

int
main(int argc, char **argv)
{
	// Parse the option flags first so the fixture paths may sit anywhere on the command line.
	double mg[9], ta[9];
	bool have_intrinsics = false;
	std::string check_path, update_path;
	std::vector<std::string> fixtures;
	std::vector<double> withholds_s; // coast horizons; empty => natural gaps only
	double recover_s = 2.0;
	for (int i = 1; i < argc; i++) {
		const std::string a = argv[i];
		if (a == "--intrinsics") {
			if (i + 18 >= argc) {
				fprintf(stderr, "--intrinsics needs 18 doubles (M_g[9] T_a[9])\n");
				return 2;
			}
			for (int k = 0; k < 9; k++) {
				mg[k] = atof(argv[i + 1 + k]);
				ta[k] = atof(argv[i + 10 + k]);
			}
			have_intrinsics = true;
			i += 18;
		} else if (a == "--withhold" || a == "--recover" || a == "--check" || a == "--update") {
			if (i + 1 >= argc) {
				fprintf(stderr, "%s needs an argument\n", a.c_str());
				return 2;
			}
			const char *v = argv[++i];
			if (a == "--withhold") {
				withholds_s.push_back(atof(v));
			} else if (a == "--recover") {
				recover_s = atof(v);
			} else if (a == "--check") {
				check_path = v;
			} else {
				update_path = v;
			}
		} else {
			fixtures.push_back(a);
		}
	}
	if (fixtures.empty()) {
		fprintf(stderr,
		        "usage: %s <file.replay>... [--withhold S]... [--recover S=2.0] "
		        "[--intrinsics M_g[9] T_a[9]] [--check baseline.json | --update baseline.json]\n",
		        argv[0]);
		return 2;
	}
	if (!check_path.empty() && !update_path.empty()) {
		fprintf(stderr, "--check and --update are mutually exclusive\n");
		return 2;
	}
	if (withholds_s.empty()) {
		withholds_s.push_back(0.0); // natural gap distribution only
	}

	// Load the whole corpus up front: a missing fixture must stop the run LOUDLY (a silently smaller
	// corpus would thin the long-gap bins back below the gating threshold, which is the exact failure
	// this tool exists to prevent).
	std::vector<g2replay::Dataset> corpus(fixtures.size());
	for (size_t i = 0; i < fixtures.size(); i++) {
		if (!g2replay::load(fixtures[i], corpus[i])) {
			// LOUD skip — the registered ctest's SKIP_RETURN_CODE turns this into a visible ctest SKIP,
			// never a silent pass. A capture-less checkout therefore reports "not run", with the why.
			fprintf(stderr,
			        "SKIP: coast fixture absent or unreadable (%s) -- the coast gate did not run. Generate "
			        "it with tools/telemetry/make_replay_fixture.py from a real capture.\n",
			        fixtures[i].c_str());
			return EXIT_SKIP;
		}
		if (corpus[i].imu.empty() || corpus[i].pose.empty()) {
			fprintf(stderr, "SKIP: %s has no imu (%zu) or pose (%zu) -- coast gate did not run\n",
			        fixtures[i].c_str(), corpus[i].imu.size(), corpus[i].pose.size());
			return EXIT_SKIP;
		}
	}

	if (have_intrinsics) {
		struct KalmanFusionInterfaceWrapper *probe = kalman_fusion_create();
		kalman_fusion_set_imu_intrinsics(probe, mg, ta);
		double rmg[9], rta[9];
		const bool applied = kalman_fusion_get_imu_intrinsics(probe, rmg, rta);
		printf("# intrinsics applied=%s  M_g=[%.4f %.4f %.4f; %.4f %.4f %.4f; %.4f %.4f %.4f]\n",
		       applied ? "yes" : "REJECTED(identity)", rmg[0], rmg[1], rmg[2], rmg[3], rmg[4], rmg[5],
		       rmg[6], rmg[7], rmg[8]);
		kalman_fusion_destroy(probe);
	}

	std::vector<Sample> samples;
	std::vector<LegResult> legs;
	double reentry_snap_max_m = 0.0;
	const int64_t recover_ns = (int64_t)(recover_s * 1e9);
	for (size_t i = 0; i < corpus.size(); i++) {
		for (double w : withholds_s) {
			const LegResult leg = run_leg(corpus[i], corpus[i].name, (int64_t)(w * 1e9), recover_ns,
			                              have_intrinsics ? mg : nullptr, ta, samples);
			reentry_snap_max_m = std::max(reentry_snap_max_m, leg.snap_max_m);
			legs.push_back(leg);
		}
	}

	const std::vector<BinMetrics> bins = reduce_bins(samples);
	print_table(have_intrinsics, legs, samples.size(), reentry_snap_max_m, bins);

	if (!update_path.empty()) {
		if (!write_baseline(update_path, reentry_snap_max_m, bins)) {
			fprintf(stderr, "ERROR: failed to write baseline %s\n", update_path.c_str());
			return 1;
		}
		printf("\nwrote baseline -> %s\n", update_path.c_str());
		return 0;
	}

	if (!check_path.empty()) {
		cJSON *baseline = read_json_file(check_path);
		if (!baseline) {
			fprintf(stderr,
			        "SKIP: no coast baseline at %s -- create it with --update; coast gate did not run\n",
			        check_path.c_str());
			return EXIT_SKIP;
		}
		if (samples.empty()) {
			// The fixture loaded but the filter produced ZERO gateable coast samples (e.g. it
			// regressed to never setting POSITION_VALID). An empty run must never PASS.
			fprintf(stderr,
			        "RESULT: FAIL -- zero coast samples collected; the gate has no evidence to pass on\n");
			cJSON_Delete(baseline);
			return 1;
		}
		std::vector<Guard> guards;
		const bool ok = compare(bins, reentry_snap_max_m, baseline, guards);
		cJSON_Delete(baseline);
		printf("\n=== coast regression check (%zu fixtures x %zu coast horizons) ===\n", fixtures.size(),
		       withholds_s.size());
		print_guard_table(guards);
		printf("\ntolerance: pos may rise <= max(%.0f%%, %.0fcm); flip%% <= +%.0f; frozen%% <= +%.0f; "
		       "snap <= +%.0fcm; sparse bins (n<%d) reported not gated\n",
		       POS_REL_TOL * 100, POS_ABS_TOL_CM, FLIP_ABS_TOL_PCT, FROZEN_ABS_TOL_PCT, SNAP_ABS_TOL_CM,
		       MIN_N_TO_GATE);
		size_t n_gated = 0;
		for (const Guard &gd : guards) {
			n_gated += gd.gated ? 1 : 0;
		}
		printf("coverage: %zu of %zu guards gated", n_gated, guards.size());
		for (const BinMetrics &m : bins) {
			if (m.n < MIN_N_TO_GATE || m.n_all < MIN_N_TO_GATE) {
				printf("; %s UNGATED (n=%d n_all=%d)", key_for_label(m.label), m.n, m.n_all);
			}
		}
		printf("\n");
		if (n_gated == 0) {
			printf("\nRESULT: FAIL -- no guard was gateable (zero gated evidence); refusing to pass\n");
			return 1;
		}
		if (!ok) {
			printf("\nRESULT: FAIL -- a guarded coast metric regressed beyond tolerance\n");
			return 1;
		}
		printf("\nRESULT: PASS\n");
	}
	return 0;
}
