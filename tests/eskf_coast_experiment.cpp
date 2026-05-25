// Copyright 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Characterize the controller's INERTIAL coast drift on real recorded data, through the real
 *        ESKF, headless. The user-facing worry is "controllers drift away then snap back when out of
 *        view". This tool measures BOTH halves of that, decomposed by how long the controller was out
 *        of view:
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
 *           controller actually moved while frozen. We report the frozen fraction per bin and the snap
 *           it produces, so the freeze horizon is a measured trade-off, not a guess.
 *
 *        Optionally (withhold>0) it also imposes artificial coast windows to probe horizons longer
 *        than the natural gap distribution.
 *
 *        Bad OPTICAL truth (degenerate PnP fly-aways the live filter itself rejects) would masquerade
 *        as huge "drift"; we reject a pose as untrustworthy truth when it implies a controller speed
 *        above MAX_SPEED_M_S over the gap, mirroring the filter's own plausibility gate, and never
 *        compare against it. Robust percentiles (median/p95) further bound any residual outliers.
 *
 * Usage: eskf_coast_experiment <file.replay> [withhold_seconds=0] [recover_seconds=2.0]
 *          [--intrinsics M_g[9] T_a[9]]
 *
 * --intrinsics seeds the optically-derived IMU intrinsics (gyro M_g then accel T_a, 18 row-major
 * doubles) through the SAME kalman_fusion_set_imu_intrinsics the driver's cache loader uses, so an
 * identity-vs-computed A/B exercises the production correction path. Omitted = identity (uncorrected).
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "xrt/xrt_defines.h"
#include "xrt/xrt_tracking.h"
#include "tracking/t_tracker_kalman_fusion_c.h"

#include "replay_fixture.hpp"

namespace {

constexpr double MAX_SPEED_M_S = 4.0;     // plausible controller speed ceiling; bad-truth rejector
constexpr double SPEED_MARGIN_M = 0.30;   // slack so legit fast flicks at short gaps aren't rejected
constexpr double BOOTSTRAP_S = 3.0;       // ignore samples until the filter has locked

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

} // namespace

int
main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr,
		        "usage: %s <file.replay> [withhold_seconds=0] [recover_seconds=2.0] "
		        "[--intrinsics M_g[9] T_a[9]]\n",
		        argv[0]);
		return 2;
	}
	g2replay::Dataset ds;
	if (!g2replay::load(argv[1], ds)) {
		fprintf(stderr, "failed to load %s\n", argv[1]);
		return 1;
	}
	if (ds.imu.empty() || ds.pose.empty()) {
		fprintf(stderr, "replay has no imu (%zu) or pose (%zu)\n", ds.imu.size(), ds.pose.size());
		return 1;
	}

	// Optional intrinsics: gyro M_g[9] then accel T_a[9], row-major, applied through the production
	// cache-load path. Parsed first so it may sit anywhere after the replay path.
	double mg[9], ta[9];
	bool have_intrinsics = false;
	std::vector<char *> pos; // the positional args (replay, withhold, recover) once --intrinsics is removed
	pos.push_back(argv[0]);
	for (int i = 1; i < argc; i++) {
		if (std::string(argv[i]) == "--intrinsics") {
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
		} else {
			pos.push_back(argv[i]);
		}
	}
	const int npos = (int)pos.size();
	const double withhold_s = (npos > 2) ? atof(pos[2]) : 0.0;
	const int64_t withhold_ns = (int64_t)(withhold_s * 1e9);
	const int64_t recover_ns = (int64_t)((npos > 3 ? atof(pos[3]) : 2.0) * 1e9);
	const int64_t t0 = ds.imu.front().t_ns;
	const int64_t bootstrap_ns = t0 + (int64_t)(BOOTSTRAP_S * 1e9);

	struct KalmanFusionInterfaceWrapper *kf = kalman_fusion_create();
	if (have_intrinsics) {
		kalman_fusion_set_imu_intrinsics(kf, mg, ta); // production cache-load path; identity = no-op
		// Confirm the filter ACCEPTED them (passed the plausibility guard) and now holds a real
		// correction — the activation read-back the offline tool's output must satisfy.
		double rmg[9], rta[9];
		const bool applied = kalman_fusion_get_imu_intrinsics(kf, rmg, rta);
		printf("# intrinsics applied=%s  M_g=[%.4f %.4f %.4f; %.4f %.4f %.4f; %.4f %.4f %.4f]\n",
		       applied ? "yes" : "REJECTED(identity)", rmg[0], rmg[1], rmg[2], rmg[3], rmg[4], rmg[5],
		       rmg[6], rmg[7], rmg[8]);
	}

	std::vector<Sample> samples;
	int folded = 0, rejected_truth = 0, withheld = 0;
	int64_t last_fold_ns = 0;            // last optical actually folded (gap reference)
	float last_fold_pos[3] = {0, 0, 0};  // its position, for the bad-truth speed gate
	bool have_fold = false;

	// Artificial-withhold bookkeeping (only when withhold_s>0).
	bool in_coast = false;
	int64_t coast_end = 0, next_coast = t0 + recover_ns;

	size_t ip = 0;
	for (const g2replay::Imu &s : ds.imu) {
		while (ip < ds.pose.size() && ds.pose[ip].t_ns <= s.t_ns) {
			const g2replay::Pose &p = ds.pose[ip];

			// Optional artificial coast window: pretend the controller left view.
			if (withhold_ns > 0) {
				if (!in_coast && p.t_ns >= next_coast) {
					in_coast = true;
					coast_end = p.t_ns + withhold_ns;
				}
				if (in_coast && p.t_ns < coast_end) {
					withheld++;
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

			// Measure the prediction for this instant BEFORE folding p.
			if (have_fold && p.t_ns > bootstrap_ns) {
				const double gap_s = (p.t_ns - last_fold_ns) * 1e-9;
				const double moved = std::sqrt((tp[0] - last_fold_pos[0]) * (tp[0] - last_fold_pos[0]) +
				                               (tp[1] - last_fold_pos[1]) * (tp[1] - last_fold_pos[1]) +
				                               (tp[2] - last_fold_pos[2]) * (tp[2] - last_fold_pos[2]));
				const bool plausible_truth = moved <= MAX_SPEED_M_S * gap_s + SPEED_MARGIN_M;
				if (!plausible_truth) {
					rejected_truth++; // degenerate optical; don't fold, don't compare
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
			last_fold_ns = p.t_ns;
			last_fold_pos[0] = tp[0];
			last_fold_pos[1] = tp[1];
			last_fold_pos[2] = tp[2];
			have_fold = true;
			folded++;
			ip++;
		}
		struct xrt_imu_sample is = {};
		is.timestamp_ns = s.t_ns;
		is.accel_m_s2 = {s.ax, s.ay, s.az};
		is.gyro_rad_secs = {s.gx, s.gy, s.gz};
		kalman_fusion_process_imu_data(kf, &is, nullptr, nullptr);
	}
	kalman_fusion_destroy(kf);

	// Bin by actual gap. Edges in ms; last bin is open-ended.
	const double edges_ms[] = {20, 50, 100, 200, 350, 500, 1000, 2000, 1e12};
	const int nb = (int)(sizeof(edges_ms) / sizeof(edges_ms[0]));
	const char *labels[] = {"  0- 20ms", " 20- 50ms", " 50-100ms", "100-200ms", "200-350ms",
	                        "350-500ms", "0.5-1.0s ", "1.0-2.0s ", "  >2.0s  "};

	printf("# %s  intrinsics=%s  withhold=%.2fs  folded=%d  samples=%zu  rejected-bad-truth=%d  withheld=%d\n",
	       pos[1], have_intrinsics ? "on" : "identity", withhold_s, folded, samples.size(), rejected_truth,
	       withheld);
	printf("# gap          n  frozen%%   pos drift (cm)            ori err (deg)\n");
	printf("# %-10s %5s %7s   med    p95    max     med    p95   flips>45deg\n", "", "", "");
	double lo = 0;
	for (int b = 0; b < nb; b++) {
		std::vector<double> pe, oe;
		int nfroz = 0, nflip = 0;
		for (const Sample &sm : samples) {
			const double g_ms = sm.gap_s * 1e3;
			if (g_ms > lo && g_ms <= edges_ms[b]) {
				pe.push_back(sm.pos_err_m);
				oe.push_back(sm.ori_err_deg);
				if (sm.frozen)
					nfroz++;
				if (sm.ori_err_deg > 45.0)
					nflip++;
			}
		}
		lo = edges_ms[b];
		if (pe.empty())
			continue;
		double pmax = 0;
		for (double v : pe)
			pmax = std::max(pmax, v);
		printf("  %-10s %5zu %6.0f%%  %5.1f %6.1f %6.1f   %5.1f %6.1f   %d\n", labels[b], pe.size(),
		       100.0 * nfroz / pe.size(), pct(pe, 0.5) * 100, pct(pe, 0.95) * 100, pmax * 100, pct(oe, 0.5),
		       pct(oe, 0.95), nflip);
	}
	return 0;
}
