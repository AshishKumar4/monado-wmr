// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Tests for the world re-anchor glide (u_world_reanchor, the B2 head resnap guard).
 *
 * The policy was designed and validated offline against the recorded head streams of capture
 * 20260703-202811 (all 54 merged resnap events; worst: 819 mm + 59.3 deg in one 66.8 ms sample)
 * by the reference simulator results/b2-resnap-design-20260704/sim/sim_core.py. These tests pin
 * the C port against that simulator ELEMENT-WISE on the same recorded streams (fixture
 * data/b2_reanchor_head_streams.bin, float32-quantized inputs + f64 reference outputs, generated
 * by sim/make_cpp_fixture.py), and assert the design's hard properties directly:
 *
 *  - clean-segment BIT-IDENTITY: every zero-excess frame presents the raw pose exactly;
 *  - zero absorption outside +-0.5 s neighborhoods of the 54 recorded events;
 *  - worst-event presented step bounded (raw 819 mm / 59.3 deg -> <= 298 mm / 6.84 deg at the
 *    30 Hz gt rate, <= 111 mm / 3.9 deg at the 60-90 Hz driver rate — the pinned sim numbers);
 *  - structural convergence of every absorption episode (<= 0.8 s at still-head caps);
 *  - the repeated-jump degenerate case stays bounded by the sanity caps and converges.
 *
 * A production-configuration pass (head speed from the tracked state absent => 0, the
 * conservative cap) re-asserts the hard properties independent of the simulator's
 * finite-difference speed input (which the design flagged as snap-contaminated).
 */
#include "catch_amalgamated.hpp"

#include "util/u_world_reanchor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint64_t FIXTURE_MAGIC = 0x3158494652573242ULL; // "B2WRFIX1" LE
constexpr int ROW_F64 = 22;

struct Row
{
	double t;
	double p[3];
	double q[4]; // xyzw, float32-exact values
	double gyro_int_deg;
	double gyro_dps;
	double speed;
	// reference simulator outputs
	double ref_abs_ang;
	double ref_abs_pos;
	double ref_dq_ang;
	double ref_dp_norm;
	double ref_pres_p[3];
	double ref_pres_q[4];
};

struct Stream
{
	std::string name;
	std::vector<Row> rows;
};

struct Fixture
{
	double t0 = 0.0;
	double worst_event_t = 0.0;
	std::vector<double> events;
	std::vector<Stream> streams;
	bool file_present = false; //!< the fixture file opened (distinguishes missing from corrupt)
	bool loaded = false;
};

Fixture
load_fixture()
{
	Fixture fx;
	const std::string path = std::string(G2_TEST_DATA_DIR) + "/b2_reanchor_head_streams.bin";
	FILE *f = fopen(path.c_str(), "rb");
	if (f == nullptr) {
		return fx;
	}
	fx.file_present = true;
	auto rd = [&](void *dst, size_t sz) { return fread(dst, 1, sz, f) == sz; };
	uint64_t magic = 0;
	uint32_t n_events = 0;
	uint32_t n_streams = 0;
	bool ok = rd(&magic, 8) && magic == FIXTURE_MAGIC && rd(&fx.t0, 8) && rd(&fx.worst_event_t, 8) &&
	          rd(&n_events, 4);
	if (ok) {
		fx.events.resize(n_events);
		ok = rd(fx.events.data(), 8 * n_events) && rd(&n_streams, 4);
	}
	for (uint32_t s = 0; ok && s < n_streams; s++) {
		char name[9] = {0};
		uint32_t n = 0;
		ok = rd(name, 8) && rd(&n, 4);
		if (!ok) {
			break;
		}
		Stream st;
		st.name = name;
		st.rows.resize(n);
		std::vector<double> buf(ROW_F64);
		for (uint32_t i = 0; ok && i < n; i++) {
			ok = rd(buf.data(), 8 * ROW_F64);
			Row &r = st.rows[i];
			r.t = buf[0];
			memcpy(r.p, &buf[1], 3 * 8);
			memcpy(r.q, &buf[4], 4 * 8);
			r.gyro_int_deg = buf[8];
			r.gyro_dps = buf[9];
			r.speed = buf[10];
			r.ref_abs_ang = buf[11];
			r.ref_abs_pos = buf[12];
			r.ref_dq_ang = buf[13];
			r.ref_dp_norm = buf[14];
			memcpy(r.ref_pres_p, &buf[15], 3 * 8);
			memcpy(r.ref_pres_q, &buf[18], 4 * 8);
		}
		fx.streams.push_back(std::move(st));
	}
	fclose(f);
	fx.loaded = ok && !fx.streams.empty();
	return fx;
}

struct xrt_pose
row_pose(const Row &r)
{
	struct xrt_pose p;
	p.position.x = (float)r.p[0];
	p.position.y = (float)r.p[1];
	p.position.z = (float)r.p[2];
	p.orientation.x = (float)r.q[0];
	p.orientation.y = (float)r.q[1];
	p.orientation.z = (float)r.q[2];
	p.orientation.w = (float)r.q[3];
	return p;
}

struct PortOut
{
	std::vector<double> abs_ang, abs_pos, dq_ang, dp_norm;
	std::vector<struct xrt_pose> presented;
};

//! Drive the C port over a fixture stream exactly like the reference simulator's main loop:
//! per new sample, update (detect + absorb + decay) then present. @p use_sim_speed selects the
//! simulator's recorded speed input (parity legs) vs 0 (the conservative production floor).
PortOut
run_port(const Stream &st, bool use_sim_speed)
{
	PortOut out;
	const size_t n = st.rows.size();
	out.abs_ang.assign(n, 0.0);
	out.abs_pos.assign(n, 0.0);
	out.dq_ang.assign(n, 0.0);
	out.dp_norm.assign(n, 0.0);
	out.presented.resize(n);

	struct u_world_reanchor wr;
	u_world_reanchor_init(&wr);
	struct xrt_pose prev = row_pose(st.rows[0]);
	u_world_reanchor_apply(&wr, &prev, &out.presented[0]);
	for (size_t i = 1; i < n; i++) {
		const Row &r = st.rows[i];
		struct xrt_pose cur = row_pose(r);
		const double dt = r.t - st.rows[i - 1].t;
		u_world_reanchor_update(&wr, &u_world_reanchor_default_params, &prev, &cur, dt, r.gyro_int_deg,
		                        r.gyro_dps, use_sim_speed ? r.speed : 0.0, &out.abs_ang[i], &out.abs_pos[i]);
		u_world_reanchor_get_magnitude(&wr, &out.dq_ang[i], &out.dp_norm[i]);
		u_world_reanchor_apply(&wr, &cur, &out.presented[i]);
		prev = cur;
	}
	return out;
}

double
quat_step_deg(const struct xrt_pose &a, const struct xrt_pose &b)
{
	const double d = fabs((double)a.orientation.x * b.orientation.x + (double)a.orientation.y * b.orientation.y +
	                      (double)a.orientation.z * b.orientation.z + (double)a.orientation.w * b.orientation.w);
	return 2.0 * acos(fmin(1.0, d)) * 180.0 / M_PI;
}

double
pos_step_m(const struct xrt_pose &a, const struct xrt_pose &b)
{
	const double dx = (double)b.position.x - a.position.x;
	const double dy = (double)b.position.y - a.position.y;
	const double dz = (double)b.position.z - a.position.z;
	return sqrt(dx * dx + dy * dy + dz * dz);
}

bool
near_any_event(const Fixture &fx, double t, double radius_s)
{
	for (double ev : fx.events) {
		if (fabs(t - ev) <= radius_s) {
			return true;
		}
	}
	return false;
}

//! The design's hard properties, asserted on a port run of one stream. Worst-event bounds are
//! per-stream (pinned sim numbers). Purity is skipped for the synthetic stream (its injected
//! second jump is deliberately not in the recorded event table).
void
check_hard_properties(const Fixture &fx,
                      const Stream &st,
                      const PortOut &po,
                      double worst_pos_bound_m,
                      double worst_ang_bound_deg,
                      bool check_purity,
                      double conv_bound_s)
{
	const size_t n = st.rows.size();

	// Clean-segment BIT-IDENTITY: delta-inactive frames present the raw pose exactly.
	size_t clean = 0;
	for (size_t i = 0; i < n; i++) {
		if (po.dq_ang[i] == 0.0 && po.dp_norm[i] == 0.0) {
			clean++;
			const struct xrt_pose raw = row_pose(st.rows[i]);
			REQUIRE(memcmp(&po.presented[i], &raw, sizeof(raw)) == 0);
		}
	}
	INFO(st.name << ": " << clean << " of " << n << " frames clean (bit-identical)");
	REQUIRE(clean > n / 2); // the guard must be idle for the vast majority of the session

	// Zero absorption outside +-0.5 s neighborhoods of the 54 recorded events.
	if (check_purity) {
		for (size_t i = 0; i < n; i++) {
			if (po.abs_ang[i] > 0.0 || po.abs_pos[i] > 0.0) {
				INFO(st.name << " absorbing frame at t_rel " << st.rows[i].t - fx.t0);
				REQUIRE(near_any_event(fx, st.rows[i].t, 0.5));
			}
		}
	}

	// Worst-event presented step bounded by the pinned simulator numbers.
	double worst_pos = 0.0;
	double worst_ang = 0.0;
	for (size_t i = 1; i < n; i++) {
		if (fabs(st.rows[i].t - fx.worst_event_t) <= 0.05) {
			worst_pos = fmax(worst_pos, pos_step_m(po.presented[i - 1], po.presented[i]));
			worst_ang = fmax(worst_ang, quat_step_deg(po.presented[i - 1], po.presented[i]));
		}
	}
	INFO(st.name << " worst-event presented step " << worst_pos * 1e3 << " mm / " << worst_ang << " deg");
	REQUIRE(worst_pos <= worst_pos_bound_m);
	REQUIRE(worst_ang <= worst_ang_bound_deg);

	// Structural convergence: every absorption episode reaches (<0.5 deg, <5 mm) within the
	// bound of its last absorption (episodes truncated by stream end must already be inside).
	size_t i = 0;
	while (i < n) {
		if (po.dq_ang[i] == 0.0 && po.dp_norm[i] == 0.0) {
			i++;
			continue;
		}
		size_t end = i;
		while (end < n && (po.dq_ang[end] > 0.0 || po.dp_norm[end] > 0.0)) {
			end++;
		}
		size_t last_abs = i;
		for (size_t k = i; k < end; k++) {
			if (po.abs_ang[k] > 0.0 || po.abs_pos[k] > 0.0) {
				last_abs = k;
			}
		}
		size_t conv = last_abs;
		while (conv < n && !(po.dq_ang[conv] < 0.5 && po.dp_norm[conv] < 0.005)) {
			conv++;
		}
		if (conv < n) {
			INFO(st.name << " episode at t_rel " << st.rows[i].t - fx.t0);
			REQUIRE(st.rows[conv].t - st.rows[last_abs].t <= conv_bound_s);
		} else {
			// Truncated at stream end: must already be within the convergence threshold.
			REQUIRE(po.dq_ang[n - 1] < 0.5);
			REQUIRE(po.dp_norm[n - 1] < 0.005);
		}
		i = end;
	}
}

} // namespace

TEST_CASE("world_reanchor_basics")
{
	const struct u_world_reanchor_params *prm = &u_world_reanchor_default_params;
	struct u_world_reanchor wr;
	u_world_reanchor_init(&wr);

	struct xrt_pose a = {{0.f, 0.f, 0.f, 1.f}, {1.0f, 1.5f, -0.5f}};

	SECTION("identity passthrough is an exact copy")
	{
		struct xrt_pose out;
		u_world_reanchor_apply(&wr, &a, &out);
		REQUIRE(memcmp(&out, &a, sizeof(a)) == 0);
	}

	SECTION("sub-envelope steps never touch the delta")
	{
		struct xrt_pose b = a;
		b.position.x += 0.004f; // under the 8 mm floor
		REQUIRE(!u_world_reanchor_update(&wr, prm, &a, &b, 0.033, 0.0, 0.0, 0.0, nullptr, nullptr));
		REQUIRE(!wr.active);
		struct xrt_pose out;
		u_world_reanchor_apply(&wr, &b, &out);
		REQUIRE(memcmp(&out, &b, sizeof(b)) == 0);
	}

	SECTION("a still-head jump absorbs the excess and decays to identity")
	{
		struct xrt_pose b = a;
		b.position.y += 0.5f; // 500 mm step, envelope max(2.5*0.033, 8mm) = 82.5 mm
		double aa = 0.0;
		double ap = 0.0;
		REQUIRE(u_world_reanchor_update(&wr, prm, &a, &b, 0.033, 0.0, 0.0, 0.0, &aa, &ap));
		REQUIRE(ap == Catch::Approx(0.5 - 2.5 * 0.033).margin(1e-9));
		REQUIRE(aa == 0.0);
		double mag_ang = 0.0;
		double mag_pos = 0.0;
		u_world_reanchor_get_magnitude(&wr, &mag_ang, &mag_pos);
		REQUIRE(mag_pos > 0.3);
		// presented = raw + delta: the presented step is bounded by envelope + glide, not 500 mm.
		struct xrt_pose out;
		u_world_reanchor_apply(&wr, &b, &out);
		REQUIRE(fabs((double)out.position.y - a.position.y) < 0.15);
		// Decay to identity over further still samples.
		struct xrt_pose prev = b;
		for (int i = 0; i < 40; i++) {
			struct xrt_pose cur = prev; // still head
			u_world_reanchor_update(&wr, prm, &prev, &cur, 0.033, 0.0, 0.0, 0.0, nullptr, nullptr);
			prev = cur;
		}
		REQUIRE(!wr.active);
		u_world_reanchor_apply(&wr, &b, &out);
		REQUIRE(memcmp(&out, &b, sizeof(b)) == 0);
	}

	SECTION("the sanity cap bounds the delta")
	{
		struct xrt_pose b = a;
		b.position.z += 3.0f; // 3 m, beyond the 1.5 m cap
		u_world_reanchor_update(&wr, prm, &a, &b, 0.033, 0.0, 0.0, 0.0, nullptr, nullptr);
		double mag_pos = 0.0;
		u_world_reanchor_get_magnitude(&wr, nullptr, &mag_pos);
		REQUIRE(mag_pos <= 1.5 + 1e-9);
	}

	SECTION("step_to_world_delta maps the pre-step head onto the post-step head excess")
	{
		struct u_world_reanchor_step st;
		struct xrt_pose b = a;
		// 90 deg yaw about world-up + a 0.4 m translation, still head (all of it is excess-ish).
		b.orientation = {0.f, (float)sin(M_PI / 4), 0.f, (float)cos(M_PI / 4)};
		b.position.x += 0.4f;
		REQUIRE(u_world_reanchor_compute_excess(prm, &a, &b, 0.033, 0.0, &st));
		REQUIRE(st.exc_ang_deg == Catch::Approx(90.0 - 0.35).margin(1e-6));
		struct xrt_pose delta;
		u_world_reanchor_step_to_world_delta(&st, &a.position, &delta);
		// x' = delta.q * x + delta.p applied at the pivot == pivot + unit_dvec * exc_pos.
		struct xrt_vec3 pv = a.position;
		const double qx = delta.orientation.x, qy = delta.orientation.y, qz = delta.orientation.z,
		             qw = delta.orientation.w;
		const double tx = 2.0 * (qy * pv.z - qz * pv.y);
		const double ty = 2.0 * (qz * pv.x - qx * pv.z);
		const double tz = 2.0 * (qx * pv.y - qy * pv.x);
		const double rx = pv.x + qw * tx + (qy * tz - qz * ty) + delta.position.x;
		const double ry = pv.y + qw * ty + (qz * tx - qx * tz) + delta.position.y;
		const double rz = pv.z + qw * tz + (qx * ty - qy * tx) + delta.position.z;
		REQUIRE(rx == Catch::Approx(pv.x + st.unit_dvec[0] * st.exc_pos_m).margin(1e-5));
		REQUIRE(ry == Catch::Approx(pv.y + st.unit_dvec[1] * st.exc_pos_m).margin(1e-5));
		REQUIRE(rz == Catch::Approx(pv.z + st.unit_dvec[2] * st.exc_pos_m).margin(1e-5));
	}
}

TEST_CASE("world_reanchor_recorded_streams")
{
	Fixture fx = load_fixture();
	if (!fx.file_present) {
		SKIP("fixture data/b2_reanchor_head_streams.bin missing; recorded-stream validation skipped");
	}
	// Present but corrupt/truncated must FAIL, never skip: a bad fixture regen would otherwise
	// silently green the whole recorded-stream/parity battery.
	INFO("fixture data/b2_reanchor_head_streams.bin present but corrupt/truncated");
	REQUIRE(fx.loaded);
	REQUIRE(fx.events.size() == 54);
	REQUIRE(fx.streams.size() == 4);

	// Pinned worst-event bounds (the reference simulator's numbers on these streams at the
	// production parameters; raw step 819 mm / 59.3 deg).
	auto bounds_pos = [](const std::string &name) { return name == "hp" ? 0.1114 : 0.2985; };
	auto bounds_ang = [](const std::string &name) { return name == "hp" ? 3.90 : 6.85; };

	for (const Stream &st : fx.streams) {
		DYNAMIC_SECTION("simulator parity + hard properties: " << st.name)
		{
			PortOut po = run_port(st, /*use_sim_speed=*/true);
			const size_t n = st.rows.size();

			// Element-wise parity with the reference simulator (identical float32
			// inputs, identical formulas => double-precision agreement).
			for (size_t i = 0; i < n; i++) {
				const Row &r = st.rows[i];
				REQUIRE((po.abs_ang[i] > 0.0) == (r.ref_abs_ang > 0.0));
				REQUIRE((po.abs_pos[i] > 0.0) == (r.ref_abs_pos > 0.0));
				REQUIRE(po.abs_ang[i] == Catch::Approx(r.ref_abs_ang).margin(1e-9));
				REQUIRE(po.abs_pos[i] == Catch::Approx(r.ref_abs_pos).margin(1e-9));
				REQUIRE(po.dq_ang[i] == Catch::Approx(r.ref_dq_ang).margin(1e-9));
				REQUIRE(po.dp_norm[i] == Catch::Approx(r.ref_dp_norm).margin(1e-9));
				REQUIRE(fabs((double)po.presented[i].position.x - r.ref_pres_p[0]) < 1e-6);
				REQUIRE(fabs((double)po.presented[i].position.y - r.ref_pres_p[1]) < 1e-6);
				REQUIRE(fabs((double)po.presented[i].position.z - r.ref_pres_p[2]) < 1e-6);
				// Hemisphere-safe quaternion comparison.
				const double d[4] = {po.presented[i].orientation.x, po.presented[i].orientation.y,
				                     po.presented[i].orientation.z, po.presented[i].orientation.w};
				double dmax_pos = 0.0;
				double dmax_neg = 0.0;
				for (int k = 0; k < 4; k++) {
					dmax_pos = fmax(dmax_pos, fabs(d[k] - r.ref_pres_q[k]));
					dmax_neg = fmax(dmax_neg, fabs(d[k] + r.ref_pres_q[k]));
				}
				REQUIRE(fmin(dmax_pos, dmax_neg) < 1e-6);
			}

			check_hard_properties(fx, st, po, bounds_pos(st.name), bounds_ang(st.name),
			                      /*check_purity=*/st.name != "syn" && st.name != "skw",
			                      /*conv_bound_s=*/0.8);
		}

		DYNAMIC_SECTION("production configuration (IMU-derived speed floor): " << st.name)
		{
			// The simulator's finite-difference speed input is snap-contaminated (design
			// section 3 surprise 1); production keys the cap on the tracked-state velocity.
			// speed = 0 is its conservative floor — the hard bars must hold there too. The
			// design's 0.8 s convergence gate covers the RECORDED events at still-head caps;
			// the synthetic's stacked double jump (peak |delta_p| ~636 mm) at the bare
			// v0 = 1 m/s floor needs (636-100) mm / 1 m/s + tau*ln — ~0.93 s structurally, so
			// that one harsher-than-designed combination gets its own derived 1.0 s bound.
			PortOut po = run_port(st, /*use_sim_speed=*/false);
			const bool injected = st.name == "syn" || st.name == "skw";
			check_hard_properties(fx, st, po, bounds_pos(st.name), bounds_ang(st.name),
			                      /*check_purity=*/!injected,
			                      /*conv_bound_s=*/injected ? 1.0 : 0.8);
		}
	}

	// Repeated-jump degenerate case: the synthetic stream's second 59 deg / 0.8 m jump lands
	// 0.3 s into the worst event's glide; the composed delta must respect the sanity caps.
	const Stream &syn = fx.streams[2];
	REQUIRE(syn.name == "syn");
	PortOut po = run_port(syn, true);
	double peak_ang = 0.0;
	double peak_pos = 0.0;
	for (size_t i = 0; i < syn.rows.size(); i++) {
		peak_ang = fmax(peak_ang, po.dq_ang[i]);
		peak_pos = fmax(peak_pos, po.dp_norm[i]);
	}
	INFO("synthetic repeated-jump peak delta " << peak_ang << " deg / " << peak_pos * 1e3 << " mm");
	REQUIRE(peak_ang <= 90.0 + 1e-9);
	REQUIRE(peak_pos <= 1.5 + 1e-9);
	REQUIRE(peak_ang > 80.0); // the two 59 deg jumps really did stack (grazes the cap)

	// R2#1 commutator regression: the skew-axis stream's second 59 deg jump lands 0.3 s into
	// the worst event's glide about an axis perpendicular to the first. dq maps raw->presented,
	// so the excess must RIGHT-compose (dq_new = dq_old*R_exc^-1); the refuted left-composition
	// presented the commutator as a 30.3 deg single-frame step here, while correct composition
	// glides at 3.4 deg (reference simulator, both compositions, on this exact stream).
	const Stream &skw = fx.streams[3];
	REQUIRE(skw.name == "skw");
	PortOut pk = run_port(skw, true);
	double skew_step = 0.0;
	double skew_peak_ang = 0.0;
	for (size_t i = 1; i < skw.rows.size(); i++) {
		skew_peak_ang = fmax(skew_peak_ang, pk.dq_ang[i]);
		if (fabs(skw.rows[i].t - (fx.worst_event_t + 0.3)) <= 0.06) {
			skew_step = fmax(skew_step, quat_step_deg(pk.presented[i - 1], pk.presented[i]));
		}
	}
	INFO("skew second-event max presented frame-step " << skew_step << " deg, peak delta " << skew_peak_ang
	                                                   << " deg");
	REQUIRE(skew_peak_ang > 60.0); // the second jump really was absorbed mid-glide (overlap happened)
	REQUIRE(skew_step <= 8.0);     // no commutator jump: presented stays glide-smooth
}
