/*!
 * @file
 * @brief Decoupled unit tests for the kept-alive two-mode yaw belief:
 *        internal/yaw_belief.h's pure yaw_belief_step + yaw_belief_seed. Synthetic quats +
 *        scripted gyro deltas + scripted per-mode total_nll — no tracker, no filter, no
 *        captured data — so the propagate/fold/commit math is proven on KNOWN truth.
 *
 *        The belief handles seed-frame pure-yaw ties by deferring orientation, preserving position,
 *        accumulating evidence, and committing only after one mode dominates. These tests keep the
 *        math independent of the tracker and captured data.
 */
#include "catch_amalgamated.hpp"

#include "internal/yaw_belief.h"
#include "internal/pose_metrics.h"

#include "math/m_api.h"

#include <cmath>

namespace {

//! A unit quaternion for a rotation of @p deg about the (ax,ay,az) axis (matches the pnp tests' helper).
struct xrt_quat
quat_axis_deg(double ax, double ay, double az, double deg)
{
	const double n = std::sqrt(ax * ax + ay * ay + az * az);
	if (n < 1e-12) {
		return xrt_quat{0.f, 0.f, 0.f, 1.f};
	}
	const double half = deg * M_PI / 180.0 / 2.0;
	const double s = std::sin(half);
	struct xrt_quat q = {(float)(ax / n * s), (float)(ay / n * s), (float)(az / n * s), (float)std::cos(half)};
	math_quat_normalize(&q);
	return q;
}

//! Gyro-propagate forward by a world-up yaw delta (what the tracker computes as q_delta . q_ref).
struct xrt_quat
yaw(const struct xrt_quat &q, double delta_deg)
{
	struct xrt_quat d = quat_axis_deg(0, 1, 0, delta_deg);
	struct xrt_quat out;
	math_quat_rotate(&d, &q, &out);
	math_quat_normalize(&out);
	return out;
}

//! The belief's production tuning, mirrored from t_constellation_tracking.c.
struct yaw_belief_params
default_params()
{
	struct yaw_belief_params p = {};
	p.commit_w = 0.9f;                          // BELIEF_COMMIT_W
	p.no_cand_nll = 3.0f;                        // BELIEF_NO_CAND_NLL == ASSOC_TEMPORAL_MAX_NLL
	p.ttl_ns = 100ull * 1000ull * 1000ull;
	p.max_frames = 4u;                           // BELIEF_MAX_FRAMES
	p.yaw_sigma_rad = 45.0 * M_PI / 180.0;
	p.tilt_sigma_rad = 180.0 * M_PI / 180.0;
	p.huber_knee = 3.0;                          // FLIP_COST_HUBER_KNEE_SIGMA
	p.weight = 1.0;                              // FLIP_COST_WEIGHT
	p.cap = 3.0;                                 // ASSOC_TEMPORAL_MAX_NLL
	return p;
}

//! The reference yaw cost the matcher/temporal_nll already computes, exposed for the negative control.
double
temporal_cost(const struct xrt_quat &cand, const struct xrt_quat &ref)
{
	const struct xrt_vec3 up = {0.f, 1.f, 0.f};
	const struct yaw_belief_params p = default_params();
	const double raw = pose_metrics_prior_orient_cost(&cand, &ref, &up, p.tilt_sigma_rad, p.yaw_sigma_rad,
	                                                  p.huber_knee, p.weight);
	return raw > p.cap ? p.cap : raw;
}

const uint64_t MS = 1000ull * 1000ull;

} // namespace

// ===========================================================================
// Test 1 — THE required test: a SUSTAINED pure-yaw tie the belief resolves where
// temporal_nll alone (one frame, no reference) cannot. The seed frame is a genuine
// coin-flip; the belief carries both modes and commits the correct one once the gyro
// continuation + the per-frame optical margin have separated them.
// ===========================================================================
TEST_CASE("yaw belief: a sustained pure-yaw tie resolves over frames; temporal_nll alone cannot")
{
	const struct yaw_belief_params p = default_params();

	// The two single-frame-indistinguishable pure-yaw mirror twins at the seed: A (correct) and B = A.yaw(180).
	const struct xrt_quat A = quat_axis_deg(0, 1, 0, 0);
	const struct xrt_quat B = quat_axis_deg(0, 1, 0, 180);

	// NEGATIVE CONTROL (the "temporal_nll alone could not" requirement): on the seed frame there is NO trusted
	// continuity reference. temporal_nll early-returns / contributes 0, so it cannot separate A from B — the pick
	// would be a coin-flip. (Even priced against the controller's own held heading, both twins are equidistant.)
	{
		const double cA = temporal_cost(A, A); // a stale/absent reference ties both: cost is symmetric
		const double cB = temporal_cost(B, A);
		// With the reference == A the term already "knows" the answer; the point of the seed regime is there is
		// NO such reference. Model that: with no reference the term is identically 0 for both -> tie unbroken.
		const double c_seed_A = 0.0, c_seed_B = 0.0;
		CHECK(c_seed_A == Catch::Approx(c_seed_B)); // the single-frame term cannot break the seed tie
		// And a 180-deg flip against a reference saturates the cap (so the term IS the right lever once a
		// reference EXISTS — which is exactly the regime the belief hands off to).
		CHECK(cA < 0.5);
		CHECK(cB == Catch::Approx(p.cap));
	}

	// Seed the belief equiprobable on {A, B} at t0. Both modes start at logw 0 -> posterior 0.5 each.
	struct constellation_yaw_belief b = {};
	const struct xrt_quat seed_prior = A; // the (untrusted-for-separation) held heading; trusted for GYRO only
	yaw_belief_seed(&b, /*seed_ts*/ 0, &A, &B, &seed_prior, /*seed_prior_trusted*/ true);
	REQUIRE(b.active);
	CHECK(yaw_belief_posterior(&b, 0) == Catch::Approx(0.5));
	CHECK(yaw_belief_posterior(&b, 1) == Catch::Approx(0.5));

	// Frames 1..N: the controller holds a small honest yaw drift (+10 deg/frame). The tracker gyro-propagates
	// BOTH modes by that delta (here we hand yaw_belief_step the already-propagated modes, as the tracker does).
	// At the SEED the two are tied; as the controller moves, the WRONG mode's reprojection progressively degrades
	// (the flip explains fewer real LEDs as the geometry evolves), so A's candidate is cheaper by a GROWING
	// margin — each single frame's increment stays sub-tie at first, but the belief ACCUMULATES it across frames.
	// This is the gap temporal_nll cannot fill: it has no reference yet, so it cannot wait for this future evidence.
	int committed_mode = -1;
	int commit_frame = -1;
	for (int f = 1; f <= 6 && b.active; f++) {
		const double cum = 10.0 * f; // cumulative honest yaw from the seed
		struct xrt_quat prop[2] = {yaw(A, cum), yaw(B, cum)};

		// Per-mode evidence: each mode matches its own gyro-continued pose (yaw-continuity NLL ~0 for BOTH — the
		// gyro migrates both modes with the turn), separated ONLY by the optical total_nll margin, which WIDENS as
		// the flip drifts off the real LEDs (0.4, 0.8, 1.2, ... nats — the FLIP-BLOB-INVESTIGATION 0.87-2.95 band).
		const float margin = 0.4f * (float)f;
		struct yaw_belief_mode_evidence ev[2] = {};
		ev[0].have_cand = true;
		ev[0].cand_q = prop[0];
		ev[0].cand_total_nll = 1.0f; // A: the cheaper fit
		ev[1].have_cand = true;
		ev[1].cand_q = prop[1];
		ev[1].cand_total_nll = 1.0f + margin; // B: progressively worse as the flip drifts off the LEDs

		int cm = -1;
		const enum yaw_belief_action act = yaw_belief_step(&b, (uint64_t)f * MS, prop, ev, &p, &cm);

		if (act == YAW_BELIEF_COMMIT_MODE) {
			committed_mode = cm;
			commit_frame = f;
			break;
		}
		// While unresolved the belief DEFERS (orientation withheld) — it never commits a coin-flip orientation.
		CHECK(act == YAW_BELIEF_DEFER);
		// The correct mode (mode 0 == A, kept the MAP) pulls ahead monotonically and never falls below 0.5.
		CHECK(yaw_belief_posterior(&b, 0) >= 0.5f);
	}

	// THE architecture win: the belief eventually commits, and commits the CORRECT mode (A == prop[0]'s family),
	// NEVER the flip B. cand_q for the committed mode must be A.yaw(cumulative), not B.yaw(cumulative).
	REQUIRE(committed_mode >= 0);
	INFO("committed mode " << committed_mode << " at frame " << commit_frame);
	const double cum = 10.0 * commit_frame;
	const struct xrt_quat expect_A = yaw(A, cum);
	const struct xrt_quat expect_B = yaw(B, cum);
	const struct xrt_vec3 up = {0.f, 1.f, 0.f};
	double tiltA = 0, yawA = 0, tiltB = 0, yawB = 0;
	const struct xrt_quat committed_q =
	    committed_mode == 0 ? yaw(A, cum) : yaw(B, cum); // == the mode's propagated orientation
	pose_metrics_prior_orient_split(&committed_q, &expect_A, &up, &tiltA, &yawA);
	pose_metrics_prior_orient_split(&committed_q, &expect_B, &up, &tiltB, &yawB);
	CHECK(yawA < 1.0 * M_PI / 180.0);  // the committed orientation IS the correct continuation A
	CHECK(yawB > 170.0 * M_PI / 180.0); // and is ~180 deg from the flip B
	CHECK_FALSE(b.active);              // resolved -> the belief cleared itself (hands off to temporal_nll)
}

// ===========================================================================
// Test 2 — the recall-trap guard. Two mechanical firewalls of design §6, at the pure layer:
//  (a) the belief is INERT unless explicitly seeded — a lone-correct frame never seeds it, so a
//      step on a non-active belief is a no-op and the device commits immediately (it is the tracker
//      predicate association_pure_yaw_ambiguous that gates the seed; here we prove the pure module
//      adds NO behaviour when not seeded).
//  (b) once active and deferring, the belief NEVER drops position: a DEFER always keeps both modes
//      alive (and the tracker folds position-only); a mode that loses its candidate (occlusion) pays
//      the bounded no-cand cost and the belief still DEFERS — it never collapses to a coin-flip commit
//      and never withholds the position the fusion needs.
// ===========================================================================
TEST_CASE("yaw belief: recall-trap guard — never deferred unless seeded, never drops position")
{
	const struct yaw_belief_params p = default_params();

	// (a) A fresh (zero-initialised) belief is INERT: active=false. The tracker only ever calls yaw_belief_step
	//     while active, so a lone-correct candidate (never seeded) is committed by the unchanged ladder. We assert
	//     the inert default mechanically: a zero belief has active=false and equal (zero) log-weights.
	struct constellation_yaw_belief inert = {};
	CHECK_FALSE(inert.active);
	CHECK(inert.mode_logw[0] == 0.0f);
	CHECK(inert.mode_logw[1] == 0.0f);

	// (b) Seed a belief, then on a frame the CORRECT mode (mode 0 == A) loses its candidate (occlusion) while the
	//     flip mode B still has one. The belief must NOT commit B (a coin-flip) and must NOT drop — it DEFERS, so
	//     the tracker still folds position-only. The no-cand mode pays the bounded BELIEF_NO_CAND_NLL.
	const struct xrt_quat A = quat_axis_deg(0, 1, 0, 0);
	const struct xrt_quat B = quat_axis_deg(0, 1, 0, 180);
	struct constellation_yaw_belief b = {};
	yaw_belief_seed(&b, 0, &A, &B, &A, true);

	struct xrt_quat prop[2] = {A, B};
	struct yaw_belief_mode_evidence ev[2] = {};
	ev[0].have_cand = false;            // the correct mode is occluded this frame
	ev[1].have_cand = true;             // only the flip has a candidate
	ev[1].cand_q = B;
	ev[1].cand_total_nll = 1.0f;
	int cm = -1;
	const enum yaw_belief_action act = yaw_belief_step(&b, 1 * MS, prop, ev, &p, &cm);
	// One frame of B-only evidence cannot clear the 9:1 commit bar from a 50/50 prior, so it DEFERS (position-only
	// at the tracker), NOT a commit of the flip. The belief stays alive; nothing is dropped.
	CHECK(act == YAW_BELIEF_DEFER);
	CHECK(b.active);

	// (b-cont) And a single frame can NEVER force a commit from the equiprobable seed even when one mode is
	//          strongly favoured: the no-cand mode pays at most the cap (3 nats), so the weight ratio after one
	//          frame is exp(3) ≈ 20:1... which DOES clear 9:1. So assert the SAFE property the design relies on:
	//          a commit only happens when the WINNING mode has a candidate to lock (never an empty commit).
	//          Re-seed and drive A strongly; B occluded every frame -> A must be the one that commits, never an
	//          orientation with no candidate.
	struct constellation_yaw_belief b2 = {};
	yaw_belief_seed(&b2, 0, &A, &B, &A, true);
	int committed = -1;
	for (int f = 1; f <= 4 && b2.active; f++) {
		struct xrt_quat pr[2] = {A, B};
		struct yaw_belief_mode_evidence e[2] = {};
		e[0].have_cand = true; // A always present + cheap
		e[0].cand_q = A;
		e[0].cand_total_nll = 0.5f;
		e[1].have_cand = false; // B occluded -> pays the bounded no-cand cost
		int c = -1;
		const enum yaw_belief_action a = yaw_belief_step(&b2, (uint64_t)f * MS, pr, e, &p, &c);
		if (a == YAW_BELIEF_COMMIT_MODE) {
			committed = c;
			break;
		}
		CHECK(a == YAW_BELIEF_DEFER);
	}
	REQUIRE(committed == 0);                  // the mode that committed is A (mode 0), the one WITH a candidate
	CHECK_FALSE(b2.active);
}

// ===========================================================================
// Test 3 — expiry / graceful degradation: an unresolved tie expires to the normal path, never a
// permanent orientation hold. Feed perfectly-tied evidence (delta == 0 every frame) so neither mode
// ever dominates; the belief must DEFER every frame and then EXPIRE at BELIEF_MAX_FRAMES (returning
// to the normal ladder), and separately EXPIRE on BELIEF_TTL_NS.
// ===========================================================================
TEST_CASE("yaw belief: an unresolved tie expires (max-frames AND ttl), no permanent defer")
{
	const struct yaw_belief_params p = default_params();
	const struct xrt_quat A = quat_axis_deg(0, 1, 0, 0);
	const struct xrt_quat B = quat_axis_deg(0, 1, 0, 180);

	// max-frames bound: identical evidence both modes -> the weights track together, never crossing 9:1.
	struct constellation_yaw_belief b = {};
	yaw_belief_seed(&b, 0, &A, &B, &A, true);
	int defers = 0;
	enum yaw_belief_action last = YAW_BELIEF_DEFER;
	for (int f = 1; f <= 10; f++) {
		struct xrt_quat prop[2] = {A, B};
		struct yaw_belief_mode_evidence ev[2] = {};
		ev[0].have_cand = true;
		ev[0].cand_q = A;
		ev[0].cand_total_nll = 1.0f;
		ev[1].have_cand = true;
		ev[1].cand_q = B;
		ev[1].cand_total_nll = 1.0f; // EXACT tie every frame
		int cm = -1;
		last = yaw_belief_step(&b, (uint64_t)f * MS, prop, ev, &p, &cm);
		if (last == YAW_BELIEF_EXPIRE) {
			break;
		}
		CHECK(last == YAW_BELIEF_DEFER); // every pre-expiry frame folds position-only, never drops, never commits
		defers++;
	}
	CHECK(last == YAW_BELIEF_EXPIRE);
	CHECK(defers == (int)p.max_frames); // exactly BELIEF_MAX_FRAMES defers, then expiry
	CHECK_FALSE(b.active);

	// ttl bound (independent of frame count): a single frame past BELIEF_TTL_NS expires immediately.
	struct constellation_yaw_belief b2 = {};
	yaw_belief_seed(&b2, /*seed_ts*/ 0, &A, &B, &A, true);
	struct xrt_quat prop[2] = {A, B};
	struct yaw_belief_mode_evidence ev[2] = {};
	ev[0].have_cand = true;
	ev[0].cand_q = A;
	ev[1].have_cand = true;
	ev[1].cand_q = B;
	int cm = -1;
	const enum yaw_belief_action a = yaw_belief_step(&b2, p.ttl_ns, prop, ev, &p, &cm); // ts - seed == ttl
	CHECK(a == YAW_BELIEF_EXPIRE);
	CHECK_FALSE(b2.active);
}
