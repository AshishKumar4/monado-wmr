#include "catch_amalgamated.hpp"

#include "internal/l2_accept_gate.h"

#include <cmath>

namespace {

l2_accept_params
params()
{
	return {
	    6,
	    2,
	    12.0 * M_PI / 180.0,
	    2.0,
	    20.0,
	};
}

l2_accept_evidence
recoverable()
{
	return {
	    12,
	    2,
	    0.8,
	    14.0,
	    2.0 * M_PI / 180.0,
	    true,
	    true,
	    false,
	};
}

} // namespace

TEST_CASE("L2 accepts only rich multi-view tilt-consistent candidates")
{
	auto p = params();
	auto e = recoverable();
	REQUIRE(l2_accept_recoverable(&e, &p));

	e.distinct_view_count = 1;
	REQUIRE_FALSE(l2_accept_recoverable(&e, &p));

	e = recoverable();
	e.matched_count = 5;
	REQUIRE_FALSE(l2_accept_recoverable(&e, &p));

	e = recoverable();
	e.tilt_error_rad = 20.0 * M_PI / 180.0;
	REQUIRE_FALSE(l2_accept_recoverable(&e, &p));

	e = recoverable();
	e.reproj_per_led_px2 = 2.5;
	REQUIRE_FALSE(l2_accept_recoverable(&e, &p));

	e = recoverable();
	e.total_nll = 25.0;
	REQUIRE_FALSE(l2_accept_recoverable(&e, &p));
}

TEST_CASE("L2 stays additive and inert without trusted tilt")
{
	auto p = params();
	auto e = recoverable();
	e.already_lock_eligible = true;
	REQUIRE_FALSE(l2_accept_recoverable(&e, &p));

	e = recoverable();
	e.tilt_valid = false;
	REQUIRE_FALSE(l2_accept_recoverable(&e, &p));

	e = recoverable();
	e.pose_match_good = false;
	REQUIRE_FALSE(l2_accept_recoverable(&e, &p));
}

TEST_CASE("single-view prior gate rejects only high-prior-disagreement monocular poses")
{
	const single_view_prior_gate_params p{6.0, 0.8, 1.0};

	single_view_prior_gate_evidence e{1, 7.0, 0.1, true};
	REQUIRE(single_view_prior_gate_disagrees(&e, &p));

	e.prior_nll = 6.0;
	REQUIRE_FALSE(single_view_prior_gate_disagrees(&e, &p));

	e = {1, 1.1, 0.9, true};
	REQUIRE(single_view_prior_gate_disagrees(&e, &p));

	e = {1, 0.5, 1.2, true};
	REQUIRE_FALSE(single_view_prior_gate_disagrees(&e, &p));

	e = {2, 20.0, 2.0, true};
	REQUIRE_FALSE(single_view_prior_gate_disagrees(&e, &p));

	e = {1, 20.0, 2.0, false};
	REQUIRE_FALSE(single_view_prior_gate_disagrees(&e, &p));
}
