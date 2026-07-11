// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  L1 multi-cam single-cam-depth dispute verdict (the pure decision core).
 * @ingroup constellation
 *
 * A single-camera pose commit pins tilt + bearing but leaves RANGE (depth) ill-conditioned: a few
 * near-coplanar LEDs in one view fit at ~1px reprojection yet can sit decimetres off in depth. When a 2nd
 * rigidly-mounted camera co-sees the controller, the rig baseline triangulates the depth directly. This
 * header is the pure adjudication of the committed depth against that triangulation, decoupled from the
 * tracker's struct plumbing so the verdict ladder + its recall-monotone invariant are unit-testable on
 * synthetic geometry (see design-L1-multicam-depth.md §2.2 + §2.6).
 *
 * RECALL-MONOTONE BY CONSTRUCTION: the verdict can NEVER reject or refine on the ABSENCE of a 2nd view —
 * only on the PRESENCE of one that disputes the depth beyond the shared-covariance gate. A frame only the
 * committing camera sees returns L1V_AGREE and leaves the commit byte-identical to baseline (the dev2
 * single-cam-dwell guard). Asserted directly by tests_multicam_triangulate's L1 cases.
 */
#pragma once

#include <math.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

enum l1_verdict
{
	L1V_AGREE = 0,   //!< no usable 2nd view, the 2nd view confirms the depth (<=Nσ), or triangulation declined
	L1V_REFINE = 1,  //!< >=2 cams each >=4 LEDs dispute the depth, triangulation confident: re-fold position
	L1V_DISPUTE = 2, //!< a 2nd cam (2-3 LEDs) disputes the depth too weakly to re-solve: reject the commit
};

//! Inputs to the verdict, already reduced to geometry (no tracker structs). All distances in metres.
struct l1_depth_evidence
{
	double disp_m;             //!< ||triangulated_position - committed_position||
	double triangulated_std_m; //!< the triangulation's reported 1σ (0 if it declined; see tri_succeeded)
	double committed_std_m;    //!< the committed single-cam observation's 1σ (the position-only fold's own)
	int best_second_view_leds; //!< most device LEDs in any NON-committing camera (0 == single-view frame)
	int n_cams_ge_refine_leds; //!< cameras (incl. the committing one) carrying >= the REFINE per-cam LED bar
	bool tri_succeeded;        //!< the rig-baseline triangulation produced a position (>=2 LEDs, >=2 cams)
};

//! Tuning, all geometric: the dispute test is a shared-covariance Nσ test (not a fixed cm), the 2nd view
//! must carry enough LEDs to constrain depth, and REFINE only trusts a triangulation at/below a std floor.
struct l1_depth_params
{
	double dispute_sigma;      //!< Nσ of the COMBINED (triangulated + committed) std to call a depth dispute
	int min_dispute_leds;      //!< a 2nd camera below this LED count gives only a bearing -> cannot dispute
	int refine_per_cam_leds;   //!< per-camera LED bar; >= this in >=2 cams is strong enough to re-solve
	double refine_max_std_m;   //!< REFINE only when the triangulated 1σ is at/below this confident floor
};

/*!
 * Adjudicate a single-cam commit's depth against the rig-baseline triangulation.
 *
 * The ladder, in order (the FIRST escape that applies wins, so monotonicity is structural):
 *  1. No non-committing camera reaches @p min_dispute_leds         -> L1V_AGREE  (single-view; nothing disputes)
 *  2. The triangulation declined                                   -> L1V_AGREE  (no usable 2-view geometry)
 *  3. The depth difference is within @p dispute_sigma of the shared covariance -> L1V_AGREE (2nd view confirms)
 *  4. Otherwise it is a genuine depth dispute:
 *       >=2 cams each >= refine_per_cam_leds AND tri std <= floor   -> L1V_REFINE  (re-fold triangulated pos)
 *       else                                                        -> L1V_DISPUTE (reject; coast)
 */
static inline enum l1_verdict
l1_depth_verdict(const struct l1_depth_evidence *e, const struct l1_depth_params *p)
{
	if (e->best_second_view_leds < p->min_dispute_leds) {
		return L1V_AGREE; // recall-monotone escape hatch: the absence of a 2nd view never changes the commit
	}
	if (!e->tri_succeeded) {
		return L1V_AGREE;
	}
	const double gate = p->dispute_sigma * sqrt(e->triangulated_std_m * e->triangulated_std_m +
	                                             e->committed_std_m * e->committed_std_m);
	if (e->disp_m <= gate) {
		return L1V_AGREE; // the healthy multi-cam frame: the 2nd view confirms the committed depth
	}
	if (e->n_cams_ge_refine_leds >= 2 && e->triangulated_std_m <= p->refine_max_std_m) {
		return L1V_REFINE;
	}
	return L1V_DISPUTE;
}

#ifdef __cplusplus
}
#endif
