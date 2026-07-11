// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Per-LED multi-camera triangulation of a controller position. See multicam_triangulate.h.
 * @ingroup constellation
 */
#include "multicam_triangulate.h"

#include "math/m_api.h"
#include "math/m_vec3.h"

#include <cmath>
#include <math.h>
#include <string.h>

/* Tuning, all geometric / physical (no free magic numbers):
 *  - MIN_PARALLAX_SIN: a pair of rays whose included angle has |sin| below this is too near-parallel to
 *    triangulate a depth (the closest-point becomes ill-conditioned); such a LED is dropped. ~2 deg.
 *  - MAX_RAY_RESIDUAL_M: a triangulated LED whose mean point-to-ray distance exceeds this is a mislabel /
 *    inconsistent correspondence and is rejected. 2 cm is generous vs the ~mm geometric accuracy of two
 *    good rays at arm's length, while still catching a wrong cross-camera pairing (which lands decimetres
 *    off because the rays then don't intersect).
 *  - OUTLIER_MAD_K: per-LED controller-origin estimates are robustly combined; an estimate further than
 *    this many (scaled) MADs from the median is an outlier and excluded from the final average. 3.0 is the
 *    standard robust cutoff.
 *  - MIN_INLIER_SPREAD_STD_M: a floor on the reported position std so a tight cluster of few LEDs does not
 *    report unrealistic confidence (the lever-arm yaw error + extrinsic/centroid noise live here). */
static const double MIN_PARALLAX_SIN = 0.035;   // ~2 deg between the two most-separated rays
static const double MAX_RAY_RESIDUAL_M = 0.02;  // 2 cm mean point-to-ray distance
static const double OUTLIER_MAD_K = 3.0;
static const double MIN_INLIER_SPREAD_STD_M = 0.01; // 1 cm floor

namespace {

struct V3d
{
	double x, y, z;
};

V3d
sub(const V3d &a, const V3d &b)
{
	return {a.x - b.x, a.y - b.y, a.z - b.z};
}

double
dot(const V3d &a, const V3d &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

double
norm(const V3d &a)
{
	return sqrt(dot(a, a));
}

//! One camera's bearing ray to a labelled LED, in the OpenCV-world frame: origin (camera centre) + unit
//! direction. Carries its source (view, blob) so a surviving LED can register its cross-view evidence.
struct LedRay
{
	V3d origin;
	V3d dir; // unit
	int view_id;
	int blob_id;
};

//! Least-squares closest point to a set of >= 2 rays (the midpoint generalisation): minimise
//! Sum_i ||(I - d_i d_i^T)(x - o_i)||^2  =>  A x = b with A = Sum (I - d_i d_i^T), b = A-weighted origins.
//! Returns false if A is ill-conditioned (all rays near-parallel). Also returns the mean point-to-ray
//! distance of the solution (the triangulation residual) for the consistency gate.
bool
closest_point_to_rays(const LedRay *rays, int n, V3d &out_point, double &out_residual_m)
{
	double A[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
	double b[3] = {0, 0, 0};
	for (int i = 0; i < n; i++) {
		const V3d &d = rays[i].dir;
		// P_i = I - d d^T (projects onto the plane perpendicular to the ray).
		const double P[3][3] = {{1.0 - d.x * d.x, -d.x * d.y, -d.x * d.z},
		                        {-d.y * d.x, 1.0 - d.y * d.y, -d.y * d.z},
		                        {-d.z * d.x, -d.z * d.y, 1.0 - d.z * d.z}};
		const double o[3] = {rays[i].origin.x, rays[i].origin.y, rays[i].origin.z};
		for (int r = 0; r < 3; r++) {
			for (int c = 0; c < 3; c++) {
				A[r][c] += P[r][c];
				b[r] += P[r][c] * o[c];
			}
		}
	}
	// Solve A x = b (3x3, symmetric PSD) by the cofactor inverse; gate on the determinant for conditioning.
	const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
	                   A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
	                   A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
	if (!(fabs(det) > 1e-9)) {
		return false; // rays near-parallel / degenerate -> no usable depth
	}
	const double inv[3][3] = {
	    {(A[1][1] * A[2][2] - A[1][2] * A[2][1]) / det, (A[0][2] * A[2][1] - A[0][1] * A[2][2]) / det,
	     (A[0][1] * A[1][2] - A[0][2] * A[1][1]) / det},
	    {(A[1][2] * A[2][0] - A[1][0] * A[2][2]) / det, (A[0][0] * A[2][2] - A[0][2] * A[2][0]) / det,
	     (A[0][2] * A[1][0] - A[0][0] * A[1][2]) / det},
	    {(A[1][0] * A[2][1] - A[1][1] * A[2][0]) / det, (A[0][1] * A[2][0] - A[0][0] * A[2][1]) / det,
	     (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / det}};
	out_point = {inv[0][0] * b[0] + inv[0][1] * b[1] + inv[0][2] * b[2],
	             inv[1][0] * b[0] + inv[1][1] * b[1] + inv[1][2] * b[2],
	             inv[2][0] * b[0] + inv[2][1] * b[1] + inv[2][2] * b[2]};

	// Mean point-to-ray distance: |(x - o_i) - ((x - o_i).d_i) d_i|, averaged.
	double sum = 0.0;
	for (int i = 0; i < n; i++) {
		const V3d w = sub(out_point, rays[i].origin);
		const double along = dot(w, rays[i].dir);
		const V3d perp = {w.x - along * rays[i].dir.x, w.y - along * rays[i].dir.y, w.z - along * rays[i].dir.z};
		sum += norm(perp);
	}
	out_residual_m = sum / n;
	return true;
}

//! Largest |sin(angle)| between any pair of the rays — the triangulation parallax. Near 0 means all rays
//! are near-collinear (a single effective viewpoint), so the depth is unobservable however many cameras.
double
max_parallax_sin(const LedRay *rays, int n)
{
	double best = 0.0;
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			// |d_i x d_j| = |sin| for unit vectors.
			const double cx = rays[i].dir.y * rays[j].dir.z - rays[i].dir.z * rays[j].dir.y;
			const double cy = rays[i].dir.z * rays[j].dir.x - rays[i].dir.x * rays[j].dir.z;
			const double cz = rays[i].dir.x * rays[j].dir.y - rays[i].dir.y * rays[j].dir.x;
			const double s = sqrt(cx * cx + cy * cy + cz * cz);
			if (s > best) {
				best = s;
			}
		}
	}
	return best;
}

double
median_of(double *v, int n)
{
	// Small n: insertion sort in place, then take the middle (average of the two middles for even n).
	for (int i = 1; i < n; i++) {
		double key = v[i];
		int j = i - 1;
		while (j >= 0 && v[j] > key) {
			v[j + 1] = v[j];
			j--;
		}
		v[j + 1] = key;
	}
	return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

constexpr int MAX_TRI = MULTICAM_TRI_MAX_VIEWS * 64;

//! One LED triangulated from >= 2 cross-view rays: its model index, its world point, and the rays that
//! produced it (carried so the result can register the cross-view evidence). The two correspondence
//! front-ends (label-based and epipolar) both emit this; the shared aggregator consumes it.
struct TriLed
{
	int led_id;
	V3d world_pt;
	LedRay rays[MULTICAM_TRI_MAX_VIEWS];
	int n_rays;
};

//! Compute a blob's bearing ray in the world frame: undistort to a normalised camera bearing, rotate into
//! the world frame by the camera extrinsic (origin = camera centre). Identical undistort + frame as the PnP.
//! Returns false only for a degenerate (zero-length) direction.
bool
blob_world_ray(const struct multicam_tri_view *view, int blob_id, int view_id, LedRay &out)
{
	const struct blob *b = &view->blobs[blob_id];
	float nx = 0.f, ny = 0.f;
	t_camera_models_undistort(&view->calib->calib, b->x, b->y, &nx, &ny);
	const struct xrt_vec3 bearing_cam = {nx, ny, 1.0f};
	struct xrt_vec3 dir_world;
	math_quat_rotate_vec3(&view->P_world_cam.orientation, &bearing_cam, &dir_world);
	const double len = sqrt((double)dir_world.x * dir_world.x + (double)dir_world.y * dir_world.y +
	                        (double)dir_world.z * dir_world.z);
	if (!(len > 1e-9)) {
		return false;
	}
	out.origin = {(double)view->P_world_cam.position.x, (double)view->P_world_cam.position.y,
	              (double)view->P_world_cam.position.z};
	out.dir = {dir_world.x / len, dir_world.y / len, dir_world.z / len};
	out.view_id = view_id;
	out.blob_id = blob_id;
	return true;
}

//! Triangulate a set of >= 2 rays to a world point, applying the parallax + ray-residual consistency gates.
//! Returns false (no usable point) when the rays are near-parallel, ill-conditioned, or do not intersect
//! (a wrong cross-camera pairing lands decimetres off). Shared by both correspondence front-ends.
bool
triangulate_rays(const LedRay *rays, int n_rays, V3d &out_pt)
{
	if (n_rays < 2 || max_parallax_sin(rays, n_rays) < MIN_PARALLAX_SIN) {
		return false;
	}
	double residual = 0.0;
	if (!closest_point_to_rays(rays, n_rays, out_pt, residual)) {
		return false;
	}
	return residual <= MAX_RAY_RESIDUAL_M && std::isfinite(out_pt.x) && std::isfinite(out_pt.y) &&
	       std::isfinite(out_pt.z);
}

//! Shared consensus aggregator: map each triangulated LED to a controller-origin estimate (subtract its
//! model lever arm rotated by the prior orientation), then robustly average the per-LED estimates into ONE
//! controller position + a yaw-aware 1-sigma, requiring >= max(2,min_leds) mutually-agreeing LEDs spanning
//! >= 2 cameras. Fills @p out (already memset by the caller). This is the precision guard the public
//! contract promises: a single LED, a collapsed consensus, or a single-camera baseline all decline.
bool
aggregate_tri_leds(const TriLed *tri,
                   int n_tri,
                   const struct t_constellation_led_model *leds_model,
                   const struct xrt_quat *q_prior,
                   float prior_yaw_sigma_rad,
                   int min_leds,
                   struct multicam_tri_result *out)
{
	if (n_tri < (min_leds < 2 ? 2 : min_leds)) {
		return false; // one LED alone leaves the lever-arm yaw error unchecked and gives no agreement
	}
	V3d origin_est[MAX_TRI];
	double origin_lever_m[MAX_TRI];
	for (int i = 0; i < n_tri; i++) {
		// p_origin = led_world - R_prior * led_pos (CV-world<-CV-model).
		struct xrt_vec3 lever;
		math_quat_rotate_vec3(q_prior, &leds_model->leds[tri[i].led_id].pos, &lever);
		origin_est[i] = {tri[i].world_pt.x - lever.x, tri[i].world_pt.y - lever.y, tri[i].world_pt.z - lever.z};
		origin_lever_m[i] = (double)m_vec3_len(leds_model->leds[tri[i].led_id].pos);
	}

	// Robustly aggregate: per-axis median, drop estimates beyond OUTLIER_MAD_K scaled MADs (a mislabelled /
	// mismatched LED that passed its own ray-residual gate but disagrees with the consensus).
	double mx[MAX_TRI], my[MAX_TRI], mz[MAX_TRI];
	for (int i = 0; i < n_tri; i++) {
		mx[i] = origin_est[i].x;
		my[i] = origin_est[i].y;
		mz[i] = origin_est[i].z;
	}
	const V3d med = {median_of(mx, n_tri), median_of(my, n_tri), median_of(mz, n_tri)};
	double dists[MAX_TRI], dsort[MAX_TRI];
	for (int i = 0; i < n_tri; i++) {
		dists[i] = norm(sub(origin_est[i], med));
		dsort[i] = dists[i];
	}
	const double mad = median_of(dsort, n_tri);
	const double scale = (1.4826 * mad > 1e-4) ? 1.4826 * mad : 1e-4;

	V3d sum = {0, 0, 0};
	int n_in = 0;
	bool inlier[MAX_TRI];
	for (int i = 0; i < n_tri; i++) {
		inlier[i] = dists[i] <= OUTLIER_MAD_K * scale;
		if (inlier[i]) {
			sum.x += origin_est[i].x;
			sum.y += origin_est[i].y;
			sum.z += origin_est[i].z;
			n_in++;
		}
	}
	if (n_in < 2) {
		return false; // consensus collapsed -> not enough agreeing LEDs
	}
	const V3d mean = {sum.x / n_in, sum.y / n_in, sum.z / n_in};

	double var = 0.0;
	double lever_sum = 0.0;
	int n_views_mask = 0;
	bool view_seen[16] = {false};
	for (int i = 0; i < n_tri; i++) {
		if (!inlier[i]) {
			continue;
		}
		var += norm(sub(origin_est[i], mean)) * norm(sub(origin_est[i], mean));
		lever_sum += origin_lever_m[i];
		for (int r = 0; r < tri[i].n_rays; r++) {
			if (out->num_blob_refs < MAX_TRI) {
				out->blob_refs[out->num_blob_refs].view_id = tri[i].rays[r].view_id;
				out->blob_refs[out->num_blob_refs].blob_id = tri[i].rays[r].blob_id;
				out->blob_refs[out->num_blob_refs].led_id = tri[i].led_id;
				out->num_blob_refs++;
			}
			const int vid = tri[i].rays[r].view_id;
			if (vid >= 0 && vid < 16 && !view_seen[vid]) {
				view_seen[vid] = true;
				n_views_mask++;
			}
		}
	}
	var = var / n_in;

	// Honest, yaw-aware position std: the inter-LED spread (triangulation + per-LED correspondence noise) in
	// quadrature with the lever-arm yaw term (|mean lever| * yaw_sigma) — the dominant error the spread MISSES
	// when the visible LEDs share a lever direction (their common shift under a yaw-mis-rotated prior cancels
	// out of the spread). Tilt is gravity-anchored (driftless) so only yaw contributes.
	const double mean_lever = n_in > 0 ? lever_sum / n_in : 0.0;
	const double yaw_sigma = prior_yaw_sigma_rad > 0.f ? (double)prior_yaw_sigma_rad : 0.0;
	const double lever_yaw_std = mean_lever * yaw_sigma;
	double std_m = sqrt(var + lever_yaw_std * lever_yaw_std);
	if (std_m < MIN_INLIER_SPREAD_STD_M) {
		std_m = MIN_INLIER_SPREAD_STD_M;
	}
	if (n_views_mask < 2) {
		return false; // the surviving inlier LEDs collapsed onto a single camera -> no real baseline
	}

	out->position.x = (float)mean.x;
	out->position.y = (float)mean.y;
	out->position.z = (float)mean.z;
	out->position_std_m = (float)std_m;
	out->num_leds = n_in;
	out->num_views = n_views_mask;
	return true;
}

} // namespace

bool
multicam_triangulate_position(const struct multicam_tri_view *views,
                              int num_views,
                              const struct t_constellation_led_model *leds_model,
                              const struct xrt_quat *prior_orientation,
                              float prior_yaw_sigma_rad,
                              int min_leds,
                              struct multicam_tri_result *out_result)
{
	if (views == NULL || leds_model == NULL || prior_orientation == NULL || out_result == NULL ||
	    num_views < 2 || leds_model->num_leds <= 0) {
		return false;
	}
	memset(out_result, 0, sizeof(*out_result));
	if (min_leds < 1) {
		min_leds = 1;
	}

	struct xrt_quat q_prior = *prior_orientation;
	math_quat_normalize(&q_prior);

	// Correspondence front-end: collect one ray per view for each LED by its FRONT-END LABEL (device,
	// led_id), triangulate the co-seen LEDs, and hand the triangulated set to the shared aggregator. A blob's
	// label is its (device, led_id); blobs across cameras sharing the same label are the same physical LED.
	TriLed tri[MAX_TRI];
	int n_tri = 0;
	const int max_leds = leds_model->num_leds < 64 ? leds_model->num_leds : 64;
	for (int led = 0; led < max_leds && n_tri < MAX_TRI; led++) {
		LedRay rays[MULTICAM_TRI_MAX_VIEWS];
		int n_rays = 0;
		for (int vi = 0; vi < num_views && n_rays < MULTICAM_TRI_MAX_VIEWS; vi++) {
			const struct multicam_tri_view *view = &views[vi];
			if (view->blobs == NULL || view->num_blobs <= 0 || view->calib == NULL) {
				continue;
			}
			for (int bi = 0; bi < view->num_blobs; bi++) {
				const struct blob *b = &view->blobs[bi];
				if (LED_OBJECT_ID(b->led_id) != leds_model->id || LED_LOCAL_ID(b->led_id) != led) {
					continue;
				}
				if (blob_world_ray(view, bi, vi, rays[n_rays])) {
					n_rays++;
				}
				break; // one ray per view per LED (labels are deduped to one blob per LED per view)
			}
		}
		V3d led_world;
		if (!triangulate_rays(rays, n_rays, led_world)) {
			continue; // < 2 views, near-parallel, or an inconsistent cross-camera pairing -> reject
		}
		tri[n_tri].led_id = led;
		tri[n_tri].world_pt = led_world;
		tri[n_tri].n_rays = n_rays;
		for (int r = 0; r < n_rays; r++) {
			tri[n_tri].rays[r] = rays[r];
		}
		n_tri++;
	}

	return aggregate_tri_leds(tri, n_tri, leds_model, &q_prior, prior_yaw_sigma_rad, min_leds, out_result);
}

bool
multicam_triangulate_epipolar_position(const struct multicam_tri_view *views,
                                       int num_views,
                                       const struct t_constellation_led_model *leds_model,
                                       const struct xrt_pose *P_world_obj_prior,
                                       float prior_yaw_sigma_rad,
                                       float reach_m,
                                       float model_gate_m,
                                       int min_leds,
                                       struct multicam_tri_result *out_result)
{
	if (views == NULL || leds_model == NULL || P_world_obj_prior == NULL || out_result == NULL ||
	    num_views < 2 || leds_model->num_leds <= 0 || !(reach_m > 0.f) || !(model_gate_m > 0.f)) {
		return false;
	}
	memset(out_result, 0, sizeof(*out_result));
	if (min_leds < 1) {
		min_leds = 1;
	}

	struct xrt_quat q_prior = P_world_obj_prior->orientation;
	math_quat_normalize(&q_prior);
	const V3d prior_pos = {(double)P_world_obj_prior->position.x, (double)P_world_obj_prior->position.y,
	                       (double)P_world_obj_prior->position.z};

	// Project each model LED to a world position under the prior pose: the gate anchor for clutter rejection.
	// A candidate epipolar point is only kept if it matches one of these (the recovered constellation must be
	// the rigid controller at the predicted body pose), and each model LED is claimed by at most one point.
	const int max_leds = leds_model->num_leds < 64 ? leds_model->num_leds : 64;
	V3d led_world_prior[64];
	for (int led = 0; led < max_leds; led++) {
		struct xrt_vec3 lp;
		math_quat_rotate_vec3(&q_prior, &leds_model->leds[led].pos, &lp);
		led_world_prior[led] = {prior_pos.x + lp.x, prior_pos.y + lp.y, prior_pos.z + lp.z};
	}

	// Per model LED: the distinct-view rays of every cross-view pair that matched it (one ray per view).
	// Merging across pairs lets an LED imaged by 3 cameras accumulate 3 rays; the re-triangulation below then
	// solves it from all of them. No per-pair point is kept — the merged-ray solve is the estimate.
	struct LedAccum
	{
		bool used;
		LedRay rays[MULTICAM_TRI_MAX_VIEWS];
		int n_rays;
	};
	LedAccum acc[64];
	for (int i = 0; i < max_leds; i++) {
		acc[i].used = false;
		acc[i].n_rays = 0;
	}

	// Precompute every blob's world ray once (origin = camera centre, unit bearing in world).
	struct BlobRay
	{
		LedRay ray;
		bool ok;
	};
	BlobRay bray[MULTICAM_TRI_MAX_VIEWS][64];
	int n_view_blobs[MULTICAM_TRI_MAX_VIEWS] = {0};
	for (int vi = 0; vi < num_views; vi++) {
		const struct multicam_tri_view *view = &views[vi];
		const int nb = (view->blobs != NULL && view->calib != NULL && view->num_blobs > 0)
		                   ? (view->num_blobs < 64 ? view->num_blobs : 64)
		                   : 0;
		n_view_blobs[vi] = nb;
		for (int bi = 0; bi < nb; bi++) {
			bray[vi][bi].ok = blob_world_ray(view, bi, vi, bray[vi][bi].ray);
		}
	}

	// Epipolar matching over every camera pair: a blob in A and a blob in B whose rays nearly intersect at a
	// point that is in reach AND matches a model LED is a same-LED correspondence. Assign it to its nearest
	// model LED and merge both rays into that LED's accumulator (one ray per view).
	for (int a = 0; a < num_views; a++) {
		for (int ai = 0; ai < n_view_blobs[a]; ai++) {
			if (!bray[a][ai].ok) {
				continue;
			}
			for (int b = a + 1; b < num_views; b++) {
				for (int bi = 0; bi < n_view_blobs[b]; bi++) {
					if (!bray[b][bi].ok) {
						continue;
					}
					LedRay pair[2] = {bray[a][ai].ray, bray[b][bi].ray};
					V3d pt;
					if (!triangulate_rays(pair, 2, pt)) {
						continue; // near-parallel or rays do not intersect -> not the same LED
					}
					if (norm(sub(pt, prior_pos)) > (double)reach_m) {
						continue; // out of arm's reach of the predicted body -> wrong pairing / clutter
					}
					// Nearest model LED under the prior pose, within the gate (rigid-body consistency).
					int best_led = -1;
					double best_d = (double)model_gate_m;
					for (int led = 0; led < max_leds; led++) {
						const double d = norm(sub(pt, led_world_prior[led]));
						if (d < best_d) {
							best_d = d;
							best_led = led;
						}
					}
					if (best_led < 0) {
						continue; // matches no model LED -> clutter, drop
					}
					LedAccum &la = acc[best_led];
					la.used = true;
					// Accumulate every distinct-view ray for this LED (one per view): a LED imaged by 3
					// cameras gains all 3 rays across the pairs that matched it, sharpening the solve.
					for (int k = 0; k < 2; k++) {
						bool seen = false;
						for (int r = 0; r < la.n_rays; r++) {
							if (la.rays[r].view_id == pair[k].view_id) {
								seen = true;
								break;
							}
						}
						if (!seen && la.n_rays < MULTICAM_TRI_MAX_VIEWS) {
							la.rays[la.n_rays++] = pair[k];
						}
					}
				}
			}
		}
	}

	// Re-triangulate each claimed model LED from its merged rays (an LED seen in >2 views is now sharper than
	// the seeding pair), then aggregate by the shared robust consensus — identical precision guard as the
	// label path, so a clutter point that slipped a single pair's gates is rejected by the cross-LED median.
	TriLed tri[MAX_TRI];
	int n_tri = 0;
	for (int led = 0; led < max_leds && n_tri < MAX_TRI; led++) {
		if (!acc[led].used || acc[led].n_rays < 2) {
			continue;
		}
		V3d led_world;
		if (!triangulate_rays(acc[led].rays, acc[led].n_rays, led_world)) {
			continue;
		}
		// Re-check the model gate on the merged-ray point (the sharper estimate must still be the model LED).
		if (norm(sub(led_world, led_world_prior[led])) > (double)model_gate_m) {
			continue;
		}
		tri[n_tri].led_id = led;
		tri[n_tri].world_pt = led_world;
		tri[n_tri].n_rays = acc[led].n_rays;
		for (int r = 0; r < acc[led].n_rays; r++) {
			tri[n_tri].rays[r] = acc[led].rays[r];
		}
		n_tri++;
	}

	return aggregate_tri_leds(tri, n_tri, leds_model, &q_prior, prior_yaw_sigma_rad, min_leds, out_result);
}
