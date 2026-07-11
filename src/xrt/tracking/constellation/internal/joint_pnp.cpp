// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Multi-camera joint (generalised / non-central) PnP over pooled LED bearing rays.
 * @ingroup constellation
 */
#include "joint_pnp.h"

#include "math/m_api.h"
#include "math/m_vec3.h"

#include <math.h>
#include <string.h>

/* One pooled correspondence: an LED's 3D point in the object/model frame paired with the camera that
 * observed it and the (undistorted, normalised) image bearing it must reproject to. The non-central
 * generalised PnP minimises the standard reprojection error of each correspondence in its OWN camera's
 * image plane — but over ONE shared object pose P_imu_obj in the rig frame, so cameras of different
 * extrinsics jointly constrain the single pose. q_cam_imu/t_cam_imu = the rig->camera transform
 * (= inverse of P_imu_cam) that places a rig-frame point in this camera; (obs_nx,obs_ny) is the
 * observed normalised image point; focal_px scales the normalised residual to that camera's pixels so a
 * single pixel tolerance gates every camera identically (matching the single-camera path's 3 px). */
struct joint_ray
{
	struct xrt_vec3 led_obj;     // LED position, object/model frame
	struct xrt_quat q_cam_imu;   // rig->camera rotation
	struct xrt_vec3 t_cam_imu;   // rig->camera translation
	struct xrt_vec3 cam_center;  // observing camera's centre in the rig frame (for the mirror-twin bearing)
	double obs_nx, obs_ny;       // observed normalised image point in this camera
	double focal_px;             // this camera's focal length (px)
	double inv_var;              // 1 / pos_var_px2: this blob's measurement-noise (R) weight in the WLS cost
	int view_id;                 // index of the contributing view (to count cross-camera inlier coverage)
};

/* Tuning, all derived from the single-camera path (no free magic numbers):
 *  - REPROJ_PX: the per-ray reprojection tolerance, 3 px — identical to the single-camera RANSAC-PnP's
 *    inlier threshold (ransac_pnp.cpp: reprojectionError = 3.0 / fx). Each ray's residual is converted to
 *    that ray's own camera pixels via focal_px, so one pixel tolerance is correct for every camera.
 *  - HUBER_DELTA: the robust knee == the same 3 px (an outlier mislabel bends to linear beyond it).
 *  - INLIER_TOL: 2x the knee, the post-convergence inlier gate (the single-cam path likewise keeps the
 *    LM polish loose relative to its 3 px inlier set).
 *  - GN: Levenberg-Marquardt with Marquardt damping, seeded by the prior (the rig geometry makes the
 *    prior the basin of the unique joint solution); the iteration/step caps only bound a pathological
 *    input. */
static const double REPROJ_PX = 3.0;
static const double HUBER_DELTA = REPROJ_PX;
static const double INLIER_TOL = 2.0 * REPROJ_PX;
static const int MAX_ITERS = 20;
static const double CONVERGE_STEP = 1e-7; // stop when the SE(3) step is sub-micrometre / sub-microradian
/* The minimum converged inliers a camera must retain to count as genuinely supporting the joint pose —
 * the same per-view floor the gather uses to admit a view (>= 3 labelled LEDs). Two such cameras are the
 * cross-baseline constraint that actually dissolves the mirror; a view with fewer is not a real second
 * observation and the solve falls back to the single-camera path. */
static const int MIN_INLIERS_PER_VIEW = 3;
/* The front/back mirror is DISSOLVED only when the cross-camera baseline makes the wrong branch fit clearly
 * worse than the true one. We commit the joint pose only if the rejected (loser) branch either fails the
 * consistency gate or fits at least this many times worse than the winner. A unit-free 2x margin: the two
 * branches reproject identically in a single camera, so the wrong branch differs only through the
 * disambiguating camera(s); requiring it to cost >= 2x means that camera genuinely separates them, not a
 * near-tie the short-baseline geometry left unresolved. (At a true dissolve the ratio is large; this floors
 * the marginal coin-flip out.) */
static const double DISSOLVE_COST_RATIO = 2.0;

namespace {

// --- minimal fixed-size linear algebra (a 6x6 symmetric solve; no external dependency) ---------

struct Vec6
{
	double v[6];
};

struct Mat6
{
	double m[6][6];
};

//! Solve the symmetric positive-(semi)definite system A x = b by Cholesky (LDL^T), in place. Returns
//! false if A is not numerically PD (a degenerate / rank-deficient pooled-ray configuration), so the
//! caller falls back rather than committing a garbage step.
bool
solve6_spd(const Mat6 &A_in, const Vec6 &b_in, Vec6 &x)
{
	double a[6][6];
	memcpy(a, A_in.m, sizeof(a));
	double y[6];
	for (int i = 0; i < 6; i++) {
		y[i] = b_in.v[i];
	}
	// LDL^T decomposition (a holds L below the diagonal, D on it).
	for (int j = 0; j < 6; j++) {
		double d = a[j][j];
		for (int k = 0; k < j; k++) {
			d -= a[j][k] * a[j][k] * a[k][k];
		}
		if (!(d > 1e-12)) {
			return false; // not PD -> degenerate geometry
		}
		a[j][j] = d;
		for (int i = j + 1; i < 6; i++) {
			double s = a[i][j];
			for (int k = 0; k < j; k++) {
				s -= a[i][k] * a[k][k] * a[j][k];
			}
			a[i][j] = s / d;
		}
	}
	// forward solve L z = y
	for (int i = 0; i < 6; i++) {
		double s = y[i];
		for (int k = 0; k < i; k++) {
			s -= a[i][k] * y[k];
		}
		y[i] = s;
	}
	// diagonal solve D w = z, then back solve L^T x = w
	for (int i = 0; i < 6; i++) {
		y[i] /= a[i][i];
	}
	for (int i = 5; i >= 0; i--) {
		double s = y[i];
		for (int k = i + 1; k < 6; k++) {
			s -= a[k][i] * x.v[k];
		}
		x.v[i] = s;
	}
	return true;
}

//! v3 helpers in double (the public math API is float; the GN normal equations want double).
struct V3d
{
	double x, y, z;
};

V3d
cross(const V3d &a, const V3d &b)
{
	return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

//! The LED's position in the observing camera's frame under the object pose (q,t): p_cam = R_cam_imu *
//! (R q led_obj + t) + t_cam_imu. Also returns Rp (the rotation-only object point in the rig frame, for
//! the rotation Jacobian). Used by both the cost and the GN Jacobian so they stay consistent.
V3d
point_in_cam(const struct joint_ray &ray, const struct xrt_quat &q, const struct xrt_vec3 &t, V3d &Rp_out)
{
	struct xrt_vec3 Rp;
	math_quat_rotate_vec3(&q, &ray.led_obj, &Rp);
	Rp_out = {Rp.x, Rp.y, Rp.z};
	const struct xrt_vec3 p_imu = {Rp.x + t.x, Rp.y + t.y, Rp.z + t.z};
	struct xrt_vec3 p_cam;
	math_quat_rotate_vec3(&ray.q_cam_imu, &p_imu, &p_cam);
	return {(double)p_cam.x + ray.t_cam_imu.x, (double)p_cam.y + ray.t_cam_imu.y,
	        (double)p_cam.z + ray.t_cam_imu.z};
}

/* The near-planar PnP mirror twin of an orientation: the front/back flip a single view cannot tell from
 * the truth. A planar LED patch (object surface normal = object +Z) seen along a bearing v has a twin
 * whose normal is reflected across v (same projected blobs). Reflecting the normal n -> 2(n.v)v - n is
 * achieved by rotating the orientation by 2*angle(n,v) about the axis (n x v): R_twin = Rot . R. We take
 * v as the rig-frame bearing from the cameras' mean centre to the object position (the mean of the ray
 * origins -> t). A degenerate (normal aligned with the bearing) twin falls back to the input. Used to
 * seed the joint LM from the OTHER branch so a flipped prior is still pulled to the truth. */
struct xrt_quat
mirror_twin_orientation(const struct xrt_quat &q,
                        const struct xrt_vec3 &t,
                        const struct joint_ray *rays,
                        int n_rays)
{
	// object surface normal (object +Z) in the rig frame.
	const struct xrt_vec3 obj_z = {0.f, 0.f, 1.f};
	struct xrt_vec3 n;
	math_quat_rotate_vec3(&q, &obj_z, &n);
	math_vec3_normalize(&n);

	// mean camera centre (in the rig frame) -> bearing to the object.
	V3d c = {0, 0, 0};
	for (int i = 0; i < n_rays; i++) {
		c.x += rays[i].cam_center.x;
		c.y += rays[i].cam_center.y;
		c.z += rays[i].cam_center.z;
	}
	const double inv = n_rays > 0 ? 1.0 / n_rays : 1.0;
	struct xrt_vec3 v = {(float)(t.x - c.x * inv), (float)(t.y - c.y * inv), (float)(t.z - c.z * inv)};
	math_vec3_normalize(&v);

	struct xrt_vec3 axis;
	const struct xrt_vec3 nn = n;
	math_vec3_cross(&nn, &v, &axis);
	const double axis_len = m_vec3_len(axis);
	if (axis_len < 1e-6) {
		return q; // normal already along the bearing -> no distinct planar twin
	}
	axis.x /= (float)axis_len;
	axis.y /= (float)axis_len;
	axis.z /= (float)axis_len;
	const double cos_t = fmax(-1.0, fmin(1.0, (double)m_vec3_dot(n, v)));
	const double angle = 2.0 * acos(cos_t);

	struct xrt_quat rot;
	math_quat_from_angle_vector((float)angle, &axis, &rot);
	struct xrt_quat twin;
	math_quat_rotate(&rot, &q, &twin);
	math_quat_normalize(&twin);
	return twin;
}

} // namespace

/* Build the pooled rays for one device across all views, deduplicated per LED per view. Returns the
 * number of rays and the number of distinct views that actually contributed at least one ray (the
 * cross-camera count that decides whether a joint solve is meaningful). */
static int
gather_rays(const struct joint_pnp_view *views,
            int num_views,
            struct t_constellation_led_model *leds_model,
            struct joint_ray *rays,
            int max_rays,
            int *out_contributing_views)
{
	int n = 0;
	int contributing = 0;
	for (int vi = 0; vi < num_views && n < max_rays; vi++) {
		const struct joint_pnp_view *view = &views[vi];
		if (view->blobs == NULL || view->num_blobs <= 0 || view->calib == NULL) {
			continue;
		}
		// rig->camera transform (inverse of P_imu_cam) brings a rig-frame point into this camera's frame.
		struct xrt_pose P_cam_imu;
		math_pose_invert(&view->P_imu_cam, &P_cam_imu);
		uint64_t taken = 0;
		int view_rays = 0;
		for (int bi = 0; bi < view->num_blobs && n < max_rays; bi++) {
			const struct blob *b = &view->blobs[bi];
			if (LED_OBJECT_ID(b->led_id) != leds_model->id) {
				continue; // unlabelled or another device's blob
			}
			const int led_id = LED_LOCAL_ID(b->led_id);
			if (led_id >= leds_model->num_leds || (taken & (1ULL << led_id)) != 0) {
				continue; // out of range or this LED already has a ray in this view
			}
			taken |= (1ULL << led_id);

			// Undistort the blob to a normalised image point in this camera's frame.
			float nx = 0.f, ny = 0.f;
			t_camera_models_undistort(&view->calib->calib, b->x, b->y, &nx, &ny);

			rays[n].led_obj = leds_model->leds[led_id].pos;
			rays[n].q_cam_imu = P_cam_imu.orientation;
			rays[n].t_cam_imu = P_cam_imu.position;
			rays[n].cam_center = view->P_imu_cam.position;
			rays[n].view_id = vi;
			rays[n].obs_nx = nx;
			rays[n].obs_ny = ny;
			// Per-camera focal (mean of fx,fy) to express the normalised residual in this camera's pixels.
			rays[n].focal_px = 0.5 * ((double)view->calib->calib.fx + (double)view->calib->calib.fy);
			// Per-blob measurement-noise (R) weight: a fat/clipped/edge/faint blob (large pos_var_px2, an
			// uncertain LED centre) contributes proportionally less to the pooled WLS cost. pos_var_px2 is
			// floored >= 0.25 px^2 at the source (blobwatch), so the reciprocal is well-posed.
			rays[n].inv_var = 1.0 / (double)b->pos_var_px2;
			n++;
			view_rays++;
		}
		if (view_rays > 0) {
			contributing++;
		}
	}
	if (out_contributing_views != NULL) {
		*out_contributing_views = contributing;
	}
	return n;
}

//! One ray's reprojection residual (in this camera's pixels) under the object pose (q,t), or a bounded
//! large value if the LED is behind the camera. residual = focal * ((px/pz, py/pz) - (obs_nx, obs_ny)).
static double
ray_residual_px(const struct joint_ray &ray, const struct xrt_quat &q, const struct xrt_vec3 &t)
{
	V3d Rp;
	const V3d p = point_in_cam(ray, q, t, Rp);
	if (p.z < 1e-4) {
		return 1e3 * HUBER_DELTA; // behind / at the camera: large bounded residual (an LM step won't keep it)
	}
	const double ex = ray.focal_px * (p.x / p.z - ray.obs_nx);
	const double ey = ray.focal_px * (p.y / p.z - ray.obs_ny);
	return sqrt(ex * ex + ey * ey);
}

/* The pooled robust-WLS reprojection cost of a candidate object pose: each ray's Huber loss (in pixels,
 * the knee bounding a mislabel outlier) scaled by its measurement-noise weight inv_var = 1/pos_var_px2, so
 * an uncertain (fat/clipped/edge/faint) blob centre carries proportionally less weight. Summed over all
 * cameras; used to accept/reject an LM step (the Marquardt damping update). */
static double
pooled_cost(const struct joint_ray *rays, int n_rays, const struct xrt_quat &q, const struct xrt_vec3 &t)
{
	double cost = 0.0;
	for (int i = 0; i < n_rays; i++) {
		const double r = ray_residual_px(rays[i], q, t);
		const double rho = (r <= HUBER_DELTA) ? 0.5 * r * r : HUBER_DELTA * (r - 0.5 * HUBER_DELTA);
		cost += rays[i].inv_var * rho;
	}
	return cost;
}

/* The pooled-ray consistency of a converged pose: total inliers (residual <= INLIER_TOL) and the number of
 * cameras that each strongly support it (>= MIN_INLIERS_PER_VIEW inliers). A pose is ACCEPTABLE only when at
 * least two cameras each strongly support it AND (almost) every pooled ray is in-tolerance — the genuine
 * multi-camera agreement that dissolves the mirror, not a one-camera fit with stray inliers from the other. */
static bool
pose_consistent(const struct joint_ray *rays,
                int n_rays,
                const struct xrt_quat &q,
                const struct xrt_vec3 &t,
                int *out_inliers)
{
	int inliers = 0, strong_views = 0, cur_view = -1, cur_view_inliers = 0;
	for (int i = 0; i < n_rays; i++) {
		if (rays[i].view_id != cur_view) {
			if (cur_view_inliers >= MIN_INLIERS_PER_VIEW) {
				strong_views++;
			}
			cur_view = rays[i].view_id;
			cur_view_inliers = 0;
		}
		if (ray_residual_px(rays[i], q, t) <= INLIER_TOL) {
			inliers++;
			cur_view_inliers++;
		}
	}
	if (cur_view_inliers >= MIN_INLIERS_PER_VIEW) {
		strong_views++; // the last view's run
	}
	if (out_inliers != NULL) {
		*out_inliers = inliers;
	}
	return inliers >= 6 && strong_views >= 2 && inliers + 1 >= n_rays;
}

/* Levenberg-Marquardt refine the object pose (q,t) over all pooled rays, in place. Minimises the pooled
 * Huber reprojection cost on the SE(3) tangent (LEFT/world rotation perturbation + additive translation),
 * with Marquardt damping (grow on rejection, shrink on acceptance). Returns the converged pooled cost. */
static double
lm_refine(const struct joint_ray *rays, int n_rays, struct xrt_quat *q_io, struct xrt_vec3 *t_io)
{
	struct xrt_quat q = *q_io;
	struct xrt_vec3 t = *t_io;
	double lambda = 1e-3;
	double cost = pooled_cost(rays, n_rays, q, t);

	for (int iter = 0; iter < MAX_ITERS; iter++) {
		// Accumulate the 6x6 Gauss-Newton normal equations (rotation [0..2], translation [3..5]).
		Mat6 H;
		Vec6 g;
		memset(&H, 0, sizeof(H));
		memset(&g, 0, sizeof(g));

		for (int i = 0; i < n_rays; i++) {
			V3d Rp; // R q led_obj (rotation-only object point, in the rig frame)
			const V3d pc = point_in_cam(rays[i], q, t, Rp);
			if (pc.z < 1e-4) {
				continue; // LED behind its camera at the current estimate -> no usable Jacobian
			}
			const double f = rays[i].focal_px;
			const double inv_z = 1.0 / pc.z;
			// 2D reprojection residual in pixels: r = f * ([px/pz, py/pz] - obs).
			const double res2[2] = {f * (pc.x * inv_z - rays[i].obs_nx), f * (pc.y * inv_z - rays[i].obs_ny)};
			const double r = sqrt(res2[0] * res2[0] + res2[1] * res2[1]);

			// Robust-WLS IRLS weight: the per-blob measurement-noise weight inv_var = 1/pos_var_px2 times
			// the Huber factor (1 inside the knee, delta/r outside, capping a mislabel's leverage). Matches
			// the pooled_cost loss, so the normal equations and the step-acceptance cost stay consistent.
			const double huber = (r <= HUBER_DELTA) ? 1.0 : (HUBER_DELTA / (r > 1e-12 ? r : 1e-12));
			const double weight = rays[i].inv_var * huber;

			/* d(residual)/d(SE(3) tangent of P_imu_obj), tangent = [rot(0..2, LEFT/world), trans(3..5)]:
			 *   dproj/dp_cam = (f/pz) [[1,0,-px/pz],[0,1,-py/pz]]   (the pinhole projection Jacobian)
			 *   dp_cam/dt_imu  = R_cam_imu                          (translation column block)
			 *   dp_cam/drot    = R_cam_imu * (-[Rp]_x)              (Rp = R q led_obj; LEFT-perturbation
			 *                                                        q_new = exp(drot).q, matching the retraction)
			 * So the 2x6 Jacobian J = Jproj * R_cam_imu * [ -[Rp]_x | I ]. Built column-by-column. */
			double Jp[2][3]; // dproj/dp_cam (2x3), in pixels
			Jp[0][0] = f * inv_z;
			Jp[0][1] = 0.0;
			Jp[0][2] = -f * pc.x * inv_z * inv_z;
			Jp[1][0] = 0.0;
			Jp[1][1] = f * inv_z;
			Jp[1][2] = -f * pc.y * inv_z * inv_z;

			// dp_cam for each tangent column (a rig-frame vector rotated into the camera by R_cam_imu).
			const V3d ex = {1, 0, 0}, ey = {0, 1, 0}, ez = {0, 0, 1};
			// rotation columns (LEFT perturbation exp([drot]_x).R): dRp/drot_k = e_k x Rp.
			const V3d dpi[6] = {cross(ex, Rp), cross(ey, Rp), cross(ez, Rp), ex, ey, ez};

			double J[2][6];
			for (int c = 0; c < 6; c++) {
				// rotate the rig-frame column into the camera frame.
				struct xrt_vec3 v = {(float)dpi[c].x, (float)dpi[c].y, (float)dpi[c].z};
				struct xrt_vec3 vc;
				math_quat_rotate_vec3(&rays[i].q_cam_imu, &v, &vc);
				const double pcam[3] = {vc.x, vc.y, vc.z};
				J[0][c] = Jp[0][0] * pcam[0] + Jp[0][1] * pcam[1] + Jp[0][2] * pcam[2];
				J[1][c] = Jp[1][0] * pcam[0] + Jp[1][1] * pcam[1] + Jp[1][2] * pcam[2];
			}

			for (int a = 0; a < 6; a++) {
				for (int bb = a; bb < 6; bb++) {
					H.m[a][bb] += weight * (J[0][a] * J[0][bb] + J[1][a] * J[1][bb]);
				}
				g.v[a] -= weight * (J[0][a] * res2[0] + J[1][a] * res2[1]); // negative gradient
			}
		}
		// Mirror the upper triangle into the lower.
		for (int a = 0; a < 6; a++) {
			for (int bb = 0; bb < a; bb++) {
				H.m[a][bb] = H.m[bb][a];
			}
		}

		// Try a damped step; grow damping on rejection, shrink on acceptance (Marquardt).
		bool stepped = false;
		for (int tries = 0; tries < 5; tries++) {
			Mat6 Hd = H;
			for (int a = 0; a < 6; a++) {
				Hd.m[a][a] += lambda * (H.m[a][a] + 1e-9);
			}
			Vec6 step;
			if (!solve6_spd(Hd, g, step)) {
				lambda *= 10.0;
				continue;
			}
			// Retract: q_new = exp(drot) . q (LEFT/world perturbation, matching the -[Rp]_x Jacobian).
			const struct xrt_vec3 drot = {(float)step.v[0], (float)step.v[1], (float)step.v[2]};
			struct xrt_quat dq;
			math_quat_exp(&drot, &dq);
			struct xrt_quat q_new;
			math_quat_rotate(&dq, &q, &q_new);
			math_quat_normalize(&q_new);
			struct xrt_vec3 t_new = {t.x + (float)step.v[3], t.y + (float)step.v[4], t.z + (float)step.v[5]};

			const double new_cost = pooled_cost(rays, n_rays, q_new, t_new);
			if (new_cost < cost) {
				q = q_new;
				t = t_new;
				cost = new_cost;
				lambda = lambda > 1e-9 ? lambda * 0.5 : lambda;
				stepped = true;
				const double step_norm =
				    sqrt(step.v[0] * step.v[0] + step.v[1] * step.v[1] + step.v[2] * step.v[2] +
				         step.v[3] * step.v[3] + step.v[4] * step.v[4] + step.v[5] * step.v[5]);
				if (step_norm < CONVERGE_STEP) {
					iter = MAX_ITERS; // converged
				}
				break;
			}
			lambda *= 10.0;
		}
		if (!stepped) {
			break; // no further descent (converged or stuck)
		}
	}

	*q_io = q;
	*t_io = t;
	return cost;
}

bool
joint_pnp_solve(struct xrt_pose *pose,
                const struct joint_pnp_view *views,
                int num_views,
                struct t_constellation_led_model *leds_model,
                int *num_rays_out,
                int *num_inliers)
{
	if (num_rays_out != NULL) {
		*num_rays_out = 0;
	}
	if (num_inliers != NULL) {
		*num_inliers = 0;
	}
	if (pose == NULL || views == NULL || leds_model == NULL || num_views < 2) {
		return false;
	}

	struct joint_ray rays[JOINT_PNP_MAX_VIEWS * JOINT_PNP_MAX_LEDS_PER_VIEW];
	int contributing_views = 0;
	const int n_rays =
	    gather_rays(views, num_views, leds_model, rays, JOINT_PNP_MAX_VIEWS * JOINT_PNP_MAX_LEDS_PER_VIEW, &contributing_views);
	if (num_rays_out != NULL) {
		*num_rays_out = n_rays;
	}

	/* The joint solve is only meaningful — and only dissolves the mirror — when at least two cameras
	 * each contribute a ray. With one contributing camera the geometry is exactly the single-camera
	 * (mirror-prone) case, so we decline and the caller keeps its single-camera path. Need >= 6 rays to
	 * constrain 6 DOF; the second camera is what lifts a near-planar single view above that floor. */
	if (contributing_views < 2 || n_rays < 6) {
		return false;
	}

	/* Refine the object pose jointly over all pooled rays. LM is a LOCAL optimiser, so to dissolve the
	 * mirror even when the PRIOR itself is on the wrong (flipped) branch — which is exactly when the
	 * front-end most needs help — we refine from TWO seeds and keep the lower-residual one:
	 *   (1) the prior as given;
	 *   (2) the prior reflected to its near-planar mirror twin (the front/back flip).
	 * Two cameras observe the controller from different baselines, so only the TRUE pose fits BOTH; the
	 * wrong-branch seed converges to a high pooled residual and loses. The mirror thus dissolves by
	 * construction regardless of which branch the prior was on (a single central camera could not separate
	 * them — the second camera breaks the tie). The twin is built geometrically: reflect the object's
	 * surface normal about the rig-frame bearing to the object, the exact planar two-fold relation. */
	struct xrt_quat q0 = pose->orientation;
	math_quat_normalize(&q0);
	const struct xrt_quat q_twin = mirror_twin_orientation(q0, pose->position, rays, n_rays);

	/* Refine BOTH branches independently, then keep the lower-cost one — BUT only commit it when the
	 * cross-camera baseline has genuinely DISSOLVED the mirror, i.e. the loser is decisively worse. The
	 * solve's whole premise is that only the TRUE branch fits every camera; if BOTH branches fit the pooled
	 * rays comparably (a short baseline / a marginal second view that could not separate the front/back
	 * flip), committing the lower-cost one is a coin-flip on the mirror — exactly the marginal wrong-branch
	 * commit. So we require either that the loser FAILS the consistency gate, or that its pooled cost is at
	 * least DISSOLVE_COST_RATIO times the winner's (the disambiguating camera clearly rejects it). Both arms
	 * encode the same precondition ("the second camera breaks the tie") and reuse existing quantities — no
	 * free threshold beyond the ratio, which is a unit-free 2x margin (the loser fits at least twice as
	 * badly), not a pixel magic number. Otherwise decline to the single-camera cascade (its own flip
	 * handling applies). */
	const struct xrt_quat seeds[2] = {q0, q_twin};
	struct xrt_quat ref_q[2];
	struct xrt_vec3 ref_t[2];
	double ref_cost[2];
	int ref_inliers[2] = {0, 0};
	bool ref_ok[2] = {false, false};
	for (int s = 0; s < 2; s++) {
		ref_q[s] = seeds[s];
		ref_t[s] = pose->position;
		ref_cost[s] = lm_refine(rays, n_rays, &ref_q[s], &ref_t[s]);
		ref_ok[s] = pose_consistent(rays, n_rays, ref_q[s], ref_t[s], &ref_inliers[s]);
	}

	const int win = ref_cost[0] <= ref_cost[1] ? 0 : 1;
	const int lose = 1 - win;
	if (num_inliers != NULL) {
		*num_inliers = ref_inliers[win];
	}
	if (!ref_ok[win]) {
		return false; // the better branch itself doesn't fit every camera -> not a joint solve
	}
	// Mirror dissolved iff the loser is clearly rejected: it fails the gate, or fits at least 2x worse.
	const bool dissolved = !ref_ok[lose] || ref_cost[lose] >= DISSOLVE_COST_RATIO * ref_cost[win];
	if (!dissolved) {
		return false; // both branches fit comparably -> the baseline did not break the tie; decline
	}

	pose->orientation = ref_q[win];
	pose->position = ref_t[win];
	return true;
}
