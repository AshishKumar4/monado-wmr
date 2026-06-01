// Copyright 2023 Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Implementation of LED constellation tracking
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup constellation
 */
#include <inttypes.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#include "os/os_threading.h"

#include "tracking/t_led_models.h"
#include "tracking/t_constellation_tracking.h"

#include "util/u_debug.h"
#include "util/u_frame.h"
#include "util/u_g2_telemetry.h"
#include "util/u_logging.h"
#include "util/u_sink.h"
#include "util/u_trace_marker.h"
#include "util/u_var.h"

#include "internal/blobwatch.h"
#include "internal/association_hypothesis.h"
#include "internal/camera_model.h"
#include "internal/correspondence_search.h"
#include "internal/debug_draw.h"
#include "internal/joint_pnp.h"
#include "internal/ransac_pnp.h"
#include "internal/sample.h"

DEBUG_GET_ONCE_LOG_OPTION(ct_log, "CONSTELLATION_LOG", U_LOGGING_INFO)

#define MIN_ROT_ERROR DEG_TO_RAD(30)
#define MIN_POS_ERROR 0.10

/* Covariance-driven prior-consistency gate: size the prior tolerance by the fusion's LIVE 1-sigma
 * uncertainty (PRIOR_GATE_SIGMA sigmas), instead of a fixed value. MIN_*_ERROR above are the FLOORS
 * (so when the fusion is confident the gate is no looser, and no tighter, than before -> no
 * regression; flips are rejected by the large rotation error they produce regardless). MAX_*_ERROR
 * are the CEILINGS that bound how far the gate widens after an optical dropout, when the fusion
	 * inflates its covariance -> projected-prior and cold-search hypotheses are scored against a realistic
	 * envelope instead of a stale tight prior. One statistical knob (the sigma multiplier); the rest are
	 * physical floors/ceilings. */
#define PRIOR_GATE_SIGMA 3.0 /* ~99.7% per axis */
#define MAX_POS_ERROR 0.60
#define MAX_ROT_ERROR DEG_TO_RAD(60)

/* The tight DRIFTLESS tilt sigma for the soft anisotropic mirror-flip cost. The mirror twin of a few-LED PnP
 * almost always TILTS the controller wrong. Gravity is
 * driftless: the fusion prior's TILT is gravity-anchored (the ESKF keeps
 * anchoring from the controller's accel even through an optical dropout) so the prior tilt is a valid
 * reference even when the yaw prior is stale (the dropout/"snap-back" case the gyro can't catch). A real
 * frame-to-frame tilt is « this within the tracked regime; a tilt flip is ~>=90 deg (>=3 sigma -> huge
 * penalty). Equal to the rotation FLOOR (one knob): the tilt scale tracks the per-axis prior bound. */
#define GRAVITY_TILT_TOL MIN_ROT_ERROR

/* The mirror-flip disambiguation is a SOFT cost re-rank, not a hard veto: for each candidate pose
 * and its mirror twin, cost = reproj_error_px + pose_metrics_prior_orient_cost(...), and the LOWEST-cost
 * candidate is committed — a frame is never dropped for ambiguity ("fix, don't reject"). The cost is the
 * anisotropic squared Mahalanobis distance of the candidate orientation from the prior — tilt scaled by
 * the tight driftless GRAVITY_TILT_TOL, yaw by the live fusion 1-sigma — Huber-robustified and weighted
 * into px. A confident prior makes the prior term dominate (a tilt flip's huge distance is never selected),
 * while an uncertain/untracked prior makes the yaw term vanish and reprojection decide. The
 * Huber knee reuses the 3-sigma envelope (PRIOR_GATE_SIGMA): within it the penalty is quadratic, beyond
 * it linear, so a gross outlier (flip) cannot dominate pathologically. FLIP_COST_WEIGHT commensurates the
 * dimensionless robustified distance with the per-LED reprojection error (px^2). FLIP_COST_YAW_SIGMA_MAX
 * is the untracked/long-dropout yaw scale CEILING — at 180deg it lets a 180deg flip cost d^2=1.0
 * (negligible), so twins tie on reprojection during fast motion. 90deg keeps the legitimate-motion
 * relaxation while still penalising flips at d^2=4. */
#define FLIP_COST_WEIGHT 1.0
#define FLIP_COST_HUBER_KNEE_SIGMA PRIOR_GATE_SIGMA
#define FLIP_COST_YAW_SIGMA_MAX DEG_TO_RAD(90)
/* The yaw-scale FLOOR for the soft flip cost: the minimum yaw 1-sigma the cost will use, regardless of how
 * confident the ESKF reports it is. This is a separate concept from the GRAVITY_TILT_TOL tilt scale (they are
 * different DoFs and must not share one constant), but the VALUE here is governed by a real effect the ESKF
 * yaw covariance cannot see: the optical front-end occasionally folds a wrong-yaw pose, after which the filter
 * is legitimately CONFIDENT (small reported yaw sigma) in a yaw that is actually off — an over-confident prior.
 * Measured on G2 controller captures, ~11-19% of accepted optical poses disagree with the gyro/fusion prior by
 * >45deg, so a yaw scale tighter than this floor makes the cost over-trust that stale prior and SELECT the
 * candidate that matches the wrong yaw (offline A/B: dropping the floor to 3deg raised Left wrong-branch
 * 24.8->37.0% and flip 10.0->11.3%, while 30-60deg are statistically flat). The floor caps that over-trust;
 * the live sigma still WIDENS the scale above it after a real dropout (the cost relaxes and reprojection
 * decides). Equal to MIN_ROT_ERROR, the prior gate's per-axis rotation floor: the yaw scale never claims more
 * yaw confidence than the gate's tightest accepted rotation tolerance, the empirically validated knee. */
#define FLIP_COST_YAW_SIGMA_MIN MIN_ROT_ERROR

/* Yaw-sigma above which a 4-LED single-cam PnP defers: above this, the soft prior cost can no longer
 * reliably tell the mirror twins apart. 5+ LEDs break near-coplanarity so no twin is emitted. */
#define FLIP_COST_YAW_SIGMA_GUARD DEG_TO_RAD(60)

/* Head-anchored yaw cue (Fix A): the soft flip cost adds the candidate's HEAD-RELATIVE-yaw distance
 * from the last accepted controller-in-head orientation. INDEPENDENT of the world-frame fusion prior:
 * a 180-deg mirror flip flips head-relative yaw too, so this signal stays valid even when the world
 * prior is stale, and is robust to SLAM global drift (a delta in head frame cancels common drift).
 *
 * TTL caps how long after the last accept the cue is trusted: too long and head+hand have moved
 * independently enough that head-relative yaw is no longer a tight constraint; too short and we miss
 * the consecutive-frame disambiguation that is the whole point. ~100ms covers ~3 frames at 30Hz —
 * the regime in which the twin-disambiguation matters most. */
#define HEAD_YAW_CUE_TTL_NS (100ll * 1000ll * 1000ll)
/* Yaw scale for the head-anchored term: chosen so legitimate head-vs-controller relative motion (the
 * fastest plausible 30deg / frame at 30Hz, i.e. ~900deg/s relative rotation) sits within ~1 sigma
 * while a 180-deg flip is ~3 sigma -> huber-bent linear penalty. */
#define HEAD_YAW_CUE_YAW_SIGMA DEG_TO_RAD(60)
/* Tilt scale wide enough to NOT penalise legitimate hand-tilt-vs-head-tilt motion: this term is the
 * YAW signal; the world-frame term already handles the driftless tilt. */
#define HEAD_YAW_CUE_TILT_SIGMA DEG_TO_RAD(180)

/* Covariance-gated partial-fold. When the unified associator cannot lock a pose but the fusion has
 * a usable prior, we still fold the
 * individual LEDs CONFIDENTLY matched to the prior, each gated by the ESKF's anisotropic per-LED
 * innovation covariance S = H·P·Hᵀ + R (predict_led_gate). A blob<->LED pairing is folded iff its
 * Mahalanobis distance d² = rᵀS⁻¹r ≤ this χ²₂ quantile. χ²₂ inverse-CDF: -2·ln(1-p). p=0.99 -> 9.21,
 * matching the fold's own per-LED gate (CHI2_GATE_2DOF) so the two lines of defence are consistent.
 * The anisotropic S (tight tilt / loose yaw after a dropout) makes a tilt-flipped correspondence land
 * outside the gate automatically — no separate gravity check needed here. */
#define PARTIAL_FOLD_CHI2_2DOF 9.21 /* -2*ln(1-0.99) */

#define CT_TRACE(c, ...) U_LOG_IFL_T(c->log_level, __VA_ARGS__)
#define CT_DEBUG(c, ...) U_LOG_IFL_D(c->log_level, __VA_ARGS__)
#define CT_INFO(c, ...) U_LOG_IFL_I(c->log_level, __VA_ARGS__)
#define CT_WARN(c, ...) U_LOG_IFL_W(c->log_level, __VA_ARGS__)
#define CT_ERROR(c, ...) U_LOG_IFL_E(c->log_level, __VA_ARGS__)

/* Maximum number of frames to permit waiting in the fast-processing queue */
#define MAX_FAST_QUEUE_SIZE 2

/* Predictive-ROI blob detection (Oasis driver convergent design, RE'd from MS's
 * ConnectedComponent::Locate -> ILedLocationPredictor::Predict path): for each tracked device, project
 * every LED in its constellation through the device's current ESKF state to its predicted image-pixel
 * position in the camera; the union of per-LED patches (each sized to that LED's actual prediction
 * uncertainty from predict_led_gate's S) gives a tight region the blob detector restricts its
 * flood-fill to. Eliminates ambient-noise blobs from the correspondence search (the matcher's
 * dominant "52% idle" defect on our side) and pre-localizes blobs around specific LED candidates.
 *
 * Per-LED adaptive sizing (NOT MS's fixed 16-px patch): each LED's pad = max(ROI_BASE_PAD_PX,
 * ROI_SIGMA_K * pixel_sigma), where pixel_sigma = sqrt(max diag of S). The S matrix returned by
 * predict_led_gate captures both the ESKF posterior position uncertainty AND the LED's geometric
 * sensitivity at this camera+pose. A precise prediction (small S) gets a near-minimum pad; an
 * uncertain prediction (large S, e.g. post-coast or low-confidence pose) gets a wider pad
 * proportionally. This per-LED scaling naturally handles asymmetric controllers (left vs right) and
 * coast-vs-fresh frames without a single global pad that must be conservative enough for the worst
 * case while also tight enough for the best.
 *
 * Fallback: when fewer than ROI_FALLBACK_MIN_PREDICTIONS LEDs project into the frame across all
 * tracked devices (cold start, lost tracking), the helper returns false and the caller uses the
 * full-frame path -- matching MS's PatchSearchFallbackMinPoints = 6.
 *
 * Always-on; the cold-start path falls through to full-frame via the predict_led_gate untracked
 * return. Env G2_PREDICTIVE_ROI_BASE_PAD_PX + G2_PREDICTIVE_ROI_SIGMA_K override the defaults for
 * in-headset re-tuning across lighting/motion. */
/* MS uses PredictivePatchSize/2 = 8 as their per-LED floor, but their predictor extrapolates the ESKF
 * state with IMU at the frame's EXPOSURE time, so their predictions are tight. Our predict_led_gate
 * uses the most-recent ESKF state which can lag the actual exposure by up to ~11ms (one frame at 90Hz);
 * empirically the temporal drift floor is much larger than 8px. Bisection on the headpose capture
 * (commit b8cc1fab2) found bp=32 with sigma_k=3.0 is the genuine sweet spot — 5pp wrongBr wins on
 * both controllers without crossing the threshold where larger pads admit ambient noise. The S matrix
 * scaling (per-LED) handles uncertainty growth above this floor. */
#define ROI_BASE_PAD_PX 32      /* floor: temporal-drift-aware; MS's 8 is too tight for our timing */
#define ROI_SIGMA_K 3.0f        /* per-LED pad = k * sqrt(max-diag(S)); 3σ ~ 99.7% coverage */
#define ROI_FALLBACK_MIN_PREDICTIONS 6

//! The OpenCV(+Y down, +Z away) <-> OpenXR(+Y up, +Z toward) camera-basis flip: a 180-deg rotation about
//! X (i.e. negate Y and Z). Single source for the convention so every CV<->XR conversion agrees.
static const struct xrt_pose P_YZ_FLIP = {{1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};

//! Sandwich a pose by the YZ flip (convert a pose between the OpenCV and OpenXR camera bases).
static void
pose_flip_YZ(const struct xrt_pose *in, struct xrt_pose *dest)
{
	struct xrt_pose tmp;
	math_pose_transform(&P_YZ_FLIP, in, &tmp);
	math_pose_transform(&tmp, &P_YZ_FLIP, dest);
}


/* Map an xrt_device type to the telemetry device_id (0=HMD, 1=left, 2=right). */
static uint8_t
telem_device_id(const struct xrt_device *xdev)
{
	if (xdev == NULL)
		return 0;
	switch (xdev->device_type) {
	case XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER: return 1;
	case XRT_DEVICE_TYPE_RIGHT_HAND_CONTROLLER: return 2;
	default: return 0;
	}
}

/* Pack an xrt_pose into the [px,py,pz, qx,qy,qz,qw] layout telemetry expects. */
static void
telem_pack_pose(const struct xrt_pose *p, float out[7])
{
	out[0] = p->position.x;
	out[1] = p->position.y;
	out[2] = p->position.z;
	out[3] = p->orientation.x;
	out[4] = p->orientation.y;
	out[5] = p->orientation.z;
	out[6] = p->orientation.w;
}

enum g2_search_result
{
	G2_SEARCH_SUCCESS = 0,
	G2_SEARCH_NO_SEARCHABLE_ANCHORS = 1,
	G2_SEARCH_NO_ANCHOR_WITH_3_NEIGHBOURS = 2,
	G2_SEARCH_NO_P3P_TRIALS = 3,
	G2_SEARCH_NO_POSE_CHECKS = 4,
	G2_SEARCH_ALL_POSE_CHECKS_PRUNED = 5,
	G2_SEARCH_BEST_NOT_GOOD = 6,
	G2_SEARCH_NO_GOOD_CANDIDATE = 7,
};

static enum g2_search_result
telem_classify_search_result(bool success, const struct correspondence_search_diagnostics *diag)
{
	if (success) {
		return G2_SEARCH_SUCCESS;
	}
	if (diag->searchable_anchors == 0) {
		return G2_SEARCH_NO_SEARCHABLE_ANCHORS;
	}
	if (diag->anchors_with_3_neighbours == 0) {
		return G2_SEARCH_NO_ANCHOR_WITH_3_NEIGHBOURS;
	}
	if (diag->num_trials == 0) {
		return G2_SEARCH_NO_P3P_TRIALS;
	}
	if (diag->num_pose_checks == 0) {
		return G2_SEARCH_NO_POSE_CHECKS;
	}
	if (diag->num_pose_checks_pruned >= diag->num_pose_checks) {
		return G2_SEARCH_ALL_POSE_CHECKS_PRUNED;
	}
	if (diag->best_any_blobs_matched > 0) {
		return G2_SEARCH_BEST_NOT_GOOD;
	}
	return G2_SEARCH_NO_GOOD_CANDIDATE;
}

static void
telem_emit_search_result(uint8_t device_id,
                         int view_id,
                         uint64_t timestamp_ns,
                         int pass,
                         enum correspondence_search_flags flags,
                         bool success,
                         bool prior_tilt_trusted,
                         const struct correspondence_search_diagnostics *diag)
{
	g2_telem_search(device_id, (uint8_t)view_id, timestamp_ns, (uint8_t)pass,
	                (uint8_t)telem_classify_search_result(success, diag), (uint16_t)flags,
	                prior_tilt_trusted ? 1 : 0, diag->input_blobs, diag->searchable_anchors,
	                diag->filtered_anchors, diag->anchors_with_3_neighbours, diag->neighbour_links,
	                diag->num_trials, diag->num_pose_checks, diag->num_pose_checks_pruned,
	                (uint8_t)diag->min_led_depth, (uint8_t)diag->max_led_depth,
	                (uint8_t)diag->max_blob_depth, (uint8_t)diag->best_any_pose_blob_depth,
	                (uint8_t)diag->best_any_pose_led_depth, diag->best_any_match_flags,
	                (uint8_t)diag->best_any_leds_visible, (uint8_t)diag->best_any_blobs_matched,
	                (uint8_t)diag->best_any_unmatched_blobs, diag->best_any_reproj_err_px);
}

void
t_constellation_camera_group_dump_json(const struct t_constellation_camera_group *cams, FILE *f)
{
	if (cams == NULL || f == NULL) {
		return;
	}
	fprintf(f, "{\n");
	fprintf(f, "  \"format\": \"g2-constellation-cameras\",\n");
	fprintf(f, "  \"version\": 1,\n");
	fprintf(f, "  \"cam_count\": %d,\n", cams->cam_count);
	fprintf(f, "  \"cameras\": [\n");
	for (int i = 0; i < cams->cam_count; i++) {
		const struct t_constellation_camera *c = &cams->cams[i];
		const struct t_camera_calibration *cal = &c->calibration;
		fprintf(f, "    {\n");
		fprintf(f, "      \"index\": %d,\n", i);
		fprintf(f, "      \"slam_tracking_index\": %zu,\n", c->slam_tracking_index);
		fprintf(f, "      \"width\": %d,\n", cal->image_size_pixels.w);
		fprintf(f, "      \"height\": %d,\n", cal->image_size_pixels.h);
		fprintf(f,
		        "      \"intrinsics\": [[%.12g,%.12g,%.12g],[%.12g,%.12g,%.12g],[%.12g,%.12g,%.12g]],\n",
		        cal->intrinsics[0][0], cal->intrinsics[0][1], cal->intrinsics[0][2], cal->intrinsics[1][0],
		        cal->intrinsics[1][1], cal->intrinsics[1][2], cal->intrinsics[2][0], cal->intrinsics[2][1],
		        cal->intrinsics[2][2]);
		fprintf(f, "      \"distortion_model\": \"%s\",\n",
		        t_stringify_camera_distortion_model(cal->distortion_model));
		fprintf(f, "      \"distortion\": [");
		for (int k = 0; k < XRT_DISTORTION_MAX_DIM; k++) {
			fprintf(f, "%s%.12g", k ? "," : "", cal->distortion_parameters_as_array[k]);
		}
		fprintf(f, "],\n");
		fprintf(f,
		        "      \"P_imu_cam\": {\"position\": [%.12g,%.12g,%.12g], "
		        "\"orientation\": [%.12g,%.12g,%.12g,%.12g]},\n",
		        c->P_imu_cam.position.x, c->P_imu_cam.position.y, c->P_imu_cam.position.z,
		        c->P_imu_cam.orientation.x, c->P_imu_cam.orientation.y, c->P_imu_cam.orientation.z,
		        c->P_imu_cam.orientation.w);
		fprintf(f, "      \"roi\": {\"x\": %d, \"y\": %d, \"w\": %d, \"h\": %d},\n", c->roi.offset.w,
		        c->roi.offset.h, c->roi.extent.w, c->roi.extent.h);
		fprintf(f, "      \"blob_min_threshold\": %u, \"blob_detect_threshold\": %u, \"min_threshold\": %u\n",
		        c->blob_min_threshold, c->blob_detect_threshold, c->min_threshold);
		fprintf(f, "    }%s\n", (i + 1 < cams->cam_count) ? "," : "");
	}
	fprintf(f, "  ]\n}\n");
}

struct t_constellation_tracked_device_connection
{
	/* Device and tracker each hold a reference to the connection.
	 * It's only cleaned up once both release it. */
	struct xrt_reference ref;

	/* Index in the devices array for this device */
	int id;

	/* Protect access when around API calls and disconnects */
	struct os_mutex lock;
	bool disconnected; /* Set to true once disconnect() is called */

	// Callbacks to the tracked device
	struct xrt_device *xdev;
	struct t_constellation_tracked_device_callbacks *cb;

	struct t_constellation_tracker *tracker; //! Parent tracker instance
};

struct constellation_tracker_device
{
	struct t_constellation_tracked_device_connection *connection;

	bool have_led_model;
	struct t_constellation_led_model led_model;
	struct t_constellation_search_model *search_led_model;

	bool have_last_seen_pose;
	uint64_t last_seen_pose_ts;
	struct xrt_pose last_seen_pose; // global pose
	int last_matched_blobs;
	int last_matched_cam;
	struct xrt_pose last_matched_cam_pose; // Camera-relative pose

	/* Head-anchored yaw cue for the soft mirror-flip cost (Fix A): the last accepted controller
	 * orientation expressed in the HMD/head IMU frame, and its timestamp. Used as a SECOND prior
	 * (alongside the world-frame fusion prior) when ranking mirror twins on the next solve: a 180-deg
	 * flip is reflected in both world and head-relative yaw, so adding the head-relative yaw distance
	 * roughly doubles flip-vs-legit discrimination at the matcher gate. Stale by HEAD_YAW_CUE_TTL_NS;
	 * silently skipped when no fresh accept is available. */
	bool have_last_head_rel_quat;
	uint64_t last_head_rel_quat_ts;
	struct xrt_quat last_head_rel_quat; // controller orientation in head/IMU frame
};

struct constellation_tracker_camera_state
{
	//! Distortion params
	struct camera_model camera_model;
	//! ROI in the full frame mosaic
	struct xrt_rect roi;
	//! Camera's pose relative to the HMD GENERIC_TRACKER_POSE (IMU)
	struct xrt_pose P_imu_cam;

	//! Constellation tracking - fast tracking thread
	struct os_mutex bw_lock; /* Protects blobwatch process vs release from long thread */
	blobwatch *bw;
	int last_num_blobs;

	//! Pose-hypothesis search state used by the unified associator.
	struct correspondence_search *cs;

	//! Debug output
	struct u_sink_debug debug_sink;
	struct xrt_pose debug_last_pose;
	struct xrt_vec3 debug_last_gravity_vector;

	//! The index into the slam tracking camera array this camera represents
	size_t slam_tracking_index;
};

/*!
 * An @ref xrt_frame_sink that analyses video frame groups for LED constellation tracking
 * @implements xrt_frame_sink
 * @implements xrt_frame_node
 */
struct t_constellation_tracker
{
	//! Receive (mosaic) frames from the camera
	struct xrt_frame_sink base;
	//! frame node to insert in the xfctx
	struct xrt_frame_node node;

	/*! HMD device we get observation base poses from
	 * and that owns the xfctx keeping this node alive */
	struct xrt_device *hmd_xdev;

	struct os_mutex tracked_device_lock;

	//! Tracked device communication connections
	int num_devices;
	struct constellation_tracker_device devices[CONSTELLATION_MAX_DEVICES];

	//!< Tracking camera entries
	struct constellation_tracker_camera_state cam[XRT_TRACKING_MAX_SLAM_CAMS];
	int cam_count;

	/* Debug */
	enum u_logging_level log_level;
	bool debug_draw_normalise;
	bool debug_draw_blob_tint;
	bool debug_draw_blob_circles;
	bool debug_draw_blob_ids;
	bool debug_draw_blob_unique_ids;
	bool debug_draw_leds;
	bool debug_draw_prior_leds;
	bool debug_draw_last_leds;
	bool debug_draw_pose_bounds;
	bool debug_draw_device_bounds;

	uint64_t last_frame_timestamp;

	uint64_t last_fast_analysis_ms;
	uint64_t last_blob_analysis_ms;

	// Fast tracking thread
	struct xrt_frame_sink *fast_q_sink;
	struct xrt_frame_sink fast_process_sink;

	//! Frames fully processed through the pipeline. Lets the offline harness barrier on
	//! per-frame completion instead of racing a fixed sleep (debug/test only).
	atomic_uint_fast64_t frames_completed;

	struct xrt_device_masks_sample controller_masks_sample;
	struct xrt_device_masks_sink *controller_masks_sink;
};

static void
constellation_tracked_device_connection_notify_frame(struct t_constellation_tracked_device_connection *ctdc,
                                                     uint64_t frame_mono_ns,
                                                     uint64_t frame_sequence)
{
	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->notify_frame_received) {
		ctdc->cb->notify_frame_received(ctdc->xdev, frame_mono_ns, frame_sequence);
	}
	os_mutex_unlock(&ctdc->lock);
}

static void
constellation_tracked_device_connection_notify_pose(struct t_constellation_tracked_device_connection *ctdc,
                                                    timepoint_ns frame_mono_ns,
                                                    const struct xrt_pose *pose)
{
	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->push_observed_pose) {
		ctdc->cb->push_observed_pose(ctdc->xdev, frame_mono_ns, pose);
	}
	os_mutex_unlock(&ctdc->lock);
}

static void
constellation_tracked_device_connection_notify_position(struct t_constellation_tracked_device_connection *ctdc,
                                                        timepoint_ns frame_mono_ns,
                                                        const struct xrt_vec3 *position,
                                                        const struct xrt_vec3 *position_variance)
{
	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->push_observed_position) {
		ctdc->cb->push_observed_position(ctdc->xdev, frame_mono_ns, position, position_variance);
	}
	os_mutex_unlock(&ctdc->lock);
}

static void
constellation_tracked_device_connection_notify_leds(struct t_constellation_tracked_device_connection *ctdc,
                                                    timepoint_ns frame_mono_ns,
                                                    const struct xrt_pose *P_xrworld_cam,
                                                    const struct t_constellation_cam_calib *cam_calib,
                                                    const struct t_constellation_led_obs *leds,
                                                    size_t led_count)
{
	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->push_observed_leds) {
		ctdc->cb->push_observed_leds(ctdc->xdev, frame_mono_ns, P_xrworld_cam, cam_calib, leds, led_count);
	}
	os_mutex_unlock(&ctdc->lock);
}

/* Query the fusion's per-LED gate (zhat + 2x2 innovation covariance S) for one candidate blob<->LED
 * pairing. Returns false until the fusion is tracking, so partial fold is disabled at cold start. */
static bool
constellation_tracked_device_connection_predict_led_gate(struct t_constellation_tracked_device_connection *ctdc,
                                                         const struct xrt_pose *P_xrworld_cam,
                                                         const struct t_constellation_cam_calib *cam_calib,
                                                         const struct xrt_vec3 *led_obj,
                                                         float out_zhat[2],
                                                         float out_S[4])
{
	bool ret = false;

	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->predict_led_gate) {
		ret = ctdc->cb->predict_led_gate(ctdc->xdev, P_xrworld_cam, cam_calib, led_obj, out_zhat, out_S);
	}
	os_mutex_unlock(&ctdc->lock);

	return ret;
}

static void
constellation_tracked_device_connection_notify_brightness_update(struct t_constellation_tracked_device_connection *ctdc,
                                                                 uint8_t average_brightness)
{
	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->push_brightness_update) {
		ctdc->cb->push_brightness_update(ctdc->xdev, average_brightness);
	}
	os_mutex_unlock(&ctdc->lock);
}

static bool
constellation_tracked_device_connection_get_led_model(struct t_constellation_tracked_device_connection *ctdc,
                                                      struct t_constellation_led_model *led_model)
{
	bool ret = false;

	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->get_led_model) {
		ret = ctdc->cb->get_led_model(ctdc->xdev, led_model);
	}
	os_mutex_unlock(&ctdc->lock);

	return ret;
}

static bool
constellation_tracked_device_connection_get_pose_uncertainty(struct t_constellation_tracked_device_connection *ctdc,
                                                             double *position_std,
                                                             double *orientation_std,
                                                             double *yaw_std)
{
	bool ret = false;

	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->get_pose_uncertainty) {
		ret = ctdc->cb->get_pose_uncertainty(ctdc->xdev, position_std, orientation_std, yaw_std);
	}
	os_mutex_unlock(&ctdc->lock);

	return ret;
}

static bool
constellation_tracked_device_connection_get_tracked_pose(struct t_constellation_tracked_device_connection *ctdc,
                                                         uint64_t timestamp_ns,
                                                         struct xrt_space_relation *xsr)
{
	bool ret = false;

	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected) {
		struct xrt_device *xdev = ctdc->xdev;
		xrt_device_get_tracked_pose(xdev, XRT_INPUT_GENERIC_TRACKER_POSE, timestamp_ns, xsr);
		ret = true;
	}
	os_mutex_unlock(&ctdc->lock);

	return ret;
}

//! The fusion's RAW predicted prior (no body-lock/reach/re-entry), the honest estimate to gate + flip-cost
//! against. Optional callback; returns false if the device doesn't expose it (caller falls back to the
//! reported pose).
static bool
constellation_tracked_device_connection_get_predicted_pose(struct t_constellation_tracked_device_connection *ctdc,
                                                            uint64_t when_ns,
                                                            struct xrt_space_relation *xsr)
{
	bool ret = false;

	os_mutex_lock(&ctdc->lock);
	if (!ctdc->disconnected && ctdc->cb->get_predicted_pose) {
		ret = ctdc->cb->get_predicted_pose(ctdc->xdev, when_ns, xsr);
	}
	os_mutex_unlock(&ctdc->lock);

	return ret;
}

static void
constellation_tracker_receive_frame(struct xrt_frame_sink *sink, struct xrt_frame *xf)
{
	struct t_constellation_tracker *ct = container_of(sink, struct t_constellation_tracker, base);

	assert(xf->format == XRT_FORMAT_L8);

	// Tell the controllers about the frame so they can their timesync estimate
	os_mutex_lock(&ct->tracked_device_lock);
	for (int i = 0; i < ct->num_devices; i++) {
		constellation_tracked_device_connection_notify_frame(ct->devices[i].connection, xf->timestamp,
		                                                     xf->source_sequence);
	}
	os_mutex_unlock(&ct->tracked_device_lock);

	ct->last_frame_timestamp = xf->timestamp;
	xrt_sink_push_frame(ct->fast_q_sink, xf);
}

static void
constellation_tracker_node_break_apart(struct xrt_frame_node *node)
{
	DRV_TRACE_MARKER();
}

static void
mark_matching_blobs(struct t_constellation_tracker *ct,
                    struct xrt_pose *pose,
                    struct blobservation *bwobs,
                    struct t_constellation_led_model *led_model,
                    struct pose_metrics_blob_match_info *blob_match_info)
{
	/* First clear existing blob labels for this device */
	int i;
	for (i = 0; i < bwobs->num_blobs; i++) {
		struct blob *b = bwobs->blobs + i;
		uint32_t led_object_id = LED_OBJECT_ID(b->led_id);

		/* Skip blobs which already have an ID not belonging to this device */
		if (led_object_id != led_model->id) {
			continue;
		}

		if (b->led_id != LED_INVALID_ID) {
			b->prev_led_id = b->led_id;
		}
		b->led_id = LED_INVALID_ID;
	}


	/* Iterate the visible LEDs and mark matching blobs with this device ID and LED ID */
	for (i = 0; i < blob_match_info->num_visible_leds; i++) {
		struct pose_metrics_visible_led_info *led_info = blob_match_info->visible_leds + i;
		struct t_constellation_led *led = led_info->led;

		if (led_info->matched_blob != NULL) {
			struct blob *b = led_info->matched_blob;

			b->led_id = LED_MAKE_ID(led_model->id, led->id);
			CT_DEBUG(ct, "Marking LED %d/%d at %f,%f angle %f now %d (was %d)", led_model->id, led->id,
			         b->x, b->y, RAD_TO_DEG(acosf(led_info->facing_dot)), b->led_id, b->prev_led_id);
		} else {
			CT_DEBUG(ct, "No blob for device %d LED %d @ %f,%f size %f px angle %f", led_model->id, led->id,
			         led_info->pos_px.x, led_info->pos_px.y, 2 * led_info->led_radius_px,
			         RAD_TO_DEG(acosf(led_info->facing_dot)));
		}
	}
}

/* Feed this view's matched LEDs to the controller fusion as per-LED reprojection observations.
 * Caller must have matched dev_state->blob_match_info to the folded pose. Deduplicated per view via
 * led_emit_view_mask. Frames: led_obj = P_device_model . P_YZ(led->pos) (model->device, OpenCV->OpenXR)
 * and the extrinsic is P_cam_world(CV) . P_YZ, so the filter reproduces the constellation projection. */
static void
emit_view_led_observations(struct tracking_sample_device_state *dev_state,
                           struct constellation_tracker_device *device,
                           struct constellation_tracker_camera_state *cam,
                           struct tracking_sample_frame *view,
                           int view_id,
                           timepoint_ns sample_ts)
{
	if (view_id < 0 || view_id >= 16 || (dev_state->led_emit_view_mask & (1u << view_id)) != 0) {
		return; // out of range, or already folded this view this sample
	}

	struct t_constellation_led_obs led_obs[MAX_OBJECT_LEDS];
	int n_led_obs = 0;
	for (int i = 0; i < dev_state->blob_match_info.num_visible_leds; i++) {
		struct pose_metrics_visible_led_info *visible_led = &dev_state->blob_match_info.visible_leds[i];
		if (visible_led->matched_blob == NULL) {
			continue;
		}
		float nx = 0.f, ny = 0.f;
		t_camera_models_undistort(&cam->camera_model.calib, visible_led->matched_blob->x,
		                          visible_led->matched_blob->y, &nx, &ny);
		// Undistorted normalized ray -> undistorted PIXEL via the real pinhole intrinsics, so the
		// fusion reprojects in physical pixels and its noise/gate are focal-independent.
		led_obs[n_led_obs].obs_px =
		    (struct xrt_vec2){cam->camera_model.calib.fx * nx + cam->camera_model.calib.cx,
		                      cam->camera_model.calib.fy * ny + cam->camera_model.calib.cy};
		led_obs[n_led_obs].pos_var_px2 = visible_led->matched_blob->pos_var_px2;
		// led->pos is LED-MODEL frame; the fusion tracks the DEVICE pose, so map model->device after
		// the OpenCV->OpenXR YZ flip (mirrors the forward model flip(P_xrworld_device . P_device_model)).
		// Negating Y,Z is P_YZ_FLIP applied to a point (the explicit form is cheapest for a single point).
		struct xrt_vec3 led_flip = {visible_led->led->pos.x, -visible_led->led->pos.y,
		                            -visible_led->led->pos.z};
		math_pose_transform_point(&device->led_model.P_device_model, &led_flip,
		                          &led_obs[n_led_obs].led_obj);
		n_led_obs++;
	}
	if (n_led_obs == 0) {
		return;
	}

	struct xrt_pose P_xrworld_cam;
	math_pose_transform(&view->P_cam_world, &P_YZ_FLIP, &P_xrworld_cam);
	const struct t_constellation_cam_calib cam_calib = {cam->camera_model.calib.fx, cam->camera_model.calib.fy,
	                                                    cam->camera_model.calib.cx, cam->camera_model.calib.cy};
	constellation_tracked_device_connection_notify_leds(device->connection, sample_ts, &P_xrworld_cam,
	                                                    &cam_calib, led_obs, (size_t)n_led_obs);
	dev_state->led_emit_view_mask |= (1u << view_id);
}

static void
submit_device_pose(struct t_constellation_tracker *ct,
                   struct tracking_sample_device_state *dev_state,
                   struct constellation_tracking_sample *sample,
                   int view_id,
                   struct xrt_pose *P_cam_obj,
                   bool is_recovered)
{
	struct constellation_tracker_camera_state *cam = ct->cam + view_id;
	struct tracking_sample_frame *view = sample->views + view_id;
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct pose_metrics *score = &dev_state->score;

	mark_matching_blobs(ct, P_cam_obj, view->bwobs, &device->led_model, &dev_state->blob_match_info);

	os_mutex_lock(&cam->bw_lock);
	blobwatch_update_labels(cam->bw, view->bwobs, device->led_model.id);
	os_mutex_unlock(&cam->bw_lock);

	/* Telemetry: an accepted optical pose. Pose is camera-relative [p,q].
	 * A pose that came in via the labelled-blob re-acquisition path is marked
	 * outcome=2 (recovered) so analysts can distinguish recoveries from normal
	 * accepts; all others are outcome=1 (accepted). */
	if (g2_telem_enabled()) {
		uint8_t dev_id = telem_device_id(device->connection->xdev);
		bool is_new_lock = !dev_state->found_device_pose;
		float pose7[7];
		telem_pack_pose(P_cam_obj, pose7);
		g2_telem_pose_attempt(dev_id, (uint8_t)view_id, (uint64_t)sample->timestamp,
		                      (uint8_t)score->visible_leds, (uint8_t)score->matched_blobs,
		                      (uint8_t)score->matched_blobs, (float)score->reprojection_error, pose7,
		                      /* outcome */ is_recovered ? 2 /* recovered */ : 1 /* accepted */);
		/* First accepted view for this device in this sample == lock (re)acquired. */
		if (is_new_lock) {
			g2_telem_event(dev_id, (uint64_t)sample->timestamp, 1 /* lock_acquired */, 0.0f);
		}
	}

	/* Fold this view's matched LEDs into the fusion (tightly-coupled per-LED ESKF). Done per accepted
	 * view, independent of the single "winning view" last_seen bookkeeping below, so every view that
	 * matched contributes its LEDs (the helper's per-view mask dedups against the sub-threshold paths). */
	emit_view_led_observations(dev_state, device, cam, view, view_id, sample->timestamp);

	if (dev_state->found_device_pose) {
		return;
	}
	math_pose_transform(&view->P_world_cam, P_cam_obj, &dev_state->final_pose);
	dev_state->found_device_pose = true;
	dev_state->found_pose_view_id = view_id;

	os_mutex_lock(&ct->tracked_device_lock);
	if (device->have_last_seen_pose == false || sample->timestamp > device->last_seen_pose_ts) {
		device->have_last_seen_pose = true;
		device->last_seen_pose_ts = sample->timestamp;
		device->last_seen_pose = dev_state->final_pose;
		device->last_matched_blobs = score->matched_blobs;
		device->last_matched_cam = view_id;
		device->last_matched_cam_pose = *P_cam_obj;

		/* Cache controller-in-head orientation for the next frame's head-anchored flip cue
		 * (Fix A): P_imu_obj = P_imu_cam . P_cam_obj. The TTL on the consumer side keeps a
		 * stale cache from biasing twin selection across dropouts. */
		struct xrt_pose P_imu_obj_cache;
		math_pose_transform(&cam->P_imu_cam, P_cam_obj, &P_imu_obj_cache);
		device->last_head_rel_quat = P_imu_obj_cache.orientation;
		device->last_head_rel_quat_ts = sample->timestamp;
		device->have_last_head_rel_quat = true;

		/* Submit this pose observation to the fusion / real device. Flip back to OpenXR coords first, then
		 * apply model pose */
		struct xrt_pose P_xrworld_model;
		pose_flip_YZ(&dev_state->final_pose, &P_xrworld_model);

		// Apply device -> LED model pose from xsr = P_world_device + P_device_model = model pose
		struct xrt_pose P_xrworld_device;
		math_pose_transform(&P_xrworld_model, &device->led_model.P_model_device, &P_xrworld_device);

		// Average matched-blob brightness for the LED-intensity / brightness feedback. (The per-LED
		// fusion feed is emitted by emit_view_led_observations above, per view.)
		uint32_t average_brightness = 0;
		int matched_blobs = 0;
		for (int i = 0; i < dev_state->blob_match_info.num_visible_leds; i++) {
			struct pose_metrics_visible_led_info *visible_led = &dev_state->blob_match_info.visible_leds[i];
			if (visible_led->matched_blob) {
				average_brightness += visible_led->matched_blob->brightness;
				matched_blobs++;
			}
		}

		if (matched_blobs > 0) {
			average_brightness /= matched_blobs;
			constellation_tracked_device_connection_notify_brightness_update(device->connection,
			                                                                 average_brightness);
		}

		constellation_tracked_device_connection_notify_pose(device->connection, sample->timestamp,
		                                                    &P_xrworld_device);

		// update the controller masks for this controller
		if (ct->controller_masks_sink) {
			struct xrt_pose P_imu_obj;
			math_pose_transform(&ct->cam[view_id].P_imu_cam, P_cam_obj, &P_imu_obj);

			for (int i = 0; i < ct->cam_count; i++) {
				struct constellation_tracker_camera_state *cam = &ct->cam[i];

				struct xrt_device_masks_sample_camera *sample_camera =
				    &ct->controller_masks_sample.views[cam->slam_tracking_index];

				struct xrt_device_masks_sample_device *device_mask =
				    &sample_camera->devices[dev_state->dev_index];

				struct xrt_pose P_tcam_imu;
				math_pose_invert(&cam->P_imu_cam, &P_tcam_imu);

				struct xrt_pose P_tcam_obj;
				math_pose_transform(&P_tcam_imu, &P_imu_obj, &P_tcam_obj);

				struct pose_rect device_bounds;
				pose_metrics_get_device_bounds(&P_tcam_obj, &device->led_model, &cam->camera_model,
				                               &device_bounds, NULL, NULL);

				device_mask->enabled = pose_rect_has_area(&device_bounds);
				if (device_mask->enabled) {
					device_mask->rect = (struct xrt_rect_f32){
					    .x = device_bounds.left,
					    .y = device_bounds.top,
					    .w = device_bounds.right - device_bounds.left,
					    .h = device_bounds.bottom - device_bounds.top,
					};
				}
			}

			xrt_sink_push_device_masks(ct->controller_masks_sink, &ct->controller_masks_sample);
		}
	}
	os_mutex_unlock(&ct->tracked_device_lock);
}

/* Pose-predicted LED label propagation: project the fusion's PREDICTED controller pose's LED
 * model into one view and assign each blob to the LED it lands on, back-face culled by the LED normals
 * and bounded by the prior's anisotropic per-LED gate (all inside pose_metrics_match_pose_to_blobs).
 * This is the PROJECTED-LED-motion prior — the labels follow the predicted pose through head and
 * controller motion — NOT raw pixel velocity, which parallax and ego-motion make unreliable. Returns
	 * the count of blobs newly labelled to this device, so the caller can decide a view carries enough
	 * propagated IDs to solve. The label transfer
 * itself is mark_matching_blobs; this just wraps the project+label so the joint and single-cam paths
 * seed labels identically (one source of truth for "label a view from the predicted pose"). */
static int
device_propagate_labels_in_view(struct t_constellation_tracker *ct,
                                struct tracking_sample_device_state *dev_state,
                                struct constellation_tracking_sample *sample,
                                int view_id)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct tracking_sample_frame *view = sample->views + view_id;
	struct constellation_tracker_camera_state *cam = ct->cam + view_id;

	if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
		return 0;
	}

	struct xrt_pose P_cam_obj_prior;
	math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);

	if (dev_state->prior_tilt_trusted) {
		pose_metrics_match_pose_to_blobs_prior(&P_cam_obj_prior, view->bwobs->blobs, view->bwobs->num_blobs,
		                                       &dev_state->prior_pos_error, &dev_state->prior_rot_error,
		                                       &device->led_model, &cam->camera_model,
		                                       &dev_state->blob_match_info);
	} else {
		pose_metrics_match_pose_to_blobs(&P_cam_obj_prior, view->bwobs->blobs, view->bwobs->num_blobs,
		                                 &device->led_model, &cam->camera_model,
		                                 &dev_state->blob_match_info);
	}
	mark_matching_blobs(ct, &P_cam_obj_prior, view->bwobs, &device->led_model, &dev_state->blob_match_info);

	int n_labelled = 0;
	for (int i = 0; i < dev_state->blob_match_info.num_visible_leds; i++) {
		if (dev_state->blob_match_info.visible_leds[i].matched_blob != NULL) {
			n_labelled++;
		}
	}
	return n_labelled;
}

/* Partial-information fold with a covariance gate.
 *
 * When no pose hypothesis is reliable enough to lock, this folds the few LEDs that are confidently
 * matched to the prior, without committing a pose: for each prior-visible LED we
 * ask the fusion for its predicted image point + 2x2 innovation covariance S = H·P·Hᵀ + R
 * (predict_led_gate), then accept the blob whose Mahalanobis distance d²=rᵀS⁻¹r is smallest AND
 * ≤ χ²₂(0.99). Only in-gate LEDs are folded (the ESKF then grows covariance honestly on 1-3 LEDs).
 *
 * Guards (the contract):
	 *  - COLD START / no prior: predict_led_gate returns false until the fusion is tracking, AND we require
	 *    prior_tilt_trusted (= the fusion is tracking, gravity-anchored prior available). Untracked => no-op.
 *  - MISLABEL: the anisotropic S is the guard. A flipped/garbage correspondence reprojects far from
 *    zhat (in tilt especially — S is tight there), so d² blows past the gate and the LED is NOT folded.
 *    The per-LED χ² gate inside fold_led_observations is the second, consistent line of defence.
 *  - Each blob is assigned to at most one LED: LED-order greedy (LEDs scanned in index order, each claims
 *    its min-d² of the still-free in-gate blobs; blob_taken[] enforces one blob per LED and one LED per
 *    blob). Deterministic; harmless with 1-3 sparse LEDs (each fold is re-gated by the ESKF's own χ²).
 *
 * Does NOT report a pose (no submit_device_pose, no PnP "accept"); it only feeds the filter, so the
	 * device keeps reporting its covariance-grown prior while cold-search candidates handle genuine
	 * (re)acquire. Returns true iff at least one LED was gate-folded (diagnostic). A <4-LED frame is
	 * used productively here; the accept/flip decision stays with the anisotropic prior-cost paths (the
	 * cold-search accept is flip-ranked by the same prior split when a prior exists). */
static bool
association_fold_prior_leds(struct t_constellation_tracker *ct,
                            struct tracking_sample_device_state *dev_state,
                            struct constellation_tracking_sample *sample)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;

		/* No reliable prior to gate with (cold start) -> no-op. The covariance gate
		 * needs a tracking filter (predict_led_gate returns false otherwise); prior_tilt_trusted is exactly
	 * "the fusion is tracking" (the gravity-anchored prior is available, even through a dropout). The
	 * anisotropic S handles tilt/yaw weighting itself, so no yaw-trust requirement here. */
	if (!dev_state->prior_tilt_trusted) {
		return false;
	}
	bool folded_any = false;

	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		struct constellation_tracker_camera_state *cam = ct->cam + view_id;

		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}
		blobservation *bwobs = view->bwobs;

		struct xrt_pose P_cam_obj_prior;
		math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);

		/* Enumerate the prior-visible (front-facing, in-frame) LEDs for this view. We do NOT use the
		 * fixed-radius matched_blob it fills — the covariance gate below makes the real association. */
		pose_metrics_match_pose_to_blobs(&P_cam_obj_prior, bwobs->blobs, bwobs->num_blobs, &device->led_model,
		                                 &cam->camera_model, &dev_state->blob_match_info);

		/* The extrinsic + intrinsics the fusion gate must use (identical to emit_view_led_observations). */
		struct xrt_pose P_xrworld_cam;
		math_pose_transform(&view->P_cam_world, &P_YZ_FLIP, &P_xrworld_cam);
		const struct t_constellation_cam_calib cam_calib = {
		    cam->camera_model.calib.fx, cam->camera_model.calib.fy, cam->camera_model.calib.cx,
		    cam->camera_model.calib.cy};

		/* Precompute each blob's undistorted PIXEL position once (matches the gate's zhat units). */
		struct xrt_vec2 blob_px[MAX_BLOBS_PER_FRAME];
		bool blob_taken[MAX_BLOBS_PER_FRAME] = {false};
		const int num_blobs = bwobs->num_blobs < MAX_BLOBS_PER_FRAME ? bwobs->num_blobs : MAX_BLOBS_PER_FRAME;
		for (int b = 0; b < num_blobs; b++) {
			float nx = 0.f, ny = 0.f;
			t_camera_models_undistort(&cam->camera_model.calib, bwobs->blobs[b].x, bwobs->blobs[b].y, &nx, &ny);
			blob_px[b].x = cam->camera_model.calib.fx * nx + cam->camera_model.calib.cx;
			blob_px[b].y = cam->camera_model.calib.fy * ny + cam->camera_model.calib.cy;
		}

		int n_matched = 0;
		for (int i = 0; i < dev_state->blob_match_info.num_visible_leds; i++) {
			struct pose_metrics_visible_led_info *visible_led = &dev_state->blob_match_info.visible_leds[i];
			visible_led->matched_blob = NULL; /* covariance gate decides; ignore the radius match */

			/* led_obj in the OpenXR object frame, exactly as emit_view_led_observations builds it. */
			struct xrt_vec3 led_flip = {visible_led->led->pos.x, -visible_led->led->pos.y,
			                            -visible_led->led->pos.z};
			struct xrt_vec3 led_obj;
			math_pose_transform_point(&device->led_model.P_device_model, &led_flip, &led_obj);

			float zhat[2], S[4];
			if (!constellation_tracked_device_connection_predict_led_gate(
			        device->connection, &P_xrworld_cam, &cam_calib, &led_obj, zhat, S)) {
				/* Untracked / non-finite -> no usable gate for any LED this frame: stop (cold start). */
				return false;
			}

			/* S = [[s0,s1],[s2,s3]]; S^-1 = 1/det [[s3,-s1],[-s2,s0]]. d² = rᵀ S⁻¹ r. */
			const double det = (double)S[0] * S[3] - (double)S[1] * S[2];
			if (!(det > 1e-9)) {
				continue; /* degenerate covariance -> skip this LED (no fold) */
			}
			const double inv00 = S[3] / det, inv01 = -S[1] / det, inv10 = -S[2] / det, inv11 = S[0] / det;

			int best_b = -1;
			double best_d2 = PARTIAL_FOLD_CHI2_2DOF;
			for (int b = 0; b < num_blobs; b++) {
				if (blob_taken[b]) {
					continue;
				}
				/* Skip blobs already labelled to ANOTHER device (unlabelled == LED_INVALID_ID is OK). */
				const uint16_t bid = bwobs->blobs[b].led_id;
				if (bid != LED_INVALID_ID && LED_OBJECT_ID(bid) != device->led_model.id) {
					continue;
				}
				const double rx = (double)blob_px[b].x - zhat[0];
				const double ry = (double)blob_px[b].y - zhat[1];
				const double d2 = rx * (inv00 * rx + inv01 * ry) + ry * (inv10 * rx + inv11 * ry);
				if (d2 < best_d2) {
					best_d2 = d2;
					best_b = b;
				}
			}
			if (best_b >= 0) {
				visible_led->matched_blob = &bwobs->blobs[best_b];
				blob_taken[best_b] = true;
				n_matched++;
			}
		}

		/* Fold ONLY the covariance-gated LEDs (each re-gated by the fusion's own per-LED χ²). One or two
		 * gated LEDs are enough to keep the filter informed without committing a (flip-prone) pose. */
		if (n_matched > 0) {
			emit_view_led_observations(dev_state, device, cam, view, view_id, sample->timestamp);
			folded_any = true;
			if (g2_telem_enabled()) {
				g2_telem_event(telem_device_id(device->connection->xdev), (uint64_t)sample->timestamp,
				               9 /* partial_fold_count */, (float)n_matched);
			}
		}
	}

	return folded_any; /* true == at least one LED was gate-folded this frame (diagnostic) */
}

#define ASSOC_CLUTTER_NLL 0.2f
#define ASSOC_POS_PRIOR_WEIGHT 0.25f
#define ASSOC_VISUAL_LOCK_COST 10.0f
#define ASSOC_VISUAL_AMBIG_COST 20.0f
#define ASSOC_ABSENT_WITH_BLOBS_COST 14.0f
#define ASSOC_EV_LOCKABLE_NOT_CHOSEN 12
#define ASSOC_EV_LOCK_COMMIT_FAILED 13
#define ASSOC_EV_POSITION_ONLY_SELECTED 14
#define ASSOC_MATCH_ALL_BLOBS_MAX 16
#define ASSOC_POSITION_ONLY_MIN_MATCHED 4
#define ASSOC_POSITION_ONLY_MAX_REPROJ_PER_LED 3.0f
#define ASSOC_POSITION_ONLY_ACTION_NLL 4.0f
#define ASSOC_LED_FOLD_ACTION_NLL 7.0f
#define ASSOC_POSITION_ONLY_BASE_STD_M 0.06f
#define ASSOC_POSITION_ONLY_PER_REPROJ_STD_M 0.02f
#define ASSOC_POSITION_ONLY_SINGLE_VIEW_INFLATE_M 0.04f
#define ASSOC_POSITION_ONLY_FEW_LED_INFLATE_M 0.03f
#define ASSOC_POSITION_ONLY_MIN_STD_M 0.04f
#define ASSOC_POSITION_ONLY_MAX_STD_M 0.20f

enum association_observation_kind
{
	ASSOC_OBS_ABSENT = 0,
	ASSOC_OBS_POSE_LOCK = 1,
	ASSOC_OBS_POSITION_ONLY = 2,
	ASSOC_OBS_LED_FOLD = 3,
};

struct association_device_work
{
	struct association_pose_hypothesis hyps[ASSOCIATION_MAX_HYPOTHESES_PER_DEVICE];
	int count;
	bool has_blobs;
	bool folded_partial;
};

struct association_joint_choice
{
	const struct association_pose_hypothesis *chosen[CONSTELLATION_MAX_DEVICES];
	enum association_observation_kind kind[CONSTELLATION_MAX_DEVICES];
	float total_cost;
	int visual_count;
	bool valid;
};

struct association_blob_label_snapshot
{
	uint16_t led_id[MAX_BLOBS_PER_FRAME];
	uint16_t prev_led_id[MAX_BLOBS_PER_FRAME];
	int count;
};

static int
association_count_labelled_blobs(const blobservation *bwobs, uint16_t model_id)
{
	int n = 0;
	for (int i = 0; bwobs != NULL && i < bwobs->num_blobs; i++) {
		if (LED_OBJECT_ID(bwobs->blobs[i].led_id) == model_id) {
			n++;
		}
	}
	return n;
}

static void
association_save_labels(const blobservation *bwobs, struct association_blob_label_snapshot *snapshot)
{
	snapshot->count = bwobs != NULL && bwobs->num_blobs < MAX_BLOBS_PER_FRAME ? bwobs->num_blobs
	                                                                           : MAX_BLOBS_PER_FRAME;
	for (int i = 0; i < snapshot->count; i++) {
		snapshot->led_id[i] = bwobs->blobs[i].led_id;
		snapshot->prev_led_id[i] = bwobs->blobs[i].prev_led_id;
	}
}

static void
association_restore_labels(blobservation *bwobs, const struct association_blob_label_snapshot *snapshot)
{
	for (int i = 0; bwobs != NULL && i < snapshot->count && i < bwobs->num_blobs; i++) {
		bwobs->blobs[i].led_id = snapshot->led_id[i];
		bwobs->blobs[i].prev_led_id = snapshot->prev_led_id[i];
	}
}

static double
association_quat_angle(const struct xrt_quat *a, const struct xrt_quat *b)
{
	double d = fabs((double)a->x * b->x + (double)a->y * b->y + (double)a->z * b->z + (double)a->w * b->w);
	if (d > 1.0) {
		d = 1.0;
	}
	return 2.0 * acos(d);
}

static bool
association_pose_duplicate(const struct association_pose_hypothesis *a,
                           const struct association_pose_hypothesis *b)
{
	struct xrt_vec3 dp = m_vec3_sub(a->pose_world.position, b->pose_world.position);
	return m_vec3_len(dp) < 0.02 && association_quat_angle(&a->pose_world.orientation, &b->pose_world.orientation) <
	                                  DEG_TO_RAD(2.0);
}

static int
association_distinct_view_count(const struct association_pose_hypothesis *hyp)
{
	uint16_t mask = 0;
	for (uint8_t i = 0; hyp != NULL && i < hyp->matched_count; i++) {
		const int view_id = hyp->matched_blobs[i].view_id;
		if (view_id >= 0 && view_id < 16) {
			mask |= (uint16_t)(1u << view_id);
		}
	}

	int count = 0;
	for (; mask != 0; mask = (uint16_t)(mask & (mask - 1))) {
		count++;
	}
	return count;
}

static float
association_axis_nll(float value, float sigma)
{
	if (!(sigma > 0.0f) || !isfinite(sigma)) {
		return 0.0f;
	}
	const float s = value / sigma;
	return 0.5f * s * s;
}

static float
association_position_prior_nll(const struct pose_metrics *score,
                               const struct tracking_sample_device_state *dev_state)
{
	if (!POSE_HAS_FLAGS(score, POSE_HAD_PRIOR)) {
		return 0.0f;
	}
	return ASSOC_POS_PRIOR_WEIGHT *
	       (association_axis_nll((float)score->pos_error.x, dev_state->prior_pos_error.x) +
	        association_axis_nll((float)score->pos_error.y, dev_state->prior_pos_error.y) +
	        association_axis_nll((float)score->pos_error.z, dev_state->prior_pos_error.z));
}

static float
association_orientation_prior_nll(const struct tracking_sample_device_state *dev_state,
                                  const struct xrt_pose *candidate,
                                  const struct xrt_pose *prior,
                                  const struct xrt_vec3 *up)
{
	if (!dev_state->prior_tilt_trusted) {
		return 0.0f;
	}
	return (float)pose_metrics_prior_orient_cost(&candidate->orientation, &prior->orientation, up,
	                                            GRAVITY_TILT_TOL, dev_state->prior_yaw_sigma_rad,
	                                            FLIP_COST_HUBER_KNEE_SIGMA, FLIP_COST_WEIGHT);
}

static float
association_head_anchor_nll(const struct constellation_tracker_device *device,
                            const struct constellation_tracker_camera_state *cam,
                            const struct tracking_sample_frame *view,
                            const struct constellation_tracking_sample *sample,
                            const struct xrt_pose *P_cam_obj)
{
	if (!device->have_last_head_rel_quat || sample->timestamp < device->last_head_rel_quat_ts ||
	    sample->timestamp - device->last_head_rel_quat_ts >= (uint64_t)HEAD_YAW_CUE_TTL_NS) {
		return 0.0f;
	}

	struct xrt_pose P_imu_obj;
	math_pose_transform(&cam->P_imu_cam, P_cam_obj, &P_imu_obj);

	struct xrt_vec3 imu_gravity;
	math_quat_rotate_vec3(&cam->P_imu_cam.orientation, &view->cam_gravity_vector, &imu_gravity);

	return (float)pose_metrics_prior_orient_cost(&P_imu_obj.orientation, &device->last_head_rel_quat,
	                                            &imu_gravity, HEAD_YAW_CUE_TILT_SIGMA,
	                                            HEAD_YAW_CUE_YAW_SIGMA, FLIP_COST_HUBER_KNEE_SIGMA,
	                                            FLIP_COST_WEIGHT);
}

static float
association_prior_pose_nll(const struct constellation_tracker_device *device,
                           const struct constellation_tracker_camera_state *cam,
                           const struct tracking_sample_device_state *dev_state,
                           const struct tracking_sample_frame *view,
                           const struct constellation_tracking_sample *sample,
                           const struct xrt_pose *P_cam_obj)
{
	struct xrt_pose P_cam_obj_prior;
	math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);
	struct xrt_vec3 dp = m_vec3_sub(P_cam_obj->position, P_cam_obj_prior.position);
	return ASSOC_POS_PRIOR_WEIGHT *
	           (association_axis_nll(dp.x, dev_state->prior_pos_error.x) +
	            association_axis_nll(dp.y, dev_state->prior_pos_error.y) +
	            association_axis_nll(dp.z, dev_state->prior_pos_error.z)) +
	       association_orientation_prior_nll(dev_state, P_cam_obj, &P_cam_obj_prior, &view->cam_gravity_vector) +
	       association_head_anchor_nll(device, cam, view, sample, P_cam_obj);
}

static void
association_emit_candidate(const struct constellation_tracker_device *device,
                           const struct tracking_sample_device_state *dev_state,
                           const struct tracking_sample_frame *view,
                           int view_id,
                           uint64_t timestamp_ns,
                           const struct association_pose_hypothesis *hyp,
                           bool selected,
                           uint8_t outcome)
{
	if (hyp == NULL || view == NULL) {
		return;
	}
	struct xrt_pose P_cam_obj_prior;
	math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);

	double tilt_rad = 0.0;
	double yaw_rad = 0.0;
	if (dev_state->prior_tilt_trusted) {
		pose_metrics_prior_orient_split(&hyp->pose_cam.orientation, &P_cam_obj_prior.orientation,
		                                &view->cam_gravity_vector, &tilt_rad, &yaw_rad);
	}

	const float prior_pos_err[3] = {
	    (float)hyp->score.pos_error.x,
	    (float)hyp->score.pos_error.y,
	    (float)hyp->score.pos_error.z,
	};
	const float prior_rot_err[3] = {
	    (float)hyp->score.orient_error.x,
	    (float)hyp->score.orient_error.y,
	    (float)hyp->score.orient_error.z,
	};
	float pose[7];
	telem_pack_pose(&hyp->pose_cam, pose);
	const struct xrt_device *xdev = device != NULL && device->connection != NULL ? device->connection->xdev : NULL;
	const float prior_cost = hyp->cost.position_prior_nll + hyp->cost.orientation_prior_nll +
	                         hyp->cost.head_anchor_nll + hyp->cost.body_state_nll;
	g2_telem_candidate(telem_device_id(xdev), (uint8_t)view_id, timestamp_ns, hyp->source,
	                   (hyp->flags & ASSOC_HYP_IS_TWIN) ? 1 : 0, selected ? 1 : 0,
	                   (hyp->flags & ASSOC_HYP_HAS_TWIN) ? 1 : 0, outcome, hyp->score.match_flags,
	                   (uint8_t)hyp->visible_count, (uint8_t)hyp->matched_count,
	                   (uint8_t)hyp->unmatched_count, (uint8_t)hyp->matched_count,
	                   (float)hyp->score.reprojection_error, prior_cost, hyp->cost.total_nll,
	                   dev_state->prior_tilt_trusted ? 1 : 0, dev_state->prior_yaw_sigma_rad,
	                   (float)tilt_rad, (float)yaw_rad, prior_pos_err, prior_rot_err, pose);
}

static int
association_blob_index(const blobservation *bwobs, const struct blob *blob)
{
	if (bwobs == NULL || blob == NULL) {
		return -1;
	}
	for (int i = 0; i < bwobs->num_blobs; i++) {
		if (&bwobs->blobs[i] == blob) {
			return i;
		}
	}
	return -1;
}

static int
association_fill_blob_refs(struct association_pose_hypothesis *hyp,
                           const struct pose_metrics_blob_match_info *match_info,
                           const blobservation *bwobs,
                           int view_id)
{
	int added = 0;
	for (int i = 0; i < match_info->num_visible_leds; i++) {
		const struct pose_metrics_visible_led_info *visible = &match_info->visible_leds[i];
		if (visible->matched_blob == NULL || visible->led == NULL) {
			continue;
		}
		const int blob_idx = association_blob_index(bwobs, visible->matched_blob);
		if (blob_idx < 0) {
			continue;
		}
		const uint8_t before = hyp->matched_count;
		association_hypothesis_add_blob(hyp, (int16_t)view_id, (int16_t)blob_idx, (int16_t)visible->led->id);
		added += hyp->matched_count > before ? 1 : 0;
	}
	return added;
}

static struct t_constellation_led *
association_find_led(struct t_constellation_led_model *led_model, int led_id)
{
	for (uint8_t i = 0; i < led_model->num_leds; i++) {
		if (led_model->leds[i].id == led_id) {
			return &led_model->leds[i];
		}
	}
	return NULL;
}

static int
association_apply_hypothesis_matches(struct tracking_sample_device_state *dev_state,
                                     struct constellation_tracker_device *device,
                                     struct tracking_sample_frame *view,
                                     const struct association_pose_hypothesis *hyp,
                                     int view_id)
{
	memset(&dev_state->blob_match_info, 0, sizeof(dev_state->blob_match_info));
	dev_state->blob_match_info.all_led_ids_matched = true;
	dev_state->blob_match_info.matched_blobs = hyp->matched_count;
	dev_state->blob_match_info.unmatched_blobs = hyp->unmatched_count;
	dev_state->blob_match_info.reprojection_error = hyp->cost.reprojection_nll * (double)hyp->matched_count;

	int written = 0;
	for (uint8_t i = 0; i < hyp->matched_count && written < MAX_OBJECT_LEDS; i++) {
		const struct association_blob_ref *ref = &hyp->matched_blobs[i];
		if (!association_blob_ref_is_valid(ref) || ref->view_id != view_id || ref->blob_id < 0 ||
		    ref->blob_id >= view->bwobs->num_blobs) {
			continue;
		}
		struct t_constellation_led *led = association_find_led(&device->led_model, hyp->matched_led_ids[i]);
		if (led == NULL) {
			continue;
		}
		struct pose_metrics_visible_led_info *dst = &dev_state->blob_match_info.visible_leds[written++];
		dst->led = led;
		dst->matched_blob = &view->bwobs->blobs[ref->blob_id];
		dst->pos_px = (struct xrt_vec2){dst->matched_blob->x, dst->matched_blob->y};
	}
	dev_state->blob_match_info.num_visible_leds = written;
	dev_state->blob_match_info.matched_blobs = written;
	return written;
}

static bool
association_hypothesis_has_view(const struct association_pose_hypothesis *hyp, int view_id)
{
	for (uint8_t i = 0; hyp != NULL && i < hyp->matched_count; i++) {
		if (hyp->matched_blobs[i].view_id == view_id) {
			return true;
		}
	}
	return false;
}

static int
association_fold_hypothesis_view(struct t_constellation_tracker *ct,
                                 struct tracking_sample_device_state *dev_state,
                                 struct constellation_tracking_sample *sample,
                                 const struct association_pose_hypothesis *hyp,
                                 int view_id,
                                 bool update_labels)
{
	if (hyp == NULL || view_id < 0 || view_id >= sample->n_views ||
	    !association_hypothesis_has_view(hyp, view_id)) {
		return 0;
	}
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct tracking_sample_frame *view = sample->views + view_id;
	struct constellation_tracker_camera_state *cam = ct->cam + view_id;
	if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
		return 0;
	}
	const int selected_matches = association_apply_hypothesis_matches(dev_state, device, view, hyp, view_id);
	if (selected_matches <= 0) {
		return 0;
	}
	if (update_labels) {
		struct xrt_pose P_cam_obj;
		math_pose_transform(&view->P_cam_world, &hyp->pose_world, &P_cam_obj);
		mark_matching_blobs(ct, &P_cam_obj, view->bwobs, &device->led_model, &dev_state->blob_match_info);
		os_mutex_lock(&cam->bw_lock);
		blobwatch_update_labels(cam->bw, view->bwobs, device->led_model.id);
		os_mutex_unlock(&cam->bw_lock);
	}
	emit_view_led_observations(dev_state, device, cam, view, view_id, sample->timestamp);
	return selected_matches;
}

static bool
association_refine_multiview_pose(struct t_constellation_tracker *ct,
                                  struct tracking_sample_device_state *dev_state,
                                  struct constellation_tracking_sample *sample,
                                  const struct association_pose_hypothesis *hyp,
                                  struct xrt_pose *out_primary_cam_pose)
{
	if (hyp == NULL || out_primary_cam_pose == NULL || association_distinct_view_count(hyp) < 2) {
		return false;
	}

	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct blob view_blobs[JOINT_PNP_MAX_VIEWS][ASSOCIATION_MAX_BLOBS_PER_HYPOTHESIS];
	struct joint_pnp_view jviews[JOINT_PNP_MAX_VIEWS];
	int view_ids[JOINT_PNP_MAX_VIEWS];
	int view_counts[JOINT_PNP_MAX_VIEWS] = {0};
	int n_jviews = 0;

	for (uint8_t i = 0; i < hyp->matched_count; i++) {
		const struct association_blob_ref *ref = &hyp->matched_blobs[i];
		if (!association_blob_ref_is_valid(ref) || ref->view_id < 0 || ref->view_id >= sample->n_views ||
		    hyp->matched_led_ids[i] < 0 || hyp->matched_led_ids[i] >= device->led_model.num_leds) {
			continue;
		}
		struct tracking_sample_frame *view = sample->views + ref->view_id;
		if (view->bwobs == NULL || ref->blob_id < 0 || ref->blob_id >= view->bwobs->num_blobs) {
			continue;
		}

		int slot = -1;
		for (int j = 0; j < n_jviews; j++) {
			if (view_ids[j] == ref->view_id) {
				slot = j;
				break;
			}
		}
		if (slot < 0) {
			if (n_jviews >= JOINT_PNP_MAX_VIEWS) {
				continue;
			}
			slot = n_jviews++;
			view_ids[slot] = ref->view_id;
		}
		if (view_counts[slot] >= ASSOCIATION_MAX_BLOBS_PER_HYPOTHESIS) {
			continue;
		}
		struct blob labelled_blob = view->bwobs->blobs[ref->blob_id];
		labelled_blob.led_id = LED_MAKE_ID(device->led_model.id, hyp->matched_led_ids[i]);
		view_blobs[slot][view_counts[slot]++] = labelled_blob;
	}

	int contributing_views = 0;
	for (int j = 0; j < n_jviews; j++) {
		if (view_counts[j] == 0) {
			continue;
		}
		struct constellation_tracker_camera_state *cam = ct->cam + view_ids[j];
		jviews[contributing_views].blobs = view_blobs[j];
		jviews[contributing_views].num_blobs = view_counts[j];
		jviews[contributing_views].calib = &cam->camera_model;
		jviews[contributing_views].P_imu_cam = cam->P_imu_cam;
		contributing_views++;
	}
	if (contributing_views < 2) {
		return false;
	}

	struct xrt_pose P_imu_obj = hyp->pose_imu;
	int num_rays = 0;
	int num_inliers = 0;
	if (!joint_pnp_solve(&P_imu_obj, jviews, contributing_views, &device->led_model, &num_rays, &num_inliers)) {
		return false;
	}

	struct xrt_pose P_cam_imu;
	math_pose_invert(&ct->cam[hyp->primary_view_id].P_imu_cam, &P_cam_imu);
	math_pose_transform(&P_cam_imu, &P_imu_obj, out_primary_cam_pose);
	return true;
}

static bool
association_hypothesis_less(const struct association_pose_hypothesis *a,
                            const struct association_pose_hypothesis *b)
{
	if (a->cost.total_nll != b->cost.total_nll) {
		return a->cost.total_nll < b->cost.total_nll;
	}
	if (a->matched_count != b->matched_count) {
		return a->matched_count > b->matched_count;
	}
	return a->cost.reprojection_nll < b->cost.reprojection_nll;
}

static bool
association_lock_eligible(const struct association_pose_hypothesis *hyp)
{
	if (hyp == NULL || hyp->matched_count < 4 || !POSE_HAS_FLAGS(&hyp->score, POSE_MATCH_GOOD) ||
	    (hyp->flags & ASSOC_HYP_PARTIAL_ONLY) != 0 || hyp->cost.total_nll >= ASSOC_VISUAL_LOCK_COST) {
		return false;
	}

	return true;
}

static double
association_reprojection_per_match(const struct association_pose_hypothesis *hyp)
{
	return hyp != NULL && hyp->matched_count > 0 ? hyp->score.reprojection_error / (double)hyp->matched_count
	                                             : INFINITY;
}

static bool
association_position_only_eligible(const struct association_pose_hypothesis *hyp)
{
	if (hyp == NULL || (hyp->flags & ASSOC_HYP_PARTIAL_ONLY) != 0 ||
	    hyp->matched_count < ASSOC_POSITION_ONLY_MIN_MATCHED) {
		return false;
	}

	if (association_reprojection_per_match(hyp) > ASSOC_POSITION_ONLY_MAX_REPROJ_PER_LED) {
		return false;
	}

	return POSE_HAS_FLAGS(&hyp->score, POSE_MATCH_POSITION) || association_distinct_view_count(hyp) >= 2 ||
	       hyp->matched_count >= 6;
}

static bool
association_led_fold_eligible(const struct association_pose_hypothesis *hyp)
{
	return hyp != NULL && hyp->matched_count > 0 && hyp->cost.total_nll < ASSOC_VISUAL_AMBIG_COST;
}

static float
association_position_observation_std_m(const struct association_pose_hypothesis *hyp)
{
	float std_m = ASSOC_POSITION_ONLY_BASE_STD_M;
	std_m += (float)association_reprojection_per_match(hyp) * ASSOC_POSITION_ONLY_PER_REPROJ_STD_M;
	if (association_distinct_view_count(hyp) < 2) {
		std_m += ASSOC_POSITION_ONLY_SINGLE_VIEW_INFLATE_M;
	}
	if (hyp->matched_count <= 4) {
		std_m += ASSOC_POSITION_ONLY_FEW_LED_INFLATE_M;
	}
	if (std_m < ASSOC_POSITION_ONLY_MIN_STD_M) {
		std_m = ASSOC_POSITION_ONLY_MIN_STD_M;
	}
	if (std_m > ASSOC_POSITION_ONLY_MAX_STD_M) {
		std_m = ASSOC_POSITION_ONLY_MAX_STD_M;
	}
	return std_m;
}

static float
association_option_cost(const struct association_pose_hypothesis *hyp, enum association_observation_kind kind)
{
	switch (kind) {
	case ASSOC_OBS_POSE_LOCK: return hyp->cost.total_nll;
	case ASSOC_OBS_POSITION_ONLY: return hyp->cost.total_nll + ASSOC_POSITION_ONLY_ACTION_NLL;
	case ASSOC_OBS_LED_FOLD: return hyp->cost.total_nll + ASSOC_LED_FOLD_ACTION_NLL;
	case ASSOC_OBS_ABSENT:
	default: return 0.0f;
	}
}

static void
association_insert_hypothesis(struct association_device_work *work,
                              const struct association_pose_hypothesis *candidate)
{
	for (int i = 0; i < work->count; i++) {
		if (association_pose_duplicate(&work->hyps[i], candidate)) {
			if (association_hypothesis_less(candidate, &work->hyps[i])) {
				work->hyps[i] = *candidate;
			}
			return;
		}
	}

	if (work->count < ASSOCIATION_MAX_HYPOTHESES_PER_DEVICE) {
		work->hyps[work->count++] = *candidate;
	} else if (association_hypothesis_less(candidate, &work->hyps[work->count - 1])) {
		work->hyps[work->count - 1] = *candidate;
	}

	for (int i = work->count - 1; i > 0 && association_hypothesis_less(&work->hyps[i], &work->hyps[i - 1]); i--) {
		struct association_pose_hypothesis tmp = work->hyps[i - 1];
		work->hyps[i - 1] = work->hyps[i];
		work->hyps[i] = tmp;
	}
}

static bool
association_add_pose_hypothesis(struct t_constellation_tracker *ct,
                                struct association_device_work *work,
                                struct tracking_sample_device_state *dev_state,
                                struct constellation_tracking_sample *sample,
                                int view_id,
                                enum association_hypothesis_source source,
                                uint16_t flags,
                                const struct xrt_pose *P_cam_obj,
                                bool allow_refine)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct tracking_sample_frame *view = sample->views + view_id;
	struct constellation_tracker_camera_state *cam = ct->cam + view_id;

	if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
		return false;
	}

	struct association_pose_hypothesis hyp;
	association_hypothesis_init(&hyp, (uint8_t)dev_state->dev_index, (uint8_t)source);
	hyp.flags = flags | ASSOC_HYP_HAS_POSE;
	hyp.primary_view_id = (int16_t)view_id;
	hyp.pose_cam = *P_cam_obj;
	math_pose_transform(&view->P_world_cam, P_cam_obj, &hyp.pose_world);
	math_pose_transform(&cam->P_imu_cam, P_cam_obj, &hyp.pose_imu);

	const bool use_prior = dev_state->prior_tilt_trusted;
	struct pose_metrics primary_score = {0};
	bool have_primary_score = false;
	bool saw_partial_only = false;
	int total_visible = 0;
	int total_unmatched = 0;
	double total_reprojection = 0.0;
	/* Detection-likelihood terms summed over every contributing view. */
	double total_data_nll_detection = 0.0;
	double total_data_nll_missed_if_matched = 0.0;

	for (int match_view_id = 0; match_view_id < sample->n_views; match_view_id++) {
		struct tracking_sample_frame *match_view = sample->views + match_view_id;
		struct constellation_tracker_camera_state *match_cam = ct->cam + match_view_id;
		blobservation *bwobs = match_view->bwobs;
		if (bwobs == NULL || bwobs->num_blobs == 0) {
			continue;
		}

		struct xrt_pose P_match_cam_obj;
		math_pose_transform(&match_view->P_cam_world, &hyp.pose_world, &P_match_cam_obj);

		struct xrt_pose P_match_cam_obj_prior;
		math_pose_transform(&match_view->P_cam_world, &dev_state->P_world_obj_prior,
		                    &P_match_cam_obj_prior);

		struct pose_metrics score;
		if (use_prior) {
			pose_metrics_evaluate_pose_with_prior(&score, &P_match_cam_obj, false,
			                                      &P_match_cam_obj_prior, &dev_state->prior_pos_error,
			                                      &dev_state->prior_rot_error, bwobs->blobs,
			                                      bwobs->num_blobs, &device->led_model,
			                                      &match_cam->camera_model, NULL);
		} else {
			pose_metrics_evaluate_pose(&score, &P_match_cam_obj, bwobs->blobs, bwobs->num_blobs,
			                           &device->led_model, &match_cam->camera_model, NULL);
		}
		if (score.matched_blobs < 2) {
			continue;
		}
		const double view_error_per_blob = score.matched_blobs > 0
		                                       ? score.reprojection_error / (double)score.matched_blobs
		                                       : INFINITY;
		if (match_view_id != view_id && view_error_per_blob > 4.0) {
			continue;
		}

		struct pose_metrics_blob_match_info match_info;
		if (use_prior) {
			pose_metrics_match_pose_to_blobs_prior(&P_match_cam_obj, bwobs->blobs, bwobs->num_blobs,
			                                       &dev_state->prior_pos_error,
			                                       &dev_state->prior_rot_error, &device->led_model,
			                                       &match_cam->camera_model, &match_info);
		} else {
			pose_metrics_match_pose_to_blobs(&P_match_cam_obj, bwobs->blobs, bwobs->num_blobs,
			                                 &device->led_model, &match_cam->camera_model, &match_info);
		}

		const int added = association_fill_blob_refs(&hyp, &match_info, bwobs, match_view_id);
		if (added == 0) {
			continue;
		}

		total_visible += score.visible_leds;
		total_unmatched += score.unmatched_blobs;
		total_reprojection += score.reprojection_error;
		total_data_nll_detection += match_info.data_nll_detection;
		total_data_nll_missed_if_matched += match_info.data_nll_missed_if_matched;
		saw_partial_only |= POSE_HAS_FLAGS(&score, POSE_MATCH_PRIOR_SUPPORTED_PARTIAL);
		if (!have_primary_score || match_view_id == view_id) {
			primary_score = score;
			have_primary_score = true;
		}
	}

	if (!have_primary_score || hyp.matched_count < 2) {
		return false;
	}

	if (allow_refine) {
		struct xrt_pose refined_pose;
		if (association_refine_multiview_pose(ct, dev_state, sample, &hyp, &refined_pose)) {
			association_add_pose_hypothesis(ct, work, dev_state, sample, view_id, source, flags,
			                                &refined_pose, false);
		}
	}

	hyp.visible_count = (uint8_t)(total_visible > UINT8_MAX ? UINT8_MAX : total_visible);
	hyp.unmatched_count = (uint8_t)(total_unmatched > UINT8_MAX ? UINT8_MAX : total_unmatched);
	hyp.score = primary_score;
	hyp.score.visible_leds = hyp.visible_count;
	hyp.score.matched_blobs = hyp.matched_count;
	hyp.score.unmatched_blobs = hyp.unmatched_count;
	hyp.score.reprojection_error = total_reprojection;

	const float matched = hyp.matched_count > 0 ? (float)hyp.matched_count : 1.0f;
	const float error_per_observation = (float)(total_reprojection / (double)matched);
	const bool multi_view = association_distinct_view_count(&hyp) >= 2;
	if (hyp.matched_count >= 5 && error_per_observation < 2.0f &&
	    (multi_view || POSE_HAS_FLAGS(&primary_score, POSE_MATCH_POSITION | POSE_MATCH_ORIENT))) {
		hyp.score.match_flags |= POSE_MATCH_GOOD;
		hyp.score.match_flags &= ~POSE_MATCH_PRIOR_SUPPORTED_PARTIAL;
		if (error_per_observation < 0.75f && hyp.matched_count >= 6) {
			hyp.score.match_flags |= POSE_MATCH_STRONG;
		}
	} else if (saw_partial_only) {
		hyp.flags |= ASSOC_HYP_PARTIAL_ONLY;
		hyp.score.match_flags |= POSE_MATCH_PRIOR_SUPPORTED_PARTIAL;
	}

	/* Detection likelihood replaces fixed missed-LED penalties and matched-blob rewards. Reprojection remains
	 * a per-observation fit-quality term; detection_ref carries the cardinality preference on the same NLL scale. */
	const double detection_ref = total_data_nll_detection - total_data_nll_missed_if_matched;
	hyp.cost.reprojection_nll = error_per_observation;
	hyp.cost.missed_led_nll = (float)detection_ref;
	hyp.cost.clutter_nll = (float)total_unmatched * ASSOC_CLUTTER_NLL;
	hyp.cost.matched_evidence_nll = 0.0f;
	hyp.cost.position_prior_nll = association_position_prior_nll(&primary_score, dev_state);
	struct xrt_pose P_cam_obj_prior;
	math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);
	hyp.cost.orientation_prior_nll =
	    association_orientation_prior_nll(dev_state, P_cam_obj, &P_cam_obj_prior, &view->cam_gravity_vector);
	hyp.cost.head_anchor_nll = association_head_anchor_nll(device, cam, view, sample, P_cam_obj);
	hyp.cost.body_state_nll = 0.0f;
	hyp.cost.temporal_nll = 0.0f;
	hyp.cost.total_nll = hyp.cost.reprojection_nll + hyp.cost.missed_led_nll + hyp.cost.clutter_nll +
	                     hyp.cost.matched_evidence_nll + hyp.cost.position_prior_nll + hyp.cost.orientation_prior_nll +
	                     hyp.cost.head_anchor_nll + hyp.cost.body_state_nll + hyp.cost.temporal_nll;

	association_insert_hypothesis(work, &hyp);
	association_emit_candidate(device, dev_state, view, view_id, sample->timestamp, &hyp, false, 0);
	return true;
}

static void
association_add_prior_pose_sources(struct t_constellation_tracker *ct,
                                   struct association_device_work *work,
                                   struct tracking_sample_device_state *dev_state,
                                   struct constellation_tracking_sample *sample)
{
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}
		work->has_blobs = true;
		struct xrt_pose P_cam_obj;
		if (dev_state->prior_tilt_trusted) {
			math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj);
			association_add_pose_hypothesis(ct, work, dev_state, sample, view_id,
			                                ASSOC_SOURCE_PRIOR_POSE, ASSOC_HYP_NONE, &P_cam_obj, true);
		}
		if (dev_state->have_last_seen_pose) {
			math_pose_transform(&view->P_cam_world, &dev_state->last_seen_pose, &P_cam_obj);
			association_add_pose_hypothesis(ct, work, dev_state, sample, view_id, ASSOC_SOURCE_LAST_SEEN,
			                                ASSOC_HYP_NONE, &P_cam_obj, true);
		}
	}
}

static void
association_add_labelled_pnp_source(struct t_constellation_tracker *ct,
                                    struct association_device_work *work,
                                    struct tracking_sample_device_state *dev_state,
                                    struct constellation_tracking_sample *sample,
                                    int view_id,
                                    enum association_hypothesis_source source)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct tracking_sample_frame *view = sample->views + view_id;
	struct constellation_tracker_camera_state *cam = ct->cam + view_id;
	blobservation *bwobs = view->bwobs;

	const int labelled_count = association_count_labelled_blobs(bwobs, device->led_model.id);
	if (labelled_count < 3) {
		return;
	}

	if (labelled_count == 3) {
		struct xrt_pose p3p_solutions[4];
		const int n_solutions = pnp_solve_p3p(bwobs->blobs, bwobs->num_blobs, &device->led_model,
		                                      &cam->camera_model, p3p_solutions,
		                                      (int)ARRAY_SIZE(p3p_solutions));
		if (source == ASSOC_SOURCE_PRIOR_LABELLED_PNP && dev_state->prior_tilt_trusted) {
			int best = -1;
			float best_nll = INFINITY;
			for (int i = 0; i < n_solutions; i++) {
				const float nll =
				    association_prior_pose_nll(device, cam, dev_state, view, sample, &p3p_solutions[i]);
				if (nll < best_nll) {
					best_nll = nll;
					best = i;
				}
			}
			if (best >= 0) {
				association_add_pose_hypothesis(ct, work, dev_state, sample, view_id, source,
				                                ASSOC_HYP_HAS_TWIN, &p3p_solutions[best], false);
			}
			return;
		}
		for (int i = 0; i < n_solutions; i++) {
			association_add_pose_hypothesis(ct, work, dev_state, sample, view_id, source,
			                                ASSOC_HYP_HAS_TWIN, &p3p_solutions[i], true);
		}
		return;
	}

	struct xrt_pose P_cam_obj;
	math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj);
	struct xrt_pose twin_pose;
	bool has_twin = false;
	if (!ransac_pnp_pose_with_twin(&P_cam_obj, bwobs->blobs, bwobs->num_blobs, &device->led_model,
	                               &cam->camera_model, NULL, NULL, &twin_pose, &has_twin)) {
		return;
	}

	const uint16_t twin_flag = has_twin ? ASSOC_HYP_HAS_TWIN : ASSOC_HYP_NONE;
	association_add_pose_hypothesis(ct, work, dev_state, sample, view_id, source, twin_flag, &P_cam_obj, true);
	if (has_twin) {
		association_add_pose_hypothesis(ct, work, dev_state, sample, view_id, source,
		                                ASSOC_HYP_HAS_TWIN | ASSOC_HYP_IS_TWIN, &twin_pose, true);
	}
}

static void
association_add_label_sources(struct t_constellation_tracker *ct,
                              struct association_device_work *work,
                              struct tracking_sample_device_state *dev_state,
                              struct constellation_tracking_sample *sample)
{
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}

		association_add_labelled_pnp_source(ct, work, dev_state, sample, view_id, ASSOC_SOURCE_LABELLED_PNP);

		struct association_blob_label_snapshot snapshot;
		association_save_labels(view->bwobs, &snapshot);
		const int n_labelled = device_propagate_labels_in_view(ct, dev_state, sample, view_id);
		if (n_labelled >= 3) {
			association_add_labelled_pnp_source(ct, work, dev_state, sample, view_id,
			                                    ASSOC_SOURCE_PRIOR_LABELLED_PNP);
		}
		association_restore_labels(view->bwobs, &snapshot);
	}
}

static void
association_add_joint_pnp_source(struct t_constellation_tracker *ct,
                                 struct association_device_work *work,
                                 struct tracking_sample_device_state *dev_state,
                                 struct constellation_tracking_sample *sample)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct association_blob_label_snapshot snapshots[CONSTELLATION_MAX_CAMERAS];
	bool have_snapshot[CONSTELLATION_MAX_CAMERAS] = {false};
	struct joint_pnp_view jviews[JOINT_PNP_MAX_VIEWS];
	int view_ids[JOINT_PNP_MAX_VIEWS];
	int n_jviews = 0;

	for (int view_id = 0; view_id < sample->n_views && n_jviews < JOINT_PNP_MAX_VIEWS; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		struct constellation_tracker_camera_state *cam = ct->cam + view_id;
		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}
		association_save_labels(view->bwobs, &snapshots[view_id]);
		have_snapshot[view_id] = true;
		const int n_labelled = device_propagate_labels_in_view(ct, dev_state, sample, view_id);
		if (n_labelled < 3) {
			continue;
		}
		jviews[n_jviews].blobs = view->bwobs->blobs;
		jviews[n_jviews].num_blobs = view->bwobs->num_blobs;
		jviews[n_jviews].calib = &cam->camera_model;
		jviews[n_jviews].P_imu_cam = cam->P_imu_cam;
		view_ids[n_jviews] = view_id;
		n_jviews++;
	}

	if (n_jviews >= 2) {
		const int seed_view = view_ids[0];
		struct constellation_tracker_camera_state *seed_cam = ct->cam + seed_view;
		struct tracking_sample_frame *seed = sample->views + seed_view;
		struct xrt_pose P_cam_obj_prior, P_imu_obj;
		math_pose_transform(&seed->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);
		math_pose_transform(&seed_cam->P_imu_cam, &P_cam_obj_prior, &P_imu_obj);

		int num_rays = 0, num_inliers = 0;
		if (joint_pnp_solve(&P_imu_obj, jviews, n_jviews, &device->led_model, &num_rays, &num_inliers)) {
			for (int i = 0; i < n_jviews; i++) {
				const int view_id = view_ids[i];
				struct xrt_pose P_cam_imu, P_cam_obj;
				math_pose_invert(&ct->cam[view_id].P_imu_cam, &P_cam_imu);
				math_pose_transform(&P_cam_imu, &P_imu_obj, &P_cam_obj);
				association_add_pose_hypothesis(ct, work, dev_state, sample, view_id, ASSOC_SOURCE_JOINT_PNP,
				                                ASSOC_HYP_JOINT, &P_cam_obj, true);
			}
		}
	}

	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		if (have_snapshot[view_id]) {
			association_restore_labels(sample->views[view_id].bwobs, &snapshots[view_id]);
		}
	}
}

static void
association_add_cold_search_source(struct t_constellation_tracker *ct,
                                   struct association_device_work *work,
                                   struct tracking_sample_device_state *dev_state,
                                   struct constellation_tracking_sample *sample)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		struct constellation_tracker_camera_state *cam = ct->cam + view_id;
		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}
		correspondence_search_set_blobs(cam->cs, view->bwobs->blobs, view->bwobs->num_blobs);
		for (int pass = 0; pass < 2; pass++) {
			enum correspondence_search_flags flags = CS_FLAG_NONE;
			flags |= pass == 0 ? CS_FLAG_SHALLOW_SEARCH : CS_FLAG_DEEP_SEARCH;
			if (dev_state->prior_tilt_trusted) {
				flags |= CS_FLAG_HAVE_POSE_PRIOR | CS_FLAG_TRUST_PRIOR_ORIENT;
			}
			/* Blob labels are tracker state, not sensor truth. With the unified global selector downstream,
			 * small candidate sets should be searched without treating another device's stale label as a hard
			 * exclusion; compatibility is enforced after hypotheses exist. Keep the cap because full search is
			 * combinatorial in blob count. */
			if (view->bwobs->num_blobs <= ASSOC_MATCH_ALL_BLOBS_MAX) {
				flags |= CS_FLAG_MATCH_ALL_BLOBS;
			}
			struct xrt_pose P_cam_obj;
			math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj);
			struct correspondence_search_result results[CORRESPONDENCE_SEARCH_MAX_RESULTS];
			const int n_results = correspondence_search_find_pose_candidates(
			    cam->cs, device->search_led_model, flags, &P_cam_obj, &dev_state->prior_pos_error,
			    &dev_state->prior_rot_error, &view->cam_gravity_vector, (float)GRAVITY_TILT_TOL,
			    dev_state->prior_yaw_sigma_rad, (float)FLIP_COST_HUBER_KNEE_SIGMA,
			    (float)FLIP_COST_WEIGHT, results, CORRESPONDENCE_SEARCH_MAX_RESULTS);
			struct correspondence_search_diagnostics diag;
			correspondence_search_get_last_diagnostics(cam->cs, &diag);
			const struct xrt_device *xdev =
			    device->connection != NULL ? device->connection->xdev : NULL;
			telem_emit_search_result(telem_device_id(xdev), view_id, sample->timestamp, pass, flags,
			                         n_results > 0, dev_state->prior_tilt_trusted, &diag);
			for (int i = 0; i < n_results; i++) {
				association_add_pose_hypothesis(ct, work, dev_state, sample, view_id,
				                                ASSOC_SOURCE_COLD_SEARCH, ASSOC_HYP_NONE,
				                                &results[i].pose, true);
			}
		}
	}
}

static void
association_build_device_work(struct t_constellation_tracker *ct,
                              struct association_device_work *work,
                              struct tracking_sample_device_state *dev_state,
                              struct constellation_tracking_sample *sample,
                              bool allow_cold_search)
{
	*work = (struct association_device_work){0};
	association_add_prior_pose_sources(ct, work, dev_state, sample);
	association_add_joint_pnp_source(ct, work, dev_state, sample);
	association_add_label_sources(ct, work, dev_state, sample);

	if (allow_cold_search) {
		association_add_cold_search_source(ct, work, dev_state, sample);
	}
}

static bool
association_is_compatible_with_chosen(const struct association_pose_hypothesis *candidate,
                                      const struct association_pose_hypothesis *const chosen[],
                                      int chosen_count)
{
	if (candidate == NULL) {
		return true;
	}
	for (int i = 0; i < chosen_count; i++) {
		if (chosen[i] != NULL && !association_joint_pair_is_compatible(candidate, chosen[i])) {
			return false;
		}
	}
	return true;
}

static const struct association_pose_hypothesis *
association_best_lockable_local(const struct association_device_work *work)
{
	for (int h = 0; work != NULL && h < work->count; h++) {
		if (association_lock_eligible(&work->hyps[h])) {
			return &work->hyps[h];
		}
	}
	return NULL;
}

static uint8_t
association_count_choice_conflicts(const struct association_pose_hypothesis *candidate,
                                   const struct association_pose_hypothesis *const chosen[],
                                   int n_devices,
                                   int own_slot)
{
	uint8_t count = 0;
	for (int i = 0; candidate != NULL && i < n_devices; i++) {
		if (i == own_slot || chosen[i] == NULL) {
			continue;
		}
		count += association_hypotheses_shared_blob_count(candidate, chosen[i]);
	}
	return count;
}

static float
association_device_lower_bound(const struct association_device_work *work)
{
	float best = work->has_blobs ? ASSOC_ABSENT_WITH_BLOBS_COST : 0.0f;
	for (int h = 0; h < work->count; h++) {
		const struct association_pose_hypothesis *candidate = &work->hyps[h];
		if (association_lock_eligible(candidate)) {
			const float cost = association_option_cost(candidate, ASSOC_OBS_POSE_LOCK);
			if (cost < best) {
				best = cost;
			}
		}
		if (association_position_only_eligible(candidate)) {
			const float cost = association_option_cost(candidate, ASSOC_OBS_POSITION_ONLY);
			if (cost < best) {
				best = cost;
			}
		}
		if (association_led_fold_eligible(candidate)) {
			const float cost = association_option_cost(candidate, ASSOC_OBS_LED_FOLD);
			if (cost < best) {
				best = cost;
			}
		}
	}
	return best;
}

static float
association_remaining_lower_bound(const struct association_device_work work[], int n_devices, int index)
{
	float lower_bound = 0.0f;
	for (int i = index; i < n_devices; i++) {
		lower_bound += association_device_lower_bound(&work[i]);
	}
	return lower_bound;
}

static int
association_count_visual_observations(const enum association_observation_kind kind[], int n_devices)
{
	int count = 0;
	for (int i = 0; i < n_devices; i++) {
		if (kind[i] != ASSOC_OBS_ABSENT) {
			count++;
		}
	}
	return count;
}

static void
association_emit_lockable_not_chosen(struct t_constellation_tracker *ct,
                                     const struct tracking_sample_device_state *dev_state,
                                     const struct constellation_tracking_sample *sample,
                                     const struct association_pose_hypothesis *hyp,
                                     const struct association_pose_hypothesis *const chosen[],
                                     int slot)
{
	if (!g2_telem_enabled() || hyp == NULL) {
		return;
	}
	const struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	const struct xrt_device *xdev = device->connection != NULL ? device->connection->xdev : NULL;
	const uint8_t conflicts = association_count_choice_conflicts(hyp, chosen, sample->n_devices, slot);
	g2_telem_event(telem_device_id(xdev), sample->timestamp, ASSOC_EV_LOCKABLE_NOT_CHOSEN, (float)conflicts);
}

static void
association_select_joint_recursive(const struct association_device_work work[],
                                   int n_devices,
                                   int index,
                                   const struct association_pose_hypothesis *chosen[],
                                   enum association_observation_kind chosen_kind[],
                                   float cost,
                                   struct association_joint_choice *best)
{
	if (index == n_devices) {
		const int visual_count = association_count_visual_observations(chosen_kind, n_devices);
		if (!best->valid || cost < best->total_cost - 1e-4f ||
		    (fabsf(cost - best->total_cost) <= 1e-4f && visual_count > best->visual_count)) {
			best->valid = true;
			best->total_cost = cost;
			best->visual_count = visual_count;
			for (int i = 0; i < n_devices; i++) {
				best->chosen[i] = chosen[i];
				best->kind[i] = chosen_kind[i];
			}
		}
		return;
	}

	const float absent_cost = work[index].has_blobs ? ASSOC_ABSENT_WITH_BLOBS_COST : 0.0f;
	if (best->valid && cost + association_remaining_lower_bound(work, n_devices, index) > best->total_cost) {
		return;
	}
	if (!best->valid ||
	    cost + absent_cost + association_remaining_lower_bound(work, n_devices, index + 1) <= best->total_cost) {
		chosen[index] = NULL;
		chosen_kind[index] = ASSOC_OBS_ABSENT;
		association_select_joint_recursive(work, n_devices, index + 1, chosen, chosen_kind,
		                                   cost + absent_cost, best);
	}

	for (int h = 0; h < work[index].count; h++) {
		const struct association_pose_hypothesis *candidate = &work[index].hyps[h];
		const enum association_observation_kind kinds[] = {
		    ASSOC_OBS_POSE_LOCK,
		    ASSOC_OBS_POSITION_ONLY,
		    ASSOC_OBS_LED_FOLD,
		};
		for (size_t k = 0; k < ARRAY_SIZE(kinds); k++) {
			const enum association_observation_kind kind = kinds[k];
			if ((kind == ASSOC_OBS_POSE_LOCK && !association_lock_eligible(candidate)) ||
			    (kind == ASSOC_OBS_POSITION_ONLY && !association_position_only_eligible(candidate)) ||
			    (kind == ASSOC_OBS_LED_FOLD && !association_led_fold_eligible(candidate))) {
				continue;
			}
			const float option_cost = association_option_cost(candidate, kind);
			if (best->valid &&
			    cost + option_cost + association_remaining_lower_bound(work, n_devices, index + 1) >
			        best->total_cost) {
				continue;
			}
			if (!association_is_compatible_with_chosen(candidate, chosen, index)) {
				continue;
			}
			chosen[index] = candidate;
			chosen_kind[index] = kind;
			association_select_joint_recursive(work, n_devices, index + 1, chosen, chosen_kind,
			                                   cost + option_cost, best);
		}
	}
	chosen[index] = NULL;
	chosen_kind[index] = ASSOC_OBS_ABSENT;
}

static void
association_fold_ambiguous_hypothesis(struct t_constellation_tracker *ct,
                                      struct tracking_sample_device_state *dev_state,
                                      struct constellation_tracking_sample *sample,
                                      const struct association_pose_hypothesis *hyp)
{
	if (hyp == NULL || hyp->primary_view_id < 0 || hyp->primary_view_id >= sample->n_views) {
		return;
	}
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct tracking_sample_frame *view = sample->views + hyp->primary_view_id;
	if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
		return;
	}
	association_emit_candidate(device, dev_state, view, hyp->primary_view_id, sample->timestamp, hyp, true, 0);
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		association_fold_hypothesis_view(ct, dev_state, sample, hyp, view_id, false);
	}
}

static bool
association_commit_joint_pose(struct t_constellation_tracker *ct,
                              struct tracking_sample_device_state *dev_state,
                              struct constellation_tracking_sample *sample,
                              const struct association_pose_hypothesis *hyp)
{
	if (hyp == NULL || hyp->primary_view_id < 0 || hyp->primary_view_id >= sample->n_views) {
		return false;
	}
	struct tracking_sample_frame *view = sample->views + hyp->primary_view_id;
	if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
		return false;
	}
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	const int selected_matches =
	    association_apply_hypothesis_matches(dev_state, device, view, hyp, hyp->primary_view_id);
	if (selected_matches < 4 && hyp->matched_count < 4) {
		const struct xrt_device *xdev = device->connection != NULL ? device->connection->xdev : NULL;
		g2_telem_event(telem_device_id(xdev), sample->timestamp, ASSOC_EV_LOCK_COMMIT_FAILED,
		               (float)selected_matches);
		association_emit_candidate(device, dev_state, view, hyp->primary_view_id, sample->timestamp, hyp,
		                           true, 0);
		return false;
	}
	association_emit_candidate(device, dev_state, view, hyp->primary_view_id, sample->timestamp, hyp, true, 1);
	dev_state->score = hyp->score;
	dev_state->score.matched_blobs = hyp->matched_count;
	struct xrt_pose pose = hyp->pose_cam;
	submit_device_pose(ct, dev_state, sample, hyp->primary_view_id, &pose, false);
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		if (view_id == hyp->primary_view_id) {
			continue;
		}
		association_fold_hypothesis_view(ct, dev_state, sample, hyp, view_id, true);
	}
	return true;
}

static bool
association_commit_position_only(struct t_constellation_tracker *ct,
                                 struct tracking_sample_device_state *dev_state,
                                 struct constellation_tracking_sample *sample,
                                 const struct association_pose_hypothesis *hyp)
{
	if (hyp == NULL || hyp->primary_view_id < 0 || hyp->primary_view_id >= sample->n_views) {
		return false;
	}

	struct tracking_sample_frame *view = sample->views + hyp->primary_view_id;
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct xrt_pose P_xrworld_model;
	pose_flip_YZ(&hyp->pose_world, &P_xrworld_model);

	struct xrt_pose P_xrworld_device;
	math_pose_transform(&P_xrworld_model, &device->led_model.P_model_device, &P_xrworld_device);

	const float std_m = association_position_observation_std_m(hyp);
	const struct xrt_vec3 position_variance = {std_m * std_m, std_m * std_m, std_m * std_m};
	constellation_tracked_device_connection_notify_position(device->connection, sample->timestamp,
	                                                        &P_xrworld_device.position,
	                                                        &position_variance);

	if (g2_telem_enabled()) {
		const struct xrt_device *xdev = device->connection != NULL ? device->connection->xdev : NULL;
		association_emit_candidate(device, dev_state, view, hyp->primary_view_id, sample->timestamp, hyp,
		                           true, 0);
		g2_telem_event(telem_device_id(xdev), sample->timestamp, ASSOC_EV_POSITION_ONLY_SELECTED, std_m);
	}

	return true;
}

static void
constellation_associate_covariance_frame(struct t_constellation_tracker *ct,
                                         struct constellation_tracking_sample *sample,
                                         bool allow_cold_search)
{
	struct association_device_work work[CONSTELLATION_MAX_DEVICES] = {0};
	for (int i = 0; i < sample->n_devices; i++) {
		association_build_device_work(ct, &work[i], &sample->devices[i], sample, allow_cold_search);
	}

	const struct association_pose_hypothesis *current[CONSTELLATION_MAX_DEVICES] = {0};
	enum association_observation_kind current_kind[CONSTELLATION_MAX_DEVICES] = {0};
	struct association_joint_choice best = {0};
	association_select_joint_recursive(work, sample->n_devices, 0, current, current_kind, 0.0f, &best);
	if (!best.valid) {
		for (int i = 0; i < sample->n_devices; i++) {
			association_fold_prior_leds(ct, &sample->devices[i], sample);
		}
		return;
	}

	for (int slot = 0; slot < sample->n_devices; slot++) {
		struct tracking_sample_device_state *dev_state = sample->devices + slot;
		const struct association_pose_hypothesis *choice = best.chosen[slot];
		const enum association_observation_kind choice_kind = best.kind[slot];
		const struct association_pose_hypothesis *best_lockable = association_best_lockable_local(&work[slot]);
		if (choice == NULL && best_lockable != NULL) {
			association_emit_lockable_not_chosen(ct, dev_state, sample, best_lockable, best.chosen, slot);
		}
		if (choice_kind == ASSOC_OBS_POSE_LOCK) {
			if (association_commit_joint_pose(ct, dev_state, sample, choice)) {
				continue;
			}
		}

		if (choice_kind == ASSOC_OBS_POSITION_ONLY) {
			if (association_commit_position_only(ct, dev_state, sample, choice)) {
				work[slot].folded_partial = true;
				continue;
			}
		}

		if (choice_kind == ASSOC_OBS_LED_FOLD) {
			association_fold_ambiguous_hypothesis(ct, dev_state, sample, choice);
			work[slot].folded_partial = true;
			continue;
		}

		if (association_fold_prior_leds(ct, dev_state, sample)) {
			work[slot].folded_partial = true;
		}
	}
}

// Fast frame processing: blob extraction and match to existing predictions
static void
constellation_tracker_process_frame_fast(struct xrt_frame_sink *sink, struct xrt_frame *xf)
{
	struct t_constellation_tracker *ct = container_of(sink, struct t_constellation_tracker, fast_process_sink);
	struct xrt_space_relation xsr_base_pose;

	/* Allocate a tracking sample for everything we're about to process */
	struct constellation_tracking_sample *sample = constellation_tracking_sample_new();
	uint64_t fast_analysis_start_ts = os_monotonic_get_ns();

	CT_DEBUG(ct, "Starting analysis of frame %" PRIu64 " TS %" PRIu64, xf->source_sequence, xf->timestamp);

	/* Get the HMD's pose so we can calculate the camera view poses */
	xrt_device_get_tracked_pose(ct->hmd_xdev, XRT_INPUT_GENERIC_TRACKER_POSE, xf->timestamp, &xsr_base_pose);

	/* Record the live SLAM head pose for this frame (world, OpenXR) so the offline replay harness can
	 * reproduce the true camera->world transform instead of an IMU-only reconstruction. */
	if (g2_telem_enabled()) {
		float head7[7];
		telem_pack_pose(&xsr_base_pose.pose, head7);
		g2_telem_head_pose((uint64_t)xf->timestamp, head7);
	}

	/* Split out camera views and collect blobs across all cameras */
	assert(ct->cam_count <= XRT_TRACKING_MAX_SLAM_CAMS);
	sample->n_views = ct->cam_count;
	sample->timestamp = xf->timestamp;

	for (int i = 0; i < ct->cam_count; i++) {
		struct constellation_tracker_camera_state *cam = ct->cam + i;
		struct tracking_sample_frame *view = sample->views + i;

		// Flip the input pose to CV coords, so we can do all our operations
		// in OpenCV coords
		struct xrt_pose P_cvworld_hmdimu;
		pose_flip_YZ(&xsr_base_pose.pose, &P_cvworld_hmdimu);

		math_pose_transform(&P_cvworld_hmdimu, &cam->P_imu_cam, &view->P_world_cam);

		CT_DEBUG(ct,
		         "Prepare transforms for cam %d "
		         " HMD pose %f,%f,%f,%f pos %f,%f,%f "
		         " P_imu_cam %f,%f,%f,%f pos %f,%f,%f "
		         " P_world_cam %f,%f,%f,%f pos %f,%f,%f ",
		         i, xsr_base_pose.pose.orientation.x, xsr_base_pose.pose.orientation.y,
		         xsr_base_pose.pose.orientation.z, xsr_base_pose.pose.orientation.w,
		         xsr_base_pose.pose.position.x, xsr_base_pose.pose.position.y, xsr_base_pose.pose.position.z,

		         cam->P_imu_cam.orientation.x, cam->P_imu_cam.orientation.y, cam->P_imu_cam.orientation.z,
		         cam->P_imu_cam.orientation.w, cam->P_imu_cam.position.x, cam->P_imu_cam.position.y,
		         cam->P_imu_cam.position.z,

		         view->P_world_cam.orientation.x, view->P_world_cam.orientation.y,
		         view->P_world_cam.orientation.z, view->P_world_cam.orientation.w, view->P_world_cam.position.x,
		         view->P_world_cam.position.y, view->P_world_cam.position.z);

		// Calculate inverse from cam back to world coords
		math_pose_invert(&view->P_world_cam, &view->P_cam_world);

		const struct xrt_vec3 gravity_vector = {0.0, 1.0, 0.0};
		math_quat_rotate_vec3(&view->P_cam_world.orientation, &gravity_vector, &view->cam_gravity_vector);

		u_frame_create_roi(xf, cam->roi, &view->vframe);
		view->bw = cam->bw;

		/* The actual exposure for this frame is encoded in the full frame's pixel header
		 * (wmr_camera.c: data[6..7]); read it from xf, not the ROI sub-frame which excludes
		 * the header row. Same value for all cameras on a controller frame. */
		uint16_t frame_exposure =
		    (xf->data != NULL && xf->size > 7) ? (uint16_t)((xf->data[6] << 8) | xf->data[7]) : 0;

		os_mutex_lock(&cam->bw_lock);
			/* Predictive-ROI: project every LED of every tracked device through its ESKF state to this
			 * camera's image, build a bounding box, restrict blob search to it. Falls back to full-frame
			 * when any connected device lacks enough in-frame predictions (so one tracked controller cannot
			 * crop out a controller that needs full-frame acquisition/re-acquisition). The scan pad is
			 * per-LED-adaptive (max of the base pad floor and k * sqrt(max-diag(S))); base pad and sigma
			 * multiplier are env-tunable for in-headset re-tuning across lighting/motion. */
		static int predictive_roi_base_pad = ROI_BASE_PAD_PX;
		static float predictive_roi_sigma_k = ROI_SIGMA_K;
		static int predictive_roi_env_init = 0;
		if (!predictive_roi_env_init) {
			predictive_roi_env_init = 1;
			const char *p = getenv("G2_PREDICTIVE_ROI_BASE_PAD_PX");
			if (p != NULL && p[0] != '\0' && atoi(p) > 0) {
				predictive_roi_base_pad = atoi(p);
			}
			const char *k = getenv("G2_PREDICTIVE_ROI_SIGMA_K");
			if (k != NULL && k[0] != '\0' && atof(k) > 0.0) {
				predictive_roi_sigma_k = (float)atof(k);
			}
		}
		bool used_roi = false;
		{
			/* Build the cam_calib + the OpenXR-camera-frame pose this camera uses for projection, the
			 * same way emit_view_led_observations does so the predicted pixels live in the same coords
			 * as the LED gate uses elsewhere in the tracker. */
			struct xrt_pose P_xrworld_cam_pred;
			math_pose_transform(&view->P_cam_world, &P_YZ_FLIP, &P_xrworld_cam_pred);
			const struct t_constellation_cam_calib cam_calib_pred = {
			    cam->camera_model.calib.fx, cam->camera_model.calib.fy,
			    cam->camera_model.calib.cx, cam->camera_model.calib.cy};
				float xmin = 1e9f, ymin = 1e9f, xmax = -1e9f, ymax = -1e9f;
				int n_in_frame = 0;
				int n_roi_ready_devices = 0;
				int n_connected_devices = 0;
				for (int d = 0; d < ct->num_devices; d++) {
					struct constellation_tracker_device *device = &ct->devices[d];
					const struct t_constellation_led_model *lm = &device->led_model;
					if (device->connection == NULL || lm->leds == NULL || lm->num_leds == 0) {
						continue; /* not connected / model not loaded yet */
					}
					n_connected_devices++;
					int device_in_frame = 0;
					for (int li = 0; li < lm->num_leds; li++) {
						struct xrt_vec3 led_flip = {lm->leds[li].pos.x, -lm->leds[li].pos.y,
						                            -lm->leds[li].pos.z};
					struct xrt_vec3 led_obj;
					math_pose_transform_point(&lm->P_device_model, &led_flip, &led_obj);
					float zhat[2], S[4];
						if (!constellation_tracked_device_connection_predict_led_gate(
						        device->connection, &P_xrworld_cam_pred, &cam_calib_pred, &led_obj,
						        zhat, S)) {
							continue; /* device untracked: no usable prior */
						}
						if (!isfinite(zhat[0]) || !isfinite(zhat[1]) || !isfinite(S[0]) || !isfinite(S[3])) {
							continue; /* non-finite projection or covariance: skip */
						}
						/* zhat is in FULL-camera pixel coords (cam_calib's principal point);
						 * translate to view->vframe coords by subtracting the ROI offset. */
					const float vx = zhat[0] - (float)cam->roi.offset.w;
					const float vy = zhat[1] - (float)cam->roi.offset.h;
					if (vx >= 0.f && vx < (float)view->vframe->width && vy >= 0.f &&
					    vy < (float)view->vframe->height) {
						/* Per-LED adaptive pad: sqrt of the max-diagonal of S is an upper bound on
						 * the 1-sigma pixel-space prediction noise (max eigenvalue ≤ trace ≤ 2*max
						 * diag). k·σ gives the half-width of the LED's plausibility patch; floored
						 * at the base pad so a near-zero S can't shrink it below the centroid noise. */
						const float sx = S[0] > 0.0f ? sqrtf(S[0]) : 0.0f;
						const float sy = S[3] > 0.0f ? sqrtf(S[3]) : 0.0f;
						const float pix_sigma = sx > sy ? sx : sy;
						float per_led_pad = predictive_roi_sigma_k * pix_sigma;
						if (per_led_pad < (float)predictive_roi_base_pad) {
							per_led_pad = (float)predictive_roi_base_pad;
						}
						const float lx = vx - per_led_pad;
						const float ly = vy - per_led_pad;
						const float hx = vx + per_led_pad;
						const float hy = vy + per_led_pad;
						if (lx < xmin) xmin = lx;
						if (ly < ymin) ymin = ly;
							if (hx > xmax) xmax = hx;
							if (hy > ymax) ymax = hy;
							n_in_frame++;
							device_in_frame++;
						}
					}
					if (device_in_frame >= ROI_FALLBACK_MIN_PREDICTIONS) {
						n_roi_ready_devices++;
					}
				}
				if (n_connected_devices > 0 && n_roi_ready_devices == n_connected_devices &&
				    n_in_frame >= ROI_FALLBACK_MIN_PREDICTIONS) {
				/* xmin/ymin/xmax/ymax already carry per-LED pad; no extra global pad needed. */
				int rx = (int)floorf(xmin);
				int ry = (int)floorf(ymin);
				int rw = (int)ceilf(xmax - xmin) + 1;
				int rh = (int)ceilf(ymax - ymin) + 1;
				blobwatch_process_roi_lowthresh(cam->bw, view->vframe, frame_exposure, 0, rx, ry, rw, rh,
				                                4, 3, 12, &view->bwobs);
				used_roi = true;
			}
		}
		if (!used_roi) {
			blobwatch_process(cam->bw, view->vframe, frame_exposure, 0, &view->bwobs);
		}
		os_mutex_unlock(&cam->bw_lock);

		if (view->bwobs == NULL) {
			cam->last_num_blobs = 0;
			continue;
		}

		blobservation *bwobs = view->bwobs;
		cam->last_num_blobs = bwobs->num_blobs;

		CT_TRACE(ct, "frame %" PRIu64 " TS %" PRIu64 " cam %d ROI %d,%d w/h %d,%d Blobs: %d",
		         xf->source_sequence, xf->timestamp, i, cam->roi.offset.w, cam->roi.offset.h, cam->roi.extent.w,
		         cam->roi.extent.h, bwobs->num_blobs);

#if 0
		for (int index = 0; index < bwobs->num_blobs; index++) {
			printf("  Blob[%d]: %f,%f %dx%d id %d age %u\n", index, bwobs->blobs[index].x,
			       bwobs->blobs[index].y, bwobs->blobs[index].width, bwobs->blobs[index].height,
			       bwobs->blobs[index].led_id, bwobs->blobs[index].age);
		}
#endif
	}
	uint64_t blob_extract_finish_ts = os_monotonic_get_ns();
	ct->last_blob_analysis_ms = (blob_extract_finish_ts - fast_analysis_start_ts) / U_TIME_1MS_IN_NS;

	// Ready to start processing device poses now. Make sure we have the
	// LED models and collect the best estimate of the current pose
	// for each target device
	os_mutex_lock(&ct->tracked_device_lock);
	assert(ct->num_devices <= CONSTELLATION_MAX_DEVICES);

	for (int d = 0; d < ct->num_devices; d++) {
		struct constellation_tracker_device *device = ct->devices + d;

		if (!device->have_led_model) {
			if (!constellation_tracked_device_connection_get_led_model(device->connection,
			                                                           &device->led_model)) {
				continue; // Can't do anything without the LED info
			}

			CT_INFO(ct, "Constellation Tracker: Retrieved controller LED model for device %u",
			        device->led_model.id);
			device->search_led_model = t_constellation_search_model_new(&device->led_model);
			device->have_led_model = true;
		}

		struct tracking_sample_device_state *dev_state = sample->devices + sample->n_devices;

		// Prior pose for matching: the fusion's RAW estimate (no body-lock ride — the visual out-of-view
		// ride must never feed back as the matcher's prior), falling back to the device's reported pose for
		// a device that doesn't expose the raw estimate.
		struct xrt_space_relation xsr;
		if (!constellation_tracked_device_connection_get_predicted_pose(device->connection, xf->timestamp,
		                                                                &xsr) &&
		    !constellation_tracked_device_connection_get_tracked_pose(device->connection, xf->timestamp,
		                                                              &xsr)) {
			CT_DEBUG(ct, "Failed to retrieve prior pose for device %u", device->led_model.id);
			continue; // Can't retrieve the pose: means the device was disconnected
		}

		// Apply device -> LED model pose from xsr = P_world_device + P_device_model = P_world_model
		struct xrt_pose P_xrworld_model;
		math_pose_transform(&xsr.pose, &device->led_model.P_device_model, &P_xrworld_model);

		// Incoming controller pose is in OpenXR. Flip it to OpenCV for all our operations
		pose_flip_YZ(&P_xrworld_model, &dev_state->P_world_obj_prior);

		/* Covariance-driven prior gate: set the prior-consistency tolerance from the fusion's live
		 * 1-sigma uncertainty (PRIOR_GATE_SIGMA sigmas), clamped to [MIN_*_ERROR, MAX_*_ERROR]. A
		 * single scalar sigma applied isotropically is the frame-robust choice (the fusion covariance
		 * is world-frame; these bounds are used in the matcher's frame). When the fusion isn't tracking
		 * yet, fall back to the fixed floor. This widens the gate after an optical dropout (so
		 * prior-refine accepts frames it would otherwise drop to the slow search) and keeps it tight
		 * when confident. */
		float pos_bound = MIN_POS_ERROR, rot_bound = MIN_ROT_ERROR;
		double pos_std = 0.0, rot_std = 0.0, yaw_std = 0.0;
		bool tilt_trusted = false;
		/* Soft mirror-flip cost's yaw scale: the live fusion YAW-AXIS 1-sigma (the orientation error about
		 * world-up alone) when tracking, else half a turn (untracked -> the yaw term vanishes). Keyed off the
		 * yaw DoF specifically, NOT the worst-direction orientation sigma — the gravity-anchored tilt is
		 * observable and tight, so folding it into the yaw scale would inflate it and conflate two DoFs.
		 * Floored at FLIP_COST_YAW_SIGMA_MIN (see its definition) so the cost cannot over-trust an
		 * over-confident prior yaw; the live sigma widens the scale above the floor after a real dropout. */
		float yaw_sigma = (float)FLIP_COST_YAW_SIGMA_MAX;
		if (constellation_tracked_device_connection_get_pose_uncertainty(device->connection, &pos_std,
		                                                                 &rot_std, &yaw_std)) {
			pos_bound = (float)fmin(fmax(PRIOR_GATE_SIGMA * pos_std, MIN_POS_ERROR), MAX_POS_ERROR);
			rot_bound = (float)fmin(fmax(PRIOR_GATE_SIGMA * rot_std, MIN_ROT_ERROR), MAX_ROT_ERROR);
			yaw_sigma = (float)fmin(fmax(yaw_std, FLIP_COST_YAW_SIGMA_MIN), FLIP_COST_YAW_SIGMA_MAX);
			/* TILT is driftless (gravity-anchored): trusted whenever the fusion is tracking, even through a
			 * dropout. The soft cost's yaw scale (yaw_sigma) widens with the live yaw uncertainty, so a
			 * stale yaw self-deweights rather than needing a binary trust flag. */
			tilt_trusted = true; /* get_pose_uncertainty returned true => tracking => gravity-anchored prior */
		}
		dev_state->prior_tilt_trusted = tilt_trusted;
		dev_state->prior_yaw_sigma_rad = yaw_sigma;
		dev_state->prior_pos_error.x = dev_state->prior_pos_error.y = dev_state->prior_pos_error.z =
		    pos_bound;
		dev_state->prior_rot_error.x = dev_state->prior_rot_error.y = dev_state->prior_rot_error.z =
		    rot_bound;

		dev_state->have_last_seen_pose = device->have_last_seen_pose;
		dev_state->last_seen_pose = device->last_seen_pose;

		dev_state->dev_index = d;
		dev_state->led_model = &device->led_model;

		sample->n_devices++;
	}
	os_mutex_unlock(&ct->tracked_device_lock);

	constellation_associate_covariance_frame(ct, sample, true);

	uint64_t fast_analysis_finish_ts = os_monotonic_get_ns();
	ct->last_fast_analysis_ms = (fast_analysis_finish_ts - fast_analysis_start_ts) / U_TIME_1MS_IN_NS;

	/* Send analysis results to debug view if needed */
	enum debug_draw_flag debug_flags = DEBUG_DRAW_FLAG_NONE;
	if (ct->debug_draw_normalise)
		debug_flags |= DEBUG_DRAW_FLAG_NORMALISE;
	if (ct->debug_draw_blob_tint)
		debug_flags |= DEBUG_DRAW_FLAG_BLOB_TINT;
	if (ct->debug_draw_blob_circles)
		debug_flags |= DEBUG_DRAW_FLAG_BLOB_CIRCLE;
	if (ct->debug_draw_blob_ids)
		debug_flags |= DEBUG_DRAW_FLAG_BLOB_IDS;
	if (ct->debug_draw_blob_unique_ids)
		debug_flags |= DEBUG_DRAW_FLAG_BLOB_UNIQUE_IDS;
	if (ct->debug_draw_leds)
		debug_flags |= DEBUG_DRAW_FLAG_LEDS;
	if (ct->debug_draw_prior_leds)
		debug_flags |= DEBUG_DRAW_FLAG_PRIOR_LEDS;
	if (ct->debug_draw_last_leds)
		debug_flags |= DEBUG_DRAW_FLAG_LAST_SEEN_LEDS;
	if (ct->debug_draw_pose_bounds)
		debug_flags |= DEBUG_DRAW_FLAG_POSE_BOUNDS;
	if (ct->debug_draw_device_bounds)
		debug_flags |= DEBUG_DRAW_FLAG_DEVICE_BOUNDS;

	for (int i = 0; i < sample->n_views; i++) {
		struct constellation_tracker_camera_state *cam = ct->cam + i;
		struct tracking_sample_frame *view = sample->views + i;

		cam->debug_last_pose = view->P_world_cam;
		cam->debug_last_gravity_vector = view->cam_gravity_vector;

		if (u_sink_debug_is_active(&cam->debug_sink)) {
			struct xrt_frame *xf_src = view->vframe;
			struct xrt_frame *xf_dbg = NULL;

			u_frame_create_one_off(XRT_FORMAT_R8G8B8, xf_src->width, xf_src->height, &xf_dbg);
			xf_dbg->timestamp = xf_src->timestamp;

			// if (view->bwobs != NULL) {
			debug_draw_blobs_leds(xf_dbg, xf_src, debug_flags, view, i, &cam->camera_model, sample->devices,
			                      sample->n_devices);
			//}

			u_sink_debug_push_frame(&cam->debug_sink, xf_dbg);
			xrt_frame_reference(&xf_dbg, NULL);
		}
	}

	atomic_fetch_add_explicit(&ct->frames_completed, 1, memory_order_release);
	constellation_tracking_sample_free(sample);
}

static void
constellation_tracker_node_destroy(struct xrt_frame_node *node)
{
	struct t_constellation_tracker *ct = container_of(node, struct t_constellation_tracker, node);

	DRV_TRACE_MARKER();
	CT_DEBUG(ct, "Destroying constellation tracker");

	// Unlink the device connections and release the models
	os_mutex_lock(&ct->tracked_device_lock);
	for (int i = 0; i < ct->num_devices; i++) {
		struct constellation_tracker_device *device = ct->devices + i;

		// Clean up the LED tracking model
		t_constellation_led_model_clear(&device->led_model);
		if (device->search_led_model) {
			t_constellation_search_model_free(device->search_led_model);
		}

		if (device->connection != NULL) {
			t_constellation_tracked_device_connection_disconnect(device->connection);
			device->connection = NULL;
		}
	}
	os_mutex_unlock(&ct->tracked_device_lock);
	os_mutex_destroy(&ct->tracked_device_lock);

	//! Clean up
	for (int i = 0; i < ct->cam_count; i++) {
		struct constellation_tracker_camera_state *cam = ct->cam + i;

		u_sink_debug_destroy(&cam->debug_sink);

		if (cam->cs) {
			correspondence_search_free(cam->cs);
		}

		os_mutex_destroy(&cam->bw_lock);
		if (cam->bw) {
			blobwatch_free(cam->bw);
		}
	}

	u_var_remove_root(ct);
	free(ct);
}

int
t_constellation_tracker_create(struct xrt_frame_context *xfctx,
                               struct xrt_device *hmd_xdev,
                               struct t_constellation_camera_group *cams,
                               struct t_constellation_tracker **out_tracker,
                               struct xrt_frame_sink **out_sink,
                               struct xrt_device_masks_sink *controller_mask_sink)
{
	DRV_TRACE_MARKER();

	int ret;
	struct t_constellation_tracker *ct = calloc(1, sizeof(struct t_constellation_tracker));

	ct->log_level = debug_get_log_option_ct_log();
	ct->debug_draw_blob_tint = true;
	ct->debug_draw_blob_ids = true;
	ct->hmd_xdev = hmd_xdev;
	ct->controller_masks_sink = controller_mask_sink;
	atomic_store_explicit(&ct->frames_completed, 0, memory_order_relaxed);

	// Set up the per-camera constellation tracking pieces config and pose
	ct->cam_count = cams->cam_count;
	for (int i = 0; i < ct->cam_count; i++) {
		struct constellation_tracker_camera_state *cam = ct->cam + i;
		struct t_constellation_camera *cam_cfg = cams->cams + i;

		cam->roi = cam_cfg->roi;
		cam->P_imu_cam = cam_cfg->P_imu_cam;
		cam->slam_tracking_index = cam_cfg->slam_tracking_index;

		/* Init the camera model with size and distortion */
		cam->camera_model.width = cam_cfg->roi.extent.w;
		cam->camera_model.height = cam_cfg->roi.extent.h;
		t_camera_model_params_from_t_camera_calibration(&cam_cfg->calibration, &cam->camera_model.calib);

		os_mutex_init(&cam->bw_lock);
		cam->bw = blobwatch_new(cam_cfg->blob_min_threshold, cam_cfg->blob_detect_threshold, (uint8_t)i);
		cam->cs = correspondence_search_new(&cam->camera_model);
	}

	// Set up frame receiver
	ct->base.push_frame = constellation_tracker_receive_frame;

	// Setup node
	struct xrt_frame_node *xfn = &ct->node;
	xfn->break_apart = constellation_tracker_node_break_apart;
	xfn->destroy = constellation_tracker_node_destroy;

	ret = os_mutex_init(&ct->tracked_device_lock);
	if (ret != 0) {
		CT_ERROR(ct, "Failed to init tracked device mutex!");
		constellation_tracker_node_destroy(&ct->node);
		return -1;
	}

	// Fast processing thread
	ct->fast_process_sink.push_frame = constellation_tracker_process_frame_fast;

	if (!u_sink_queue_create(xfctx, MAX_FAST_QUEUE_SIZE, &ct->fast_process_sink, &ct->fast_q_sink)) {
		CT_ERROR(ct, "Failed to init fast analysis queue!");
		constellation_tracker_node_destroy(&ct->node);
		return -1;
	}

	u_var_add_root(ct, "Constellation Tracker", false);
	u_var_add_log_level(ct, &ct->log_level, "Log Level");
	u_var_add_ro_i32(ct, &ct->num_devices, "Num Devices");
	u_var_add_ro_u64(ct, &ct->last_frame_timestamp, "Last Frame Timestamp");
	u_var_add_ro_u64(ct, &ct->last_blob_analysis_ms, "Blob tracking time (ms)");
	u_var_add_ro_u64(ct, &ct->last_fast_analysis_ms, "Fast analysis time (ms)");

	u_var_add_bool(ct, &ct->debug_draw_normalise, "Debug: Normalise source frame");
	u_var_add_bool(ct, &ct->debug_draw_blob_tint, "Debug: Tint blobs by device assignment");
	u_var_add_bool(ct, &ct->debug_draw_blob_circles, "Debug: Draw circles around blobs");
	u_var_add_bool(ct, &ct->debug_draw_blob_ids, "Debug: Draw LED id labels for blobs");
	u_var_add_bool(ct, &ct->debug_draw_blob_unique_ids, "Debug: Draw blobs tracking ID");
	u_var_add_bool(ct, &ct->debug_draw_leds, "Debug: Draw LED position markers for found poses");
	u_var_add_bool(ct, &ct->debug_draw_prior_leds, "Debug: Draw LED markers for prior poses");
	u_var_add_bool(ct, &ct->debug_draw_last_leds, "Debug: Draw LED markers for last observed poses");
	u_var_add_bool(ct, &ct->debug_draw_pose_bounds, "Debug: Draw LED bounds rect for found poses");
	u_var_add_bool(ct, &ct->debug_draw_device_bounds, "Debug: Draw device bounds rect for found poses");

	for (int i = 0; i < ct->cam_count; i++) {
		struct constellation_tracker_camera_state *cam = ct->cam + i;
		u_var_add_ro_i32(ct, &cam->last_num_blobs, "Num Blobs");
		u_var_add_pose(ct, &cam->debug_last_pose, "Last view pose");
		u_var_add_vec3_f32(ct, &cam->debug_last_gravity_vector, "Last gravity vector");

		char cam_name[64];
		sprintf(cam_name, "Cam %u", i);
		u_sink_debug_init(&cam->debug_sink);
		u_var_add_sink_debug(ct, &cam->debug_sink, cam_name);
	}

	// Hand ownership to the frame context
	xrt_frame_context_add(xfctx, &ct->node);

	CT_DEBUG(ct, "Constellation tracker created");

	*out_tracker = ct;
	*out_sink = &ct->base;

	return 0;
}

static void
constellation_tracked_device_connection_destroy(struct t_constellation_tracked_device_connection *ctdc)
{
	DRV_TRACE_MARKER();

	os_mutex_destroy(&ctdc->lock);
	free(ctdc);
}

static struct t_constellation_tracked_device_connection *
constellation_tracked_device_connection_create(int id,
                                               struct xrt_device *xdev,
                                               struct t_constellation_tracked_device_callbacks *cb,
                                               struct t_constellation_tracker *tracker)
{
	DRV_TRACE_MARKER();

	assert(xdev != NULL);
	assert(cb != NULL);

	struct t_constellation_tracked_device_connection *ctdc =
	    calloc(1, sizeof(struct t_constellation_tracked_device_connection));

	ctdc->id = id;
	ctdc->xdev = xdev;
	ctdc->cb = cb;
	ctdc->tracker = tracker;

	/* Init 2 references - one for the tracked device, one for the tracker */
	xrt_reference_inc(&ctdc->ref);
	xrt_reference_inc(&ctdc->ref);

	int ret = os_mutex_init(&ctdc->lock);
	if (ret != 0) {
		CT_ERROR(tracker, "Constellation tracker device connection: Failed to init mutex!");
		constellation_tracked_device_connection_destroy(ctdc);
		return NULL;
	}

	return ctdc;
}

struct t_constellation_tracked_device_connection *
t_constellation_tracker_add_device(struct t_constellation_tracker *ct,
                                   struct xrt_device *xdev,
                                   struct t_constellation_tracked_device_callbacks *cb)
{
	os_mutex_lock(&ct->tracked_device_lock);
	assert(ct->num_devices < CONSTELLATION_MAX_DEVICES);

	CT_DEBUG(ct, "Constellation tracker: Adding device %d", ct->num_devices);

	struct t_constellation_tracked_device_connection *ctdc =
	    constellation_tracked_device_connection_create(ct->num_devices, xdev, cb, ct);
	if (ctdc != NULL) {
		struct constellation_tracker_device *device = ct->devices + ct->num_devices;
		device->connection = ctdc;
		device->last_matched_cam = -1;
		ct->num_devices++;

		const char *device_type;
		switch (xdev->device_type) {
		case XRT_DEVICE_TYPE_HMD: device_type = "HMD"; break;
		case XRT_DEVICE_TYPE_RIGHT_HAND_CONTROLLER: device_type = "Right"; break;
		case XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER: device_type = "Left"; break;
		case XRT_DEVICE_TYPE_ANY_HAND_CONTROLLER: device_type = "Any"; break;
		case XRT_DEVICE_TYPE_GENERIC_TRACKER: device_type = "Tracker"; break;
		default: device_type = "Unknown"; break;
		}

		char dev_name[64];
		sprintf(dev_name, "Device %u - %s", ct->num_devices, device_type);
		u_var_add_ro_text(ct, "Device", dev_name);
		u_var_add_pose(ct, &device->last_seen_pose, "Last observed global pose");
		u_var_add_u64(ct, &device->last_seen_pose_ts, "Last observed pose");
		u_var_add_ro_i32(ct, &device->last_matched_blobs, "Last matched Blobs");
		u_var_add_ro_i32(ct, &device->last_matched_cam, "Last observed camera #");
		u_var_add_pose(ct, &device->last_matched_cam_pose, "Last observed camera pose");
	}

	os_mutex_unlock(&ct->tracked_device_lock);
	return ctdc;
}

void
t_constellation_tracked_device_connection_disconnect(struct t_constellation_tracked_device_connection *ctdc)
{
	os_mutex_lock(&ctdc->lock);
	ctdc->disconnected = true;
	os_mutex_unlock(&ctdc->lock);

	if (xrt_reference_dec_and_is_zero(&ctdc->ref)) {
		constellation_tracked_device_connection_destroy(ctdc);
	}
}

uint64_t
t_constellation_tracker_debug_frames_completed(struct t_constellation_tracker *ct)
{
	return atomic_load_explicit(&ct->frames_completed, memory_order_acquire);
}
