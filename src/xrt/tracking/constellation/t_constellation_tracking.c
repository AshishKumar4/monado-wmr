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
#include <stdio.h>

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
 * inflates its covariance -> the matcher accepts the prior-refined frames through the gate instead of
 * falling through to the slow ab-initio search. Bigger re-acquisitions beyond the ceiling go through that
 * search, not this gate. One statistical knob (the sigma multiplier); the rest are physical floors/ceilings. */
#define PRIOR_GATE_SIGMA 3.0 /* ~99.7% per axis */
#define MAX_POS_ERROR 0.60
#define MAX_ROT_ERROR DEG_TO_RAD(60)

/* The tight DRIFTLESS tilt sigma for the soft anisotropic mirror-flip cost (both the twin selection and the
 * ab-initio search). The mirror twin of a few-LED PnP almost always TILTS the controller wrong. Gravity is
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
 * while an uncertain/untracked prior makes the yaw term vanish and reprojection decide (the
 * cold-start/ab-initio bootstrap). The
 * Huber knee reuses the 3-sigma envelope (PRIOR_GATE_SIGMA): within it the penalty is quadratic, beyond
 * it linear, so a gross outlier (flip) cannot dominate pathologically. FLIP_COST_WEIGHT commensurates the
 * dimensionless robustified distance with the per-LED reprojection error (px^2). FLIP_COST_YAW_SIGMA_MAX is
 * the untracked/long-dropout yaw scale: ~half a turn, so the yaw term contributes negligibly and the tilt
 * term (still gravity-anchored) carries the decision alone. */
#define FLIP_COST_WEIGHT 1.0
#define FLIP_COST_HUBER_KNEE_SIGMA PRIOR_GATE_SIGMA
#define FLIP_COST_YAW_SIGMA_MAX DEG_TO_RAD(180)
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

/* Covariance-gated partial-fold. When the fast paths cannot solve a PnP (<4 cleanly-labelled LEDs for the
 * device in any single camera — the dominant fall-through, a visibility limit not a bug) but the fusion HAS
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

	//! Full search / pose recovery thread
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
	uint64_t last_long_analysis_ms;

	struct u_var_button full_search_button;
	bool do_full_search;

	// Fast tracking thread
	struct xrt_frame_sink *fast_q_sink;
	struct xrt_frame_sink fast_process_sink;

	// Long analysis / recovery thread
	struct os_thread_helper long_analysis_thread;
	struct constellation_tracking_sample *long_analysis_pending_sample;

	//! Frames fully processed through the pipeline (incremented EXACTLY once per frame at every
	//! pipeline exit). Guarded by long_analysis_thread's lock. Lets the offline harness barrier on
	//! per-frame completion instead of racing a fixed sleep (debug/test only).
	uint64_t frames_completed;

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
 * pairing. Returns false (no gate) until the fusion is tracking — i.e. cold start / no usable prior — so
 * the caller's gated partial fold is a no-op then and the ab-initio path runs (cold-start guard). */
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
                   bool is_recovered,
                   bool joint_seed)
{
	struct constellation_tracker_camera_state *cam = ct->cam + view_id;
	struct tracking_sample_frame *view = sample->views + view_id;
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;

	struct pose_metrics *score = &dev_state->score;

	pose_metrics_match_pose_to_blobs(P_cam_obj, view->bwobs->blobs, view->bwobs->num_blobs, &device->led_model,
	                                 &cam->camera_model, &dev_state->blob_match_info);
	mark_matching_blobs(ct, P_cam_obj, view->bwobs, &device->led_model, &dev_state->blob_match_info);

	int num_leds_out = 0;
	int num_inliers = 0;

	/* Per-view single-camera RANSAC-PnP polish: drives the per-LED reprojection error down (the seed pose
	 * was matched/labelled but not necessarily LM-tight for THIS view). For a multi-camera JOINT seed the
	 * polish is FLIP-GUARDED: a from-scratch single-view re-solve can land on the front/back mirror the
	 * joint solve removed, so we keep the polish only when it stays on the joint pose's branch (within the
	 * driftless tilt tolerance) — accuracy when consistent, the disambiguated joint pose when the polish
	 * would flip. Single-camera callers (joint_seed=false) take the polish unconditionally as before. */
	{
		struct xrt_pose refine_pose = *P_cam_obj;
		if (!ransac_pnp_pose(&refine_pose, view->bwobs->blobs, view->bwobs->num_blobs, &device->led_model,
		                     &cam->camera_model, &num_leds_out, &num_inliers)) {
			CT_DEBUG(ct, "Camera %d RANSAC-PnP refinement for device %d from %u blobs failed", view_id,
			         device->led_model.id, view->bwobs->num_blobs);
		} else {
			bool keep = true;
			if (joint_seed) {
				/* geodesic angle between the polished and the joint pose; reject a polish that flipped. */
				const struct xrt_quat *a = &refine_pose.orientation, *b = &P_cam_obj->orientation;
				double d = fabs((double)a->x * b->x + (double)a->y * b->y + (double)a->z * b->z +
				                (double)a->w * b->w);
				d = d > 1.0 ? 1.0 : d;
				keep = (2.0 * acos(d)) <= GRAVITY_TILT_TOL;
			}
			CT_DEBUG(ct,
			         "Camera %d RANSAC-PnP refinement for device %d from %u blobs had %d LEDs with %d inliers "
			         "(kept=%d)",
			         view_id, device->led_model.id, view->bwobs->num_blobs, num_leds_out, num_inliers, keep);
			if (keep) {
				*P_cam_obj = refine_pose;
			}
		}
	}

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
		                      (uint8_t)num_inliers, (float)score->reprojection_error, pose7,
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

	if (!dev_state->found_device_pose) {
		math_pose_transform(&view->P_world_cam, P_cam_obj, &dev_state->final_pose);

		dev_state->found_device_pose = true;
		dev_state->found_pose_view_id = view_id;

		const struct xrt_vec3 fwd = {0.0, 0.0, -1.0};
		struct xrt_vec3 dev_fwd, dev_prior_fwd, cam_prior_fwd;

		math_quat_rotate_vec3(&dev_state->final_pose.orientation, &fwd, &dev_fwd);
		math_quat_rotate_vec3(&dev_state->P_world_obj_prior.orientation, &fwd, &dev_prior_fwd);

		struct xrt_pose P_cam_obj_prior;
		math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);
		math_quat_rotate_vec3(&P_cam_obj_prior.orientation, &fwd, &cam_prior_fwd);

		CT_DEBUG(ct,
		         "Found a pose on cam %u device %d score match_flags 0x%x matched %u "
		         "blobs of %u visible LEDs. Global pose %f,%f,%f,%f pos %f,%f,%f  fwd %f,%f,%f "
		         "cam-relative pose was %f,%f,%f,%f pos %f,%f,%f "
		         "Global prior %f,%f,%f,%f pos %f,%f,%f fwd %f,%f,%f "
		         "cam-relative prior %f,%f,%f,%f pos %f,%f,%f fwd %f,%f,%f",
		         view_id, device->led_model.id, score->match_flags, score->matched_blobs, score->visible_leds,
		         dev_state->final_pose.orientation.x, dev_state->final_pose.orientation.y,
		         dev_state->final_pose.orientation.z, dev_state->final_pose.orientation.w,
		         dev_state->final_pose.position.x, dev_state->final_pose.position.y,
		         dev_state->final_pose.position.z, dev_fwd.x, dev_fwd.y, dev_fwd.z, P_cam_obj->orientation.x,
		         P_cam_obj->orientation.y, P_cam_obj->orientation.z, P_cam_obj->orientation.w,
		         P_cam_obj->position.x, P_cam_obj->position.y, P_cam_obj->position.z,
		         dev_state->P_world_obj_prior.orientation.x, dev_state->P_world_obj_prior.orientation.y,
		         dev_state->P_world_obj_prior.orientation.z, dev_state->P_world_obj_prior.orientation.w,
		         dev_state->P_world_obj_prior.position.x, dev_state->P_world_obj_prior.position.y,
		         dev_state->P_world_obj_prior.position.z, dev_prior_fwd.x, dev_prior_fwd.y, dev_prior_fwd.z,

		         P_cam_obj_prior.orientation.x, P_cam_obj_prior.orientation.y, P_cam_obj_prior.orientation.z,
		         P_cam_obj_prior.orientation.w, P_cam_obj_prior.position.x, P_cam_obj_prior.position.y,
		         P_cam_obj_prior.position.z, cam_prior_fwd.x, cam_prior_fwd.y, cam_prior_fwd.z);
	} else if (view_id != dev_state->found_pose_view_id) {
		struct xrt_pose extra_final_pose, pose_delta;

		math_pose_transform(&view->P_world_cam, P_cam_obj, &extra_final_pose);
		math_quat_unrotate(&dev_state->final_pose.orientation, &extra_final_pose.orientation,
		                   &pose_delta.orientation);
		pose_delta.position = extra_final_pose.position;
		math_vec3_subtract(&dev_state->final_pose.position, &pose_delta.position);

		CT_DEBUG(ct,
		         "Found an extra pose on cam %u device %d score match_flags 0x%x matched %u "
		         "blobs of %u visible LEDs. Global pose %f,%f,%f,%f pos %f,%f,%f "
		         "cam-relative pose %f,%f,%f,%f pos %f,%f,%f "
		         "delta from 1st pose %f,%f,%f,%f pos %f,%f,%f",
		         view_id, device->led_model.id, score->match_flags, score->matched_blobs, score->visible_leds,
		         extra_final_pose.orientation.x, extra_final_pose.orientation.y, extra_final_pose.orientation.z,
		         extra_final_pose.orientation.w, extra_final_pose.position.x, extra_final_pose.position.y,
		         extra_final_pose.position.z, P_cam_obj->orientation.x, P_cam_obj->orientation.y,
		         P_cam_obj->orientation.z, P_cam_obj->orientation.w, P_cam_obj->position.x,
		         P_cam_obj->position.y, P_cam_obj->position.z, pose_delta.orientation.x,
		         pose_delta.orientation.y, pose_delta.orientation.z, pose_delta.orientation.w,
		         pose_delta.position.x, pose_delta.position.y, pose_delta.position.z);

		// Mix the found position with the prior one
		math_vec3_scalar_mul(0.5, &pose_delta.position);
		math_vec3_accum(&pose_delta.position, &dev_state->final_pose.position);
	}

	os_mutex_lock(&ct->tracked_device_lock);
	if (device->have_last_seen_pose == false || sample->timestamp > device->last_seen_pose_ts) {
		device->have_last_seen_pose = true;
		device->last_seen_pose_ts = sample->timestamp;
		device->last_seen_pose = dev_state->final_pose;
		device->last_matched_blobs = score->matched_blobs;
		device->last_matched_cam = view_id;
		device->last_matched_cam_pose = *P_cam_obj;

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
		average_brightness /= matched_blobs;

		constellation_tracked_device_connection_notify_brightness_update(device->connection,
		                                                                 average_brightness);

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

/* Fast matching test based on hypothesised pose */
static bool
device_try_global_pose(struct t_constellation_tracker *ct,
                       struct tracking_sample_device_state *dev_state,
                       struct constellation_tracking_sample *sample,
                       struct xrt_pose *P_world_obj_candidate)
{
	bool ret = false;
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		struct constellation_tracker_camera_state *cam = ct->cam + view_id;

		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}

		struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
		blobservation *bwobs = view->bwobs;

		struct xrt_pose P_cam_obj_candidate;
		math_pose_transform(&view->P_cam_world, P_world_obj_candidate, &P_cam_obj_candidate);

		struct xrt_pose P_cam_obj_prior;
		math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);

		pose_metrics_evaluate_pose_with_prior(&dev_state->score, &P_cam_obj_candidate, false, &P_cam_obj_prior,
		                                      &dev_state->prior_pos_error, &dev_state->prior_rot_error,
		                                      bwobs->blobs, bwobs->num_blobs, &device->led_model,
		                                      &cam->camera_model, NULL);

		if (POSE_HAS_FLAGS(&dev_state->score, POSE_MATCH_GOOD | POSE_MATCH_LED_IDS)) {
			submit_device_pose(ct, dev_state, sample, view_id, &P_cam_obj_candidate, false /* not a recovery */,
			                   false /* single-cam: take the polish unconditionally */);
			ret = true;
		}
	}

	return ret;
}

/* Solve a device pose for one view from the blobs currently labelled for this device: RANSAC-PnP
 * seeded by the prior, scored against the prior, and submitted if it is a GOOD match. Shared by the
 * recover-from-labelled-blobs and prior-refine fast paths, which differ only in how the blobs got
 * labelled (frame-to-frame tracking vs prior projection) and in the telemetry outcome. Returns true
 * if a pose was submitted. */
static bool
device_solve_view_from_labelled(struct t_constellation_tracker *ct,
                                struct tracking_sample_device_state *dev_state,
                                struct constellation_tracking_sample *sample,
                                int view_id,
                                bool is_recovery)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct t_constellation_led_model *leds_model = &device->led_model;
	struct tracking_sample_frame *view = sample->views + view_id;
	struct constellation_tracker_camera_state *cam = ct->cam + view_id;
	blobservation *bwobs = view->bwobs;
	const uint8_t telem_dev = telem_device_id(device->connection->xdev);

	/* Need enough blobs labelled for THIS device to constrain a PnP solve. */
	int num_blobs = 0;
	for (int index = 0; index < bwobs->num_blobs; index++) {
		if (LED_OBJECT_ID(bwobs->blobs[index].led_id) == leds_model->id) {
			num_blobs++;
		}
	}
	if (num_blobs < 4) {
		return false;
	}

	struct xrt_pose P_cam_obj_prior;
	math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);

	if (is_recovery) {
		/* Telemetry: a recovery attempt from labelled blobs for this device/view. */
		g2_telem_event(telem_dev, (uint64_t)sample->timestamp, 2 /* recover_attempt */, (float)num_blobs);
	}

	/* Solve, and on the near-coplanar few-LED geometry that causes the mirror two-fold ambiguity, also
	 * recover the second (twin) pose. Reprojection cannot separate the twins (both fit the blobs), so we
	 * pick the one consistent with the IMU/fusion prior — committing the validated hypothesis this frame
	 * rather than dropping a flipped solve. (prior_must_match below makes "GOOD" == prior-consistent, so a
	 * flipped candidate simply fails to score GOOD and the correct twin wins.) */
	struct xrt_pose P_cam_obj = P_cam_obj_prior;
	struct xrt_pose twin_pose;
	bool has_twin = false;
	if (!ransac_pnp_pose_with_twin(&P_cam_obj, bwobs->blobs, bwobs->num_blobs, leds_model, &cam->camera_model,
	                               NULL, NULL, &twin_pose, &has_twin)) {
		CT_DEBUG(ct, "Camera %d RANSAC-PnP for device %d from %d blobs failed", view_id, leds_model->id,
		         num_blobs);
		if (g2_telem_enabled()) {
			float pose7[7];
			telem_pack_pose(&P_cam_obj, pose7);
			g2_telem_pose_attempt(telem_dev, (uint8_t)view_id, (uint64_t)sample->timestamp, 0,
			                      (uint8_t)num_blobs, 0, 0.0f, pose7, 0 /* rejected */);
		}
		return false;
	}

	pose_metrics_evaluate_pose_with_prior(&dev_state->score, &P_cam_obj, true, &P_cam_obj_prior,
	                                      &dev_state->prior_pos_error, &dev_state->prior_rot_error, bwobs->blobs,
	                                      bwobs->num_blobs, &device->led_model, &cam->camera_model, NULL);

	/* Mirror-twin disambiguation by the soft prior-consistency cost (same mechanism as the ab-initio
	 * re-rank): cost = reproj + soft prior penalty, take the LOWER-cost twin and always commit it (never
	 * drop the frame for ambiguity). Mirror twins reproject near-identically, so the prior penalty — the
	 * anisotropic Mahalanobis distance from the prior, tilt-tight/yaw-by-live-sigma, Huber-robustified —
	 * is what separates them: a tilt flip's huge driftless tilt distance is never selected (decided even
	 * when the yaw prior is stale), while a fresh-yaw flip is out-ranked by
	 * its yaw distance. When the prior is untracked the yaw scale widens and reprojection decides. The
	 * penalty is keyed off the prior only when tilt-trusted (gravity-anchored); otherwise it is zero and
	 * the comparison is the plain reprojection ordering (cold reacquire). */
	if (has_twin) {
		struct pose_metrics twin_score;
		pose_metrics_evaluate_pose_with_prior(&twin_score, &twin_pose, true, &P_cam_obj_prior,
		                                      &dev_state->prior_pos_error, &dev_state->prior_rot_error,
		                                      bwobs->blobs, bwobs->num_blobs, &device->led_model,
		                                      &cam->camera_model, NULL);
		double prim_cost = 0.0, twin_cost = 0.0;
		if (dev_state->prior_tilt_trusted) {
			prim_cost = pose_metrics_prior_orient_cost(
			    &P_cam_obj.orientation, &P_cam_obj_prior.orientation, &view->cam_gravity_vector,
			    GRAVITY_TILT_TOL, dev_state->prior_yaw_sigma_rad, FLIP_COST_HUBER_KNEE_SIGMA,
			    FLIP_COST_WEIGHT);
			twin_cost = pose_metrics_prior_orient_cost(
			    &twin_pose.orientation, &P_cam_obj_prior.orientation, &view->cam_gravity_vector,
			    GRAVITY_TILT_TOL, dev_state->prior_yaw_sigma_rad, FLIP_COST_HUBER_KNEE_SIGMA,
			    FLIP_COST_WEIGHT);
		}
		if (pose_metrics_score_is_better_pose_prior(&dev_state->score, prim_cost, &twin_score, twin_cost)) {
			dev_state->score = twin_score;
			P_cam_obj = twin_pose;
		}
	}

	if (POSE_HAS_FLAGS(&dev_state->score, POSE_MATCH_GOOD)) {
		CT_DEBUG(ct, "Camera %d %s pose for device %d from %d blobs", view_id,
		         is_recovery ? "recovered" : "prior-refined", leds_model->id, num_blobs);
		/* submit_device_pose emits the accepted pose_attempt (outcome 2=recovered vs 1=accepted). */
		submit_device_pose(ct, dev_state, sample, view_id, &P_cam_obj, is_recovery,
		                   false /* single-cam: take the polish unconditionally */);
		return true;
	}

	/* Sub-threshold capture: pose not accepted, but the PnP solve is geometrically consistent, so
	 * fold its matched LEDs (each gated per-LED in the fusion) — partial/dropout frames still inform
	 * tracking instead of being discarded. */
	pose_metrics_match_pose_to_blobs(&P_cam_obj, bwobs->blobs, bwobs->num_blobs, leds_model,
	                                 &cam->camera_model, &dev_state->blob_match_info);
	emit_view_led_observations(dev_state, device, cam, view, view_id, sample->timestamp);

	if (g2_telem_enabled()) {
		float pose7[7];
		telem_pack_pose(&P_cam_obj, pose7);
		g2_telem_pose_attempt(telem_dev, (uint8_t)view_id, (uint64_t)sample->timestamp,
		                      (uint8_t)dev_state->score.visible_leds, (uint8_t)dev_state->score.matched_blobs, 0,
		                      (float)dev_state->score.reprojection_error, pose7, 0 /* rejected */);
	}
	return false;
}

/* Try and recover the pose from blobs already labelled for this device by frame-to-frame tracking. */
static bool
device_try_recover_pose(struct t_constellation_tracker *ct,
                        struct tracking_sample_device_state *dev_state,
                        struct constellation_tracking_sample *sample)
{
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}
		if (device_solve_view_from_labelled(ct, dev_state, sample, view_id, true /* recovery */)) {
			return true;
		}
	}
	return false;
}

/* Pose-predicted LED label propagation: project the fusion's PREDICTED controller pose's LED
 * model into one view and assign each blob to the LED it lands on, back-face culled by the LED normals
 * and bounded by the prior's anisotropic per-LED gate (all inside pose_metrics_match_pose_to_blobs).
 * This is the PROJECTED-LED-motion prior — the labels follow the predicted pose through head and
 * controller motion — NOT raw pixel velocity, which parallax and ego-motion make unreliable. Returns
 * the count of blobs newly labelled to this device, so the caller can decide a view carries enough
 * propagated IDs to solve (and to count propagation vs the ab-initio fall-through). The label transfer
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

	/* Project the predicted pose's (front-facing, in-frame) LEDs and match each to a blob within its
	 * prior-sized gate; mark_matching_blobs then writes the LED IDs onto the matched blobs. */
	pose_metrics_match_pose_to_blobs(&P_cam_obj_prior, view->bwobs->blobs, view->bwobs->num_blobs,
	                                 &device->led_model, &cam->camera_model, &dev_state->blob_match_info);
	mark_matching_blobs(ct, &P_cam_obj_prior, view->bwobs, &device->led_model, &dev_state->blob_match_info);

	int n_labelled = 0;
	for (int i = 0; i < dev_state->blob_match_info.num_visible_leds; i++) {
		if (dev_state->blob_match_info.visible_leds[i].matched_blob != NULL) {
			n_labelled++;
		}
	}
	return n_labelled;
}

/* Commit a joint multi-camera pose to every view that sees the device: score it against the prior, fold
 * each view's matched LEDs, and submit. P_imu_obj is the joint solution in the rig (IMU) frame; per view
 * the camera-relative pose is inv(P_imu_cam) . P_imu_obj. submit_device_pose is called with joint_seed so
 * its per-view RANSAC polish is FLIP-GUARDED against the joint pose — keeping the polish's accuracy but
 * rejecting any re-solve that flips back to the mirror the joint solve removed. Returns true if at least
 * one view accepted the joint pose. */
static bool
device_submit_joint_pose(struct t_constellation_tracker *ct,
                         struct tracking_sample_device_state *dev_state,
                         struct constellation_tracking_sample *sample,
                         const struct xrt_pose *P_imu_obj,
                         int contributing_views)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	const uint8_t telem_dev = telem_device_id(device->connection->xdev);
	bool submitted_any = false;

	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		struct constellation_tracker_camera_state *cam = ct->cam + view_id;
		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}

		/* Per-view camera-relative pose from the rig-frame joint solution. */
		struct xrt_pose P_cam_imu;
		math_pose_invert(&cam->P_imu_cam, &P_cam_imu);
		struct xrt_pose P_cam_obj;
		math_pose_transform(&P_cam_imu, P_imu_obj, &P_cam_obj);

		struct xrt_pose P_cam_obj_prior;
		math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);

		/* prior_must_match=true: a view accepts the joint pose only when it scores GOOD AND is consistent
		 * with the prior (position + orientation within the live covariance bounds) — the same canonical
		 * gate device_solve_view_from_labelled uses. A joint solution that fits the rays but disagrees with
		 * the prior in a view (the marginal wrong-branch / extrinsic-biased case) is not committed there. */
		pose_metrics_evaluate_pose_with_prior(&dev_state->score, &P_cam_obj, true, &P_cam_obj_prior,
		                                      &dev_state->prior_pos_error, &dev_state->prior_rot_error,
		                                      view->bwobs->blobs, view->bwobs->num_blobs, &device->led_model,
		                                      &cam->camera_model, NULL);

		if (POSE_HAS_FLAGS(&dev_state->score, POSE_MATCH_GOOD)) {
			submit_device_pose(ct, dev_state, sample, view_id, &P_cam_obj, false /* normal accept */,
			                   true /* joint seed: flip-guard the polish */);
			submitted_any = true;
		}
	}

	if (submitted_any && g2_telem_enabled()) {
		/* The joint multi-camera solve resolved this device this frame (value = contributing cameras). */
		g2_telem_event(telem_dev, (uint64_t)sample->timestamp, 11 /* joint_pnp */, (float)contributing_views);
	}
	return submitted_any;
}

/* Multi-camera JOINT (generalised / non-central) PnP fast path. A single camera that sees only a
 * near-planar LED subset has the intrinsic front/back mirror two-fold; the soft prior cost can only
 * re-RANK it away. When 2+ cameras co-see the controller, pooling their LED bearing rays through the
 * rig extrinsics into ONE generalised PnP constrains the pose jointly and the mirror dissolves by
 * construction (the flip no longer fits both cameras). We seed labels in every view from the predicted
 * pose, pool the views that carry >= 3 labelled LEDs, and if >= 2 such
 * views exist run the joint solve in the rig frame seeded by the prior. On success the joint pose is
 * committed to all contributing views; otherwise we return false and the single-camera fast paths run
 * (only one camera sees the device). */
static bool
device_try_joint_pnp(struct t_constellation_tracker *ct,
                     struct tracking_sample_device_state *dev_state,
                     struct constellation_tracking_sample *sample)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;

	/* Snapshot this device's blob labels across all views before the predicted-pose labelling below, so a
	 * DECLINED joint attempt restores them exactly — the joint path must be a no-op when it doesn't commit
	 * (it labels views the single-camera cascade might not have reached, and that must not leak into the
	 * frame-to-frame label carry / the prior-refine that follows). */
	uint16_t saved_led_id[CONSTELLATION_MAX_CAMERAS][MAX_BLOBS_PER_FRAME];
	uint16_t saved_prev_led_id[CONSTELLATION_MAX_CAMERAS][MAX_BLOBS_PER_FRAME];
	for (int v = 0; v < sample->n_views; v++) {
		struct tracking_sample_frame *view = sample->views + v;
		if (view->bwobs == NULL) {
			continue;
		}
		for (int b = 0; b < view->bwobs->num_blobs && b < MAX_BLOBS_PER_FRAME; b++) {
			saved_led_id[v][b] = view->bwobs->blobs[b].led_id;
			saved_prev_led_id[v][b] = view->bwobs->blobs[b].prev_led_id;
		}
	}

	struct joint_pnp_view jviews[JOINT_PNP_MAX_VIEWS];
	int n_jviews = 0;
	int n_multi = 0; /* views carrying enough labelled LEDs to contribute a constraint */

	for (int view_id = 0; view_id < sample->n_views && n_jviews < JOINT_PNP_MAX_VIEWS; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		struct constellation_tracker_camera_state *cam = ct->cam + view_id;
		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}

		/* Seed labels from the predicted pose, so the joint solve has correspondences in
		 * each co-visible view without depending on frame-to-frame label carry. */
		const int n_labelled = device_propagate_labels_in_view(ct, dev_state, sample, view_id);
		if (n_labelled < 3) {
			continue; /* too few rays in this view to help constrain the joint solve */
		}
		jviews[n_jviews].blobs = view->bwobs->blobs;
		jviews[n_jviews].num_blobs = view->bwobs->num_blobs;
		jviews[n_jviews].calib = &cam->camera_model;
		jviews[n_jviews].P_imu_cam = cam->P_imu_cam;
		n_jviews++;
		n_multi++;
	}

	/* Restore the saved labels and decline (a no-op). */
	bool declined = false;

	/* Only worthwhile when at least two cameras co-see the controller — the case that makes the joint
	 * solve disambiguate. With one (or zero) the geometry is the single-camera mirror case; decline. */
	if (n_multi < 2) {
		declined = true;
	}

	struct constellation_tracker_camera_state *seed_cam = NULL;
	struct tracking_sample_frame *seed_vw = NULL;
	struct xrt_pose P_cam_obj_prior = XRT_POSE_IDENTITY, P_cam_imu, P_imu_obj = XRT_POSE_IDENTITY;
	int num_rays = 0, num_inliers = 0;

	if (!declined) {
		/* The prior in the rig (IMU) frame: P_imu_obj = P_imu_cam[v] . P_cam_world . P_world_obj for any
		 * contributing view (they agree). Use the first contributing view to build it. */
		int seed_view = -1;
		for (int view_id = 0; view_id < sample->n_views; view_id++) {
			struct tracking_sample_frame *view = sample->views + view_id;
			if (view->bwobs != NULL && view->bwobs->num_blobs > 0) {
				seed_view = view_id;
				break;
			}
		}
		seed_cam = ct->cam + seed_view;
		seed_vw = sample->views + seed_view;
		math_pose_transform(&seed_vw->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);
		math_pose_invert(&seed_cam->P_imu_cam, &P_cam_imu);
		/* P_imu_obj = P_imu_cam . P_cam_obj (camera-in-rig . object-in-camera); P_cam_imu below maps back. */
		math_pose_transform(&seed_cam->P_imu_cam, &P_cam_obj_prior, &P_imu_obj);

		if (!joint_pnp_solve(&P_imu_obj, jviews, n_jviews, &device->led_model, &num_rays, &num_inliers)) {
			declined = true; /* not enough cross-camera information / no consistent joint fit */
		}
	}

	/* Gate the commit by the DRIFTLESS gravity-anchored prior tilt, the same cue the rest of the
	 * front-end uses to reject flips: the joint solve dissolves the mirror by construction, but on a
	 * marginal frame its dual-seed could still settle on the wrong branch. Only pre-empt the single-camera
	 * cascade when the joint orientation's tilt agrees with the prior (within GRAVITY_TILT_TOL) — when the
	 * fusion is tracking (tilt-trusted). Otherwise defer to the cascade rather than commit a possibly-
	 * flipped joint pose. (Cold start: no trusted tilt -> let the cascade/ab-initio bootstrap run.) */
	if (!declined && dev_state->prior_tilt_trusted) {
		struct xrt_pose P_cam_obj_joint;
		math_pose_transform(&P_cam_imu, &P_imu_obj, &P_cam_obj_joint); /* P_cam_obj = inv(P_imu_cam) . P_imu_obj */
		double tilt_rad = 0.0, yaw_rad = 0.0;
		pose_metrics_prior_orient_split(&P_cam_obj_joint.orientation, &P_cam_obj_prior.orientation,
		                                &seed_vw->cam_gravity_vector, &tilt_rad, &yaw_rad);
		if (tilt_rad > GRAVITY_TILT_TOL) {
			declined = true; /* the joint pose is on the wrong tilt branch -> defer to the cascade */
		}
	}

	if (declined) {
		/* Restore the labels the predicted-pose seeding wrote, so a non-committing joint attempt leaves the
		 * blob state untouched for the single-camera cascade and the frame-to-frame carry (no leak). */
		for (int v = 0; v < sample->n_views; v++) {
			struct tracking_sample_frame *view = sample->views + v;
			if (view->bwobs == NULL) {
				continue;
			}
			for (int b = 0; b < view->bwobs->num_blobs && b < MAX_BLOBS_PER_FRAME; b++) {
				view->bwobs->blobs[b].led_id = saved_led_id[v][b];
				view->bwobs->blobs[b].prev_led_id = saved_prev_led_id[v][b];
			}
		}
		return false;
	}

	return device_submit_joint_pose(ct, dev_state, sample, &P_imu_obj, n_multi);
}

/* Prior-refine fast path: seed blob labels from the IMU-predicted pose and solve per
 * view. This lets a slightly-off prior be pulled to a full PnP solve instead of falling through to the
 * slow ab-initio search: the main cure for the low accept rate and the re-acquisition dropout/snap.
 * Mislabels from a bad prior are rejected by the prior-consistency gate inside
 * device_solve_view_from_labelled, so it only ever helps. */
static bool
device_try_prior_refine(struct t_constellation_tracker *ct,
                        struct tracking_sample_device_state *dev_state,
                        struct constellation_tracking_sample *sample)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	const uint8_t telem_dev = telem_device_id(device->connection->xdev);

	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}

		const int n_labelled = device_propagate_labels_in_view(ct, dev_state, sample, view_id);

		if (device_solve_view_from_labelled(ct, dev_state, sample, view_id, false /* normal accept */)) {
			if (g2_telem_enabled()) {
				/* This device was resolved by predicted-pose label propagation (value = LEDs propagated)
				 * rather than falling through to the ab-initio search. */
				g2_telem_event(telem_dev, (uint64_t)sample->timestamp, 10 /* label_propagated */,
				               (float)n_labelled);
			}
			return true;
		}
	}
	return false;
}

/* Partial-information fold with a covariance gate.
 *
 * The fast paths require >=4 cleanly-PnP-able LEDs in a single camera; below that they bail and the
 * frame's optical information is DISCARDED to the slow ab-initio search (the dominant fall-through,
 * ~92% of fast bails — a visibility limit, edge-on/far/occluded controller). This folds the few LEDs
 * that ARE confidently matched to the prior, WITHOUT committing a pose: for each prior-visible LED we
 * ask the fusion for its predicted image point + 2x2 innovation covariance S = H·P·Hᵀ + R
 * (predict_led_gate), then accept the blob whose Mahalanobis distance d²=rᵀS⁻¹r is smallest AND
 * ≤ χ²₂(0.99). Only in-gate LEDs are folded (the ESKF then grows covariance honestly on 1-3 LEDs).
 *
 * Guards (the contract):
 *  - COLD START / no prior: predict_led_gate returns false until the fusion is tracking, AND we require
 *    prior_tilt_trusted (= the fusion is tracking, gravity-anchored prior available). Untracked => no-op
 *    => the existing ab-initio bootstrap runs unchanged. Never folds against an unreliable reference.
 *  - MISLABEL: the anisotropic S is the guard. A flipped/garbage correspondence reprojects far from
 *    zhat (in tilt especially — S is tight there), so d² blows past the gate and the LED is NOT folded.
 *    The per-LED χ² gate inside fold_led_observations is the second, consistent line of defence.
 *  - Each blob is assigned to at most one LED: LED-order greedy (LEDs scanned in index order, each claims
 *    its min-d² of the still-free in-gate blobs; blob_taken[] enforces one blob per LED and one LED per
 *    blob). Deterministic; harmless with 1-3 sparse LEDs (each fold is re-gated by the ESKF's own χ²).
 *
 * Does NOT report a pose (no submit_device_pose, no PnP "accept"); it only feeds the filter, so the
 * device keeps reporting its covariance-grown prior and the frame still falls through to ab-initio for a
 * genuine (re)acquire. Returns true iff at least one LED was gate-folded (diagnostic). A <4-LED frame is
 * used productively here; the accept/flip decision stays with the anisotropic prior-cost paths (the
 * ab-initio few-LED accept is flip-ranked by the same prior split). */
static bool
device_try_partial_fold(struct t_constellation_tracker *ct,
                        struct tracking_sample_device_state *dev_state,
                        struct constellation_tracking_sample *sample)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;

	/* No reliable prior to gate with (cold start) -> no-op, fall back to ab-initio. The covariance gate
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
		 * when fewer than ROI_FALLBACK_MIN_PREDICTIONS LEDs project (cold start / not tracked). The
		 * scan pad is per-LED-adaptive (max of the base pad floor and k * sqrt(max-diag(S))); base
		 * pad and sigma multiplier are env-tunable for in-headset re-tuning across lighting/motion. */
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
			for (int d = 0; d < ct->num_devices; d++) {
				struct constellation_tracker_device *device = &ct->devices[d];
				const struct t_constellation_led_model *lm = &device->led_model;
				if (device->connection == NULL || lm->leds == NULL || lm->num_leds == 0) {
					continue; /* not connected / model not loaded yet */
				}
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
					}
				}
			}
			if (n_in_frame >= ROI_FALLBACK_MIN_PREDICTIONS) {
				/* xmin/ymin/xmax/ymax already carry per-LED pad; no extra global pad needed. */
				int rx = (int)floorf(xmin);
				int ry = (int)floorf(ymin);
				int rw = (int)ceilf(xmax - xmin) + 1;
				int rh = (int)ceilf(ymax - ymin) + 1;
				blobwatch_process_roi(cam->bw, view->vframe, frame_exposure, 0, rx, ry, rw, rh,
				                      &view->bwobs);
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

	/* Try pose recovery strategies:
	 * 	Check the predicted pose
	 * 	Check the last seen pose
	 * 	Try recovery from labelled blobs
	 * 	@todo: Check the predicted orientation, but at the last seen position
	 * 	@todo: Try for a translational match with prior orientation
	 */
	bool need_full_search = false;
	for (int i = 0; i < sample->n_devices; i++) {
		struct tracking_sample_device_state *dev_state = sample->devices + i;
		struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;

		CT_DEBUG(ct, "Doing fast match search for device %d", device->led_model.id);

		/* When 2+ cameras co-see the controller, solve ONE generalised (non-central) PnP over their pooled
		 * LED bearing rays FIRST: the cross-camera baseline dissolves the single-camera front/back mirror by
		 * construction (the flip cannot fit both cameras), so a multi-cam frame is resolved by the
		 * disambiguated joint pose and never commits a single-camera flip. The commit is heavily gated
		 * (>= 2 cameras each strongly supporting it, prior-position+orientation match, gravity-anchored tilt
		 * agreement) and is a no-op (labels restored, false returned) otherwise — so a marginal frame falls
		 * straight through to the single-camera cascade below. */
		if (device_try_joint_pnp(ct, dev_state, sample)) {
			CT_DEBUG(ct, "Resolved device %d by multi-camera joint PnP in view %d", device->led_model.id,
			         dev_state->found_pose_view_id);
			continue;
		}
		if (device_try_global_pose(ct, dev_state, sample, &dev_state->P_world_obj_prior)) {
			CT_DEBUG(ct, "Found fast match search for device %d in view %d from prior pose",
			         device->led_model.id, dev_state->found_pose_view_id);
			continue;
		}
		if (dev_state->have_last_seen_pose &&
		    device_try_global_pose(ct, dev_state, sample, &dev_state->last_seen_pose)) {
			CT_DEBUG(ct, "Found fast match search for device %d in view %d from last_seen pose",
			         device->led_model.id, dev_state->found_pose_view_id);
			continue;
		}
		if (device_try_recover_pose(ct, dev_state, sample)) {
			CT_DEBUG(ct, "Found fast match search for device %d in view %d from labelled blobs",
			         device->led_model.id, dev_state->found_pose_view_id);
			continue;
		}
		/* Last fast option before the slow ab-initio search: re-label the blobs from the predicted
		 * pose and solve. Catches the common "prior slightly off, blobs present, tracking labels
		 * lost" case that the verbatim-prior checks above miss — the main accept-rate/dropout win. */
		if (device_try_prior_refine(ct, dev_state, sample)) {
			CT_DEBUG(ct, "Refined device %d from the predicted pose in view %d", device->led_model.id,
			         dev_state->found_pose_view_id);
			continue;
		}
		/* No PnP-able pose this frame (the <4-LED visibility wall). Before discarding the frame to the
		 * slow ab-initio search, FOLD the individual LEDs that are confidently matched to the prior,
		 * each gated by the ESKF's anisotropic per-LED covariance. No-op at cold start (no prior). If it
		 * folds anything, the frame's information is used and we DON'T run the flip-prone ab-initio
		 * few-LED accept on the same LEDs (~92-100% of flips). If it can't fold (no prior / no in-gate
		 * LED), ab-initio still runs as the genuine (re)acquire fallback. */
		device_try_partial_fold(ct, dev_state, sample);
		if (!dev_state->found_device_pose) {
			need_full_search = true;
		}
	}

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

	/* Grab the 'do_full_search' value in case the user clicked the
	 * button in the debug UI */
	if (ct->do_full_search || need_full_search) {
		ct->do_full_search = false;

		/* Send the sample for long analysis */
		os_thread_helper_lock(&ct->long_analysis_thread);
		if (ct->long_analysis_pending_sample != NULL) {
			/* (b) a queued sample is dropped before the long thread ever processes it: that frame
			 * exits the pipeline here, so count it. */
			constellation_tracking_sample_free(ct->long_analysis_pending_sample);
			ct->frames_completed++;
		}
		ct->long_analysis_pending_sample = sample;
		os_thread_helper_signal_locked(&ct->long_analysis_thread);
		os_thread_helper_unlock(&ct->long_analysis_thread);
	} else {
		/* (a) fast path done, no long analysis: this frame exits the pipeline here. */
		os_thread_helper_lock(&ct->long_analysis_thread);
		ct->frames_completed++;
		os_thread_helper_unlock(&ct->long_analysis_thread);
		/* not sending for long analysis: free it */
		constellation_tracking_sample_free(sample);
	}
}

static void
constellation_tracker_process_frame_long(struct t_constellation_tracker *ct,
                                         struct constellation_tracking_sample *sample)
{
	CT_DEBUG(ct, "Starting long analysis of frame TS %" PRIu64, sample->timestamp);

	bool dev_found[CONSTELLATION_MAX_DEVICES] = {0};
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		struct constellation_tracker_camera_state *cam = ct->cam + view_id;

		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue; // no blobs in this view
		}

		correspondence_search_set_blobs(cam->cs, view->bwobs->blobs, view->bwobs->num_blobs);

		for (int d = 0; d < sample->n_devices; d++) {
			struct tracking_sample_device_state *dev_state = sample->devices + d;
			struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;

			//! The fast analysis thread must have retrieved the LED model before
			//  ever sending a device for long analysis
			assert(device->have_led_model == true);

			if (dev_state->found_device_pose)
				continue; /* This device was already found. No need for a long search */

			CT_DEBUG(ct, "Doing full search for device %d in view %d", device->led_model.id, view_id);

			for (int pass = 0; pass < 2; pass++) {

				if (dev_state->found_device_pose)
					break; /* This device was already found on the previous pass */

				enum correspondence_search_flags search_flags =
				    CS_FLAG_STOP_FOR_STRONG_MATCH | CS_FLAG_HAVE_POSE_PRIOR;

				/* THE soft anisotropic prior cost: enabled whenever the (gravity-anchored, driftless)
				 * prior TILT is trusted — i.e. the fusion is tracking. It out-ranks the tilt-flipped P3P
				 * twin even when the yaw prior is stale (the dropout/snap-back case); the yaw term widens as
				 * the live yaw sigma grows. Cold start (untracked) leaves it off, so the search runs on
				 * reprojection alone (cold reacquire). */
				if (dev_state->prior_tilt_trusted)
					search_flags |= CS_FLAG_TRUST_PRIOR_ORIENT;

				if (pass == 0) {
					/* 1st pass - quick search only */
					search_flags |= CS_FLAG_SHALLOW_SEARCH;
				} else {
					/* 2nd pass - do a deep search */
					search_flags |= CS_FLAG_DEEP_SEARCH;
				}

				struct xrt_pose P_cam_obj;
				math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj);

				/* tilt scale = the tight driftless GRAVITY_TILT_TOL; yaw scale = the live fusion yaw
				 * 1-sigma (large when stale -> the yaw term vanishes); Huber knee + weight as derived. */
				if (correspondence_search_find_one_pose(
				        cam->cs, device->search_led_model, search_flags, &P_cam_obj,
				        &dev_state->prior_pos_error, &dev_state->prior_rot_error, &view->cam_gravity_vector,
				        (float)GRAVITY_TILT_TOL, dev_state->prior_yaw_sigma_rad,
				        (float)FLIP_COST_HUBER_KNEE_SIGMA, (float)FLIP_COST_WEIGHT, &dev_state->score)) {
					CT_DEBUG(ct, "Found a pose on cam %u device %d long search pass %d", view_id,
					         device->led_model.id, pass);
					submit_device_pose(ct, dev_state, sample, view_id, &P_cam_obj, false /* not a recovery */,
					                   false /* single-cam: take the polish unconditionally */);
					dev_found[d] = true;
					break;
				}
			}
		}
	}

	for (int d = 0; d < sample->n_devices; d++) {
		if (!dev_found[d]) {
			// if a long analysis did not find the device at all, then we push that it has no brightness
			struct constellation_tracker_device *lost_dev = &ct->devices[sample->devices[d].dev_index];
			constellation_tracked_device_connection_notify_brightness_update(lost_dev->connection, 0);

			/* Telemetry: a device that was previously tracked could not be found at all
			 * in this frame (fast + long search both failed) -> lock lost. */
			if (sample->devices[d].have_last_seen_pose) {
				g2_telem_event(telem_device_id(lost_dev->connection->xdev),
				               (uint64_t)sample->timestamp, 0 /* lock_lost */, 0.0f);
			}

			// update the controller masks for this controller to mark it as not active
			if (ct->controller_masks_sink) {

				for (int i = 0; i < ct->cam_count; i++) {
					struct constellation_tracker_camera_state *cam = &ct->cam[i];

					struct xrt_device_masks_sample_camera *sample_camera =
					    &ct->controller_masks_sample.views[cam->slam_tracking_index];

					struct xrt_device_masks_sample_device *device_mask =
					    &sample_camera->devices[sample->devices[d].dev_index];

					device_mask->enabled = false;
				}

				xrt_sink_push_device_masks(ct->controller_masks_sink, &ct->controller_masks_sample);
			}
		}
	}
}

static void *
constellation_tracking_long_analysis_thread(void *ptr)
{
	U_TRACE_SET_THREAD_NAME("Constellation tracker: device long recovery thread");
	struct t_constellation_tracker *ct = (struct t_constellation_tracker *)(ptr);

	os_thread_helper_lock(&ct->long_analysis_thread);
	while (os_thread_helper_is_running_locked(&ct->long_analysis_thread)) {
		/* Wait for a sample to analyse, or for shutdown */
		if (ct->long_analysis_pending_sample == NULL) {
			os_thread_helper_wait_locked(&ct->long_analysis_thread);
			if (!os_thread_helper_is_running_locked(&ct->long_analysis_thread)) {
				break;
			}
		}

		/* Take ownership of any pending sample */
		struct constellation_tracking_sample *sample = ct->long_analysis_pending_sample;
		ct->long_analysis_pending_sample = NULL;

		os_thread_helper_unlock(&ct->long_analysis_thread);
		if (sample != NULL) {
			uint64_t long_analysis_start_ts = os_monotonic_get_ns();
			constellation_tracker_process_frame_long(ct, sample);
			uint64_t long_analysis_finish_ts = os_monotonic_get_ns();

			constellation_tracking_sample_free(sample);

			ct->last_long_analysis_ms =
			    (long_analysis_finish_ts - long_analysis_start_ts) / U_TIME_1MS_IN_NS;
		}

		os_thread_helper_lock(&ct->long_analysis_thread);
		if (sample != NULL) {
			/* (c) the long thread finished this frame: it exits the pipeline here. */
			ct->frames_completed++;
		}
	}
	os_thread_helper_unlock(&ct->long_analysis_thread);

	return NULL;
}

static void
constellation_tracker_node_destroy(struct xrt_frame_node *node)
{
	struct t_constellation_tracker *ct = container_of(node, struct t_constellation_tracker, node);

	DRV_TRACE_MARKER();
	CT_DEBUG(ct, "Destroying constellation tracker");

	/* Make sure the long analysis thread isn't running */
	os_thread_helper_stop_and_wait(&ct->long_analysis_thread);

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

	/* Clean up any pending sample */
	os_thread_helper_lock(&ct->long_analysis_thread);
	if (ct->long_analysis_pending_sample != NULL) {
		constellation_tracking_sample_free(ct->long_analysis_pending_sample);
		ct->long_analysis_pending_sample = NULL;
	}
	os_thread_helper_unlock(&ct->long_analysis_thread);
	/* Then release the thread helper */
	os_thread_helper_destroy(&ct->long_analysis_thread);

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

static void
ct_full_search_btn_cb(void *ct_ptr)
{
	struct t_constellation_tracker *ct = (struct t_constellation_tracker *)ct_ptr;
	ct->do_full_search = true;
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
	ct->frames_completed = 0;

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

	ret = os_thread_helper_init(&ct->long_analysis_thread);
	if (ret != 0) {
		CT_ERROR(ct, "constellation tracker: Failed to init long analysis thread");
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

	// Long match recovery thread
	ret = os_thread_helper_start(&ct->long_analysis_thread, constellation_tracking_long_analysis_thread, ct);
	if (ret != 0) {
		CT_ERROR(ct, "constellation tracker: Failed to start long analysis thread!");
		constellation_tracker_node_destroy(&ct->node);
		return -1;
	}

	// Debug UI
	ct->full_search_button.cb = ct_full_search_btn_cb;
	ct->full_search_button.ptr = ct;

	u_var_add_root(ct, "Constellation Tracker", false);
	u_var_add_log_level(ct, &ct->log_level, "Log Level");
	u_var_add_ro_i32(ct, &ct->num_devices, "Num Devices");
	u_var_add_ro_u64(ct, &ct->last_frame_timestamp, "Last Frame Timestamp");
	u_var_add_ro_u64(ct, &ct->last_blob_analysis_ms, "Blob tracking time (ms)");
	u_var_add_ro_u64(ct, &ct->last_fast_analysis_ms, "Fast analysis time (ms)");
	u_var_add_ro_u64(ct, &ct->last_long_analysis_ms, "Long analysis time (ms)");
	u_var_add_button(ct, &ct->full_search_button, "Trigger ab-initio search");

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
	os_thread_helper_lock(&ct->long_analysis_thread);
	uint64_t n = ct->frames_completed;
	os_thread_helper_unlock(&ct->long_analysis_thread);
	return n;
}
