// Copyright 2023 Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Implementation of tunnelled controller connection,
 * that translates messages passing via an HP G2 or Sasmung Odyssey+ HMD
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup drv_wmr
 */
#include <inttypes.h>
#include <stdio.h>

#include "os/os_threading.h"

#include "util/u_debug.h"
#include "util/u_frame.h"
#include "util/u_sink.h"
#include "util/u_trace_marker.h"

#include "constellation/blobwatch.h"
#include "constellation/camera_model.h"
#include "constellation/correspondence_search.h"
#include "constellation/debug_draw.h"
#include "constellation/led_models.h"
#include "constellation/ransac_pnp.h"
#include "constellation/sample.h"

#include "wmr_controller_tracking.h"
#include "wmr_hmd.h"

DEBUG_GET_ONCE_LOG_OPTION(wmr_log, "WMR_LOG", U_LOGGING_INFO)

#define RAD_TO_DEG(RAD) ((RAD)*180. / M_PI)
#define DEG_TO_RAD(DEG) ((DEG)*M_PI / 180.)

#define MIN_ROT_ERROR DEG_TO_RAD(30)
#define MIN_POS_ERROR 0.10

#define WMR_TRACE(c, ...) U_LOG_IFL_T(c->log_level, __VA_ARGS__)
#define WMR_DEBUG(c, ...) U_LOG_IFL_D(c->log_level, __VA_ARGS__)
#define WMR_INFO(c, ...) U_LOG_IFL_I(c->log_level, __VA_ARGS__)
#define WMR_WARN(c, ...) U_LOG_IFL_W(c->log_level, __VA_ARGS__)
#define WMR_ERROR(c, ...) U_LOG_IFL_E(c->log_level, __VA_ARGS__)

/* WMR thresholds for min brightness and min-blob-required magnitude */
#define BLOB_PIXEL_THRESHOLD_WMR 0x4
#define BLOB_THRESHOLD_MIN_WMR 0x8

/* Maximum number of frames to permit waiting in the fast-processing queue */
#define MAX_FAST_QUEUE_SIZE 2

//! When projecting poses into the camera, we need
// extra flip around the X axis because OpenCV camera space coordinates have
// +Y down and +Z away from the user
static void
pose_flip_YZ(const struct xrt_pose *in, struct xrt_pose *dest)
{
	const struct xrt_pose P_YZ_flip = {
	    {1.0, 0.0, 0.0, 0.0},
	    {0.0, 0.0, 0.0},
	};

	struct xrt_pose tmp;
	math_pose_transform(&P_YZ_flip, in, &tmp);
	math_pose_transform(&tmp, &P_YZ_flip, dest);
}

struct wmr_controller_tracker_connection
{
	/* Controller and tracker each hold a reference. It's
	 * only cleaned up once both release it. */
	struct xrt_reference ref;

	/* Index in the devices array for this device */
	int id;

	/* Protect access when around API calls and disconnects */
	struct os_mutex lock;
	bool disconnected; /* Set to true once disconnect() is called */

	struct wmr_controller_base *wcb;        //! Controller instance
	struct wmr_controller_tracker *tracker; //! Tracker instance
};

struct wmr_controller_tracker_device
{
	struct wmr_controller_tracker_connection *connection;

	bool have_led_model;
	struct constellation_led_model led_model;
	struct constellation_search_model *search_led_model;

	bool have_last_seen_pose;
	uint64_t last_seen_pose_ts;
	struct xrt_pose last_seen_pose; // global pose
	int last_matched_blobs;
	int last_matched_cam;
	struct xrt_pose last_matched_cam_pose; // Camera-relative pose
};

struct wmr_controller_tracker_camera
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
};

/*!
 * An @ref xrt_frame_sink that analyses video frame groups for LED constellation tracking
 * @implements xrt_frame_sink
 * @implements xrt_frame_node
 */
struct wmr_controller_tracker
{
	//! Receive (mosaic) frames from the camera
	struct xrt_frame_sink base;
	//! frame node to insert in the xfctx
	struct xrt_frame_node node;

	/*! HMD device we get observation base poses from
	 * and that owns the xfctx keeping this node alive */
	struct xrt_device *hmd_xdev;

	struct os_mutex tracked_controller_lock;

	//! Controller communication connections
	int num_controllers;
	struct wmr_controller_tracker_device controllers[WMR_MAX_CONTROLLERS];

	//!< Tracking camera entries
	struct wmr_controller_tracker_camera cam[WMR_MAX_CAMERAS];
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

	uint64_t last_frame_timestamp;
	uint64_t last_frame_sequence;

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
};

static void
wmr_controller_tracker_connection_notify_timesync(struct wmr_controller_tracker_connection *wctc,
                                                  timepoint_ns frame_mono_ns)
{
	os_mutex_lock(&wctc->lock);
	if (!wctc->disconnected) {
		wmr_controller_base_notify_timesync(wctc->wcb, frame_mono_ns);
	}
	os_mutex_unlock(&wctc->lock);
}

static void
wmr_controller_tracker_connection_notify_pose(struct wmr_controller_tracker_connection *wctc,
                                              timepoint_ns frame_mono_ns,
                                              const struct xrt_pose *pose)
{
	os_mutex_lock(&wctc->lock);
	if (!wctc->disconnected) {
		wmr_controller_base_push_observed_pose(wctc->wcb, frame_mono_ns, pose);
	}
	os_mutex_unlock(&wctc->lock);
}

static bool
wmr_controller_tracker_connection_get_led_model(struct wmr_controller_tracker_connection *wctc,
                                                struct constellation_led_model *led_model)
{
	bool ret = false;

	os_mutex_lock(&wctc->lock);
	if (!wctc->disconnected) {
		ret = wmr_controller_base_get_led_model(wctc->wcb, led_model);
	}
	os_mutex_unlock(&wctc->lock);

	return ret;
}

static bool
wmr_controller_tracker_connection_get_tracked_pose(struct wmr_controller_tracker_connection *wctc,
                                                   uint64_t timestamp_ns,
                                                   struct xrt_space_relation *xsr)
{
	bool ret = false;

	os_mutex_lock(&wctc->lock);
	if (!wctc->disconnected) {
		struct xrt_device *xdev = (struct xrt_device *)(wctc->wcb);
		xrt_device_get_tracked_pose(xdev, XRT_INPUT_GENERIC_TRACKER_POSE, timestamp_ns, xsr);
		ret = true;
	}
	os_mutex_unlock(&wctc->lock);

	return ret;
}

static void
wmr_controller_tracker_receive_frame(struct xrt_frame_sink *sink, struct xrt_frame *xf)
{
	struct wmr_controller_tracker *wct = container_of(sink, struct wmr_controller_tracker, base);

	assert(xf->format == XRT_FORMAT_L8);

	/* The frame cadence is SLAM/controller/controller. Only controller frames are received
	 * here. On the 2nd controller frame, we need to pass the estimate of the time of the next
	 * *1st controller frame* minus 1/3 of a frame duration to the controller timesync methods.
	 * That is 2/90 + 1/270 = 5/270 or 5/3rds of the average frame duration.
	 * @todo: Also calculate how many full 3-frame cycles have passed since the last
	 * '2nd Controller frame' to add to our estimate in case of scheduling delays.
	 * @todo: Move the 'next frame' estimate into the controller timesync code so it's
	 * calculated when the led control packet is actually about to be sent.
	 * Since only controller frames get passed to here, the 2nd sequential controller
	 * frame is the one right before the next SLAM frame */
	bool is_second_frame = (wct->last_frame_sequence + 1) == xf->source_sequence;

	os_mutex_lock(&wct->tracked_controller_lock);

	// Update the controller timesync estimate
	if (is_second_frame) {
		timepoint_ns next_timesync_ts = (timepoint_ns)(xf->timestamp) + U_TIME_1MS_IN_NS * 18;

		for (int i = 0; i < wct->num_controllers; i++) {
			wmr_controller_tracker_connection_notify_timesync(wct->controllers[i].connection,
			                                                  next_timesync_ts);
		}
	}

	os_mutex_unlock(&wct->tracked_controller_lock);

	wct->last_frame_timestamp = xf->timestamp;
	wct->last_frame_sequence = xf->source_sequence;

	xrt_sink_push_frame(wct->fast_q_sink, xf);
}

static void
wmr_controller_tracker_node_break_apart(struct xrt_frame_node *node)
{
	DRV_TRACE_MARKER();
}

static void
mark_matching_blobs(struct wmr_controller_tracker *wct,
                    struct xrt_pose *pose,
                    struct blobservation *bwobs,
                    struct constellation_led_model *led_model,
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
		struct constellation_led *led = led_info->led;

		if (led_info->matched_blob != NULL) {
			struct blob *b = led_info->matched_blob;

			b->led_id = LED_MAKE_ID(led_model->id, led->id);
			WMR_DEBUG(wct, "Marking LED %d/%d at %f,%f angle %f now %d (was %d)", led_model->id, led->id,
			          b->x, b->y, RAD_TO_DEG(acosf(led_info->facing_dot)), b->led_id, b->prev_led_id);
		} else {
			WMR_DEBUG(wct, "No blob for device %d LED %d @ %f,%f size %f px angle %f", led_model->id,
			          led->id, led_info->pos_px.x, led_info->pos_px.y, 2 * led_info->led_radius_px,
			          RAD_TO_DEG(acosf(led_info->facing_dot)));
		}
	}
}

static void
submit_device_pose(struct wmr_controller_tracker *wct,
                   struct tracking_sample_device_state *dev_state,
                   struct constellation_tracking_sample *sample,
                   int view_id,
                   struct xrt_pose *P_cam_obj)
{
	struct wmr_controller_tracker_camera *cam = wct->cam + view_id;
	struct tracking_sample_frame *view = sample->views + view_id;
	struct wmr_controller_tracker_device *device = wct->controllers + dev_state->dev_index;

	struct pose_metrics *score = &dev_state->score;

	pose_metrics_match_pose_to_blobs(P_cam_obj, view->bwobs->blobs, view->bwobs->num_blobs, &device->led_model,
	                                 &cam->camera_model, &dev_state->blob_match_info);
	mark_matching_blobs(wct, P_cam_obj, view->bwobs, &device->led_model, &dev_state->blob_match_info);

	os_mutex_lock(&cam->bw_lock);
	blobwatch_update_labels(cam->bw, view->bwobs, device->led_model.id);
	os_mutex_unlock(&cam->bw_lock);

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

		WMR_DEBUG(wct,
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

		WMR_DEBUG(wct,
		          "Found an extra pose on cam %u device %d score match_flags 0x%x matched %u "
		          "blobs of %u visible LEDs. Global pose %f,%f,%f,%f pos %f,%f,%f "
		          "cam-relative pose %f,%f,%f,%f pos %f,%f,%f "
		          "delta from 1st pose %f,%f,%f,%f pos %f,%f,%f",
		          view_id, device->led_model.id, score->match_flags, score->matched_blobs, score->visible_leds,
		          extra_final_pose.orientation.x, extra_final_pose.orientation.y,
		          extra_final_pose.orientation.z, extra_final_pose.orientation.w, extra_final_pose.position.x,
		          extra_final_pose.position.y, extra_final_pose.position.z, P_cam_obj->orientation.x,
		          P_cam_obj->orientation.y, P_cam_obj->orientation.z, P_cam_obj->orientation.w,
		          P_cam_obj->position.x, P_cam_obj->position.y, P_cam_obj->position.z, pose_delta.orientation.x,
		          pose_delta.orientation.y, pose_delta.orientation.z, pose_delta.orientation.w,
		          pose_delta.position.x, pose_delta.position.y, pose_delta.position.z);

		// Mix the found position with the prior one
		math_vec3_scalar_mul(0.5, &pose_delta.position);
		math_vec3_accum(&pose_delta.position, &dev_state->final_pose.position);
	}

	os_mutex_lock(&wct->tracked_controller_lock);
	if (device->have_last_seen_pose == false || sample->timestamp > device->last_seen_pose_ts) {
		device->have_last_seen_pose = true;
		device->last_seen_pose_ts = sample->timestamp;
		device->last_seen_pose = dev_state->final_pose;
		device->last_matched_blobs = score->matched_blobs;
		device->last_matched_cam = view_id;
		device->last_matched_cam_pose = *P_cam_obj;

		/* Submit this pose observation to the fusion / real device. Flip back to OpenXR coords first */
		struct xrt_pose P_xrworld_obj;
		pose_flip_YZ(&dev_state->final_pose, &P_xrworld_obj);

		wmr_controller_tracker_connection_notify_pose(device->connection, sample->timestamp, &P_xrworld_obj);
	}
	os_mutex_unlock(&wct->tracked_controller_lock);
}

/* Fast matching test based on hypothesised pose */
static bool
device_try_global_pose(struct wmr_controller_tracker *wct,
                       struct tracking_sample_device_state *dev_state,
                       struct constellation_tracking_sample *sample,
                       struct xrt_pose *P_world_obj_candidate)
{
	bool ret = false;
	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		struct wmr_controller_tracker_camera *cam = wct->cam + view_id;

		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}

		struct wmr_controller_tracker_device *device = wct->controllers + dev_state->dev_index;
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
			submit_device_pose(wct, dev_state, sample, view_id, &P_cam_obj_candidate);
			ret = true;
		}
	}

	return ret;
}

/* Try and recover the pose from labelled blobs */
static bool
device_try_recover_pose(struct wmr_controller_tracker *wct,
                        struct tracking_sample_device_state *dev_state,
                        struct constellation_tracking_sample *sample)
{
	struct wmr_controller_tracker_device *device = wct->controllers + dev_state->dev_index;
	struct constellation_led_model *leds_model = &device->led_model;

	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		struct wmr_controller_tracker_camera *cam = wct->cam + view_id;

		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}

		struct wmr_controller_tracker_device *device = wct->controllers + dev_state->dev_index;
		blobservation *bwobs = view->bwobs;

		/* See if we still have enough labelled blobs to try to re-acquire the pose without a
		 * full search */
		int num_blobs = 0;
		for (int index = 0; index < bwobs->num_blobs; index++) {
			struct blob *b = bwobs->blobs + index;
			if (LED_OBJECT_ID(b->led_id) == leds_model->id) {
				num_blobs++;
			}
		}
		if (num_blobs < 4) {
			continue;
		}

		struct xrt_pose P_cam_obj_prior;
		math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj_prior);

		WMR_DEBUG(wct,
		          "Camera %d trying to reacquire device %d from %u blobs and cam-relative pose %f %f %f %f pos "
		          "%f %f %f "
		          "global prior pose %f %f %f %f pos %f %f %f",
		          view_id, leds_model->id, num_blobs, P_cam_obj_prior.orientation.x,
		          P_cam_obj_prior.orientation.y, P_cam_obj_prior.orientation.z, P_cam_obj_prior.orientation.w,
		          P_cam_obj_prior.position.x, P_cam_obj_prior.position.y, P_cam_obj_prior.position.z,
		          dev_state->P_world_obj_prior.orientation.x, dev_state->P_world_obj_prior.orientation.y,
		          dev_state->P_world_obj_prior.orientation.z, dev_state->P_world_obj_prior.orientation.w,
		          dev_state->P_world_obj_prior.position.x, dev_state->P_world_obj_prior.position.y,
		          dev_state->P_world_obj_prior.position.z);

		struct xrt_pose P_cam_obj = P_cam_obj_prior;

		if (!ransac_pnp_pose(&P_cam_obj, bwobs->blobs, bwobs->num_blobs, leds_model, &cam->camera_model, NULL,
		                     NULL)) {
			WMR_DEBUG(wct, "Camera %d RANSAC-PnP for device %d from %u blobs failed", view_id,
			          leds_model->id, num_blobs);
			continue;
		}

		pose_metrics_evaluate_pose_with_prior(&dev_state->score, &P_cam_obj, true, &P_cam_obj_prior,
		                                      &dev_state->prior_pos_error, &dev_state->prior_rot_error,
		                                      bwobs->blobs, bwobs->num_blobs, &device->led_model,
		                                      &cam->camera_model, NULL);

		if (POSE_HAS_FLAGS(&dev_state->score, POSE_MATCH_GOOD)) {
			submit_device_pose(wct, dev_state, sample, view_id, &P_cam_obj);
			return true;
		}
		WMR_DEBUG(wct,
		          "Camera %d device %d had %d prior blobs, but failed match with flags 0x%x. "
		          "Yielded pose %f %f %f %f pos %f %f %f (match %d of %d visible) rot_error %f %f %f pos_error "
		          "%f %f %f",
		          view_id, leds_model->id, num_blobs, dev_state->score.match_flags, P_cam_obj.orientation.x,
		          P_cam_obj.orientation.y, P_cam_obj.orientation.z, P_cam_obj.orientation.w,
		          P_cam_obj.position.x, P_cam_obj.position.y, P_cam_obj.position.z,
		          dev_state->score.matched_blobs, dev_state->score.visible_leds,
		          dev_state->score.orient_error.x, dev_state->score.orient_error.y,
		          dev_state->score.orient_error.z, dev_state->score.pos_error.x, dev_state->score.pos_error.y,
		          dev_state->score.pos_error.z);
	}

	return false;
}

// Fast frame processing: blob extraction and match to existing predictions
static void
wmr_controller_tracker_process_frame_fast(struct xrt_frame_sink *sink, struct xrt_frame *xf)
{
	struct wmr_controller_tracker *wct = container_of(sink, struct wmr_controller_tracker, fast_process_sink);
	struct xrt_space_relation xsr_base_pose;

	/* Allocate a tracking sample for everything we're about to process */
	struct constellation_tracking_sample *sample = constellation_tracking_sample_new();
	uint64_t fast_analysis_start_ts = os_monotonic_get_ns();

	WMR_DEBUG(wct, "Starting analysis of frame %" PRIu64 " TS %" PRIu64, xf->source_sequence, xf->timestamp);

	/* Get the HMD's pose so we can calculate the camera view poses */
	xrt_device_get_tracked_pose(wct->hmd_xdev, XRT_INPUT_GENERIC_TRACKER_POSE, xf->timestamp, &xsr_base_pose);

	/* Split out camera views and collect blobs across all cameras */
	assert(wct->cam_count <= CONSTELLATION_MAX_CAMERAS);
	sample->n_views = wct->cam_count;
	sample->timestamp = xf->timestamp;

	for (int i = 0; i < wct->cam_count; i++) {
		struct wmr_controller_tracker_camera *cam = wct->cam + i;
		struct tracking_sample_frame *view = sample->views + i;

		// Flip the input pose to CV coords, so we can do all our operations
		// in OpenCV coords
		struct xrt_pose P_cvworld_hmdimu;
		pose_flip_YZ(&xsr_base_pose.pose, &P_cvworld_hmdimu);

		math_pose_transform(&P_cvworld_hmdimu, &cam->P_imu_cam, &view->P_world_cam);

		WMR_DEBUG(wct,
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
		          view->P_world_cam.orientation.z, view->P_world_cam.orientation.w,
		          view->P_world_cam.position.x, view->P_world_cam.position.y, view->P_world_cam.position.z);

		// Calculate inverse from cam back to world coords
		math_pose_invert(&view->P_world_cam, &view->P_cam_world);

		const struct xrt_vec3 gravity_vector = {0.0, 1.0, 0.0};
		math_quat_rotate_vec3(&view->P_cam_world.orientation, &gravity_vector, &view->cam_gravity_vector);

		u_frame_create_roi(xf, cam->roi, &view->vframe);
		view->bw = cam->bw;

		os_mutex_lock(&cam->bw_lock);
		blobwatch_process(cam->bw, view->vframe, &view->bwobs);
		os_mutex_unlock(&cam->bw_lock);

		if (view->bwobs == NULL) {
			cam->last_num_blobs = 0;
			continue;
		}

		blobservation *bwobs = view->bwobs;
		cam->last_num_blobs = bwobs->num_blobs;

		WMR_TRACE(wct, "frame %" PRIu64 " TS %" PRIu64 " cam %d ROI %d,%d w/h %d,%d Blobs: %d",
		          xf->source_sequence, xf->timestamp, i, cam->roi.offset.w, cam->roi.offset.h,
		          cam->roi.extent.w, cam->roi.extent.h, bwobs->num_blobs);

#if 0
		for (int index = 0; index < bwobs->num_blobs; index++) {
			printf("  Blob[%d]: %f,%f %dx%d id %d age %u\n", index, bwobs->blobs[index].x,
			       bwobs->blobs[index].y, bwobs->blobs[index].width, bwobs->blobs[index].height,
			       bwobs->blobs[index].led_id, bwobs->blobs[index].age);
		}
#endif
	}
	uint64_t blob_extract_finish_ts = os_monotonic_get_ns();
	wct->last_blob_analysis_ms = (blob_extract_finish_ts - fast_analysis_start_ts) / U_TIME_1MS_IN_NS;

	// Ready to start processing device poses now. Make sure we have the
	// LED models and collect the best estimate of the current pose
	// for each target device
	os_mutex_lock(&wct->tracked_controller_lock);
	assert(wct->num_controllers <= CONSTELLATION_MAX_DEVICES);

	for (int d = 0; d < wct->num_controllers; d++) {
		struct wmr_controller_tracker_device *device = wct->controllers + d;

		if (!device->have_led_model) {
			if (!wmr_controller_tracker_connection_get_led_model(device->connection, &device->led_model)) {
				continue; // Can't do anything without the LED info
			}

			WMR_INFO(wct, "WMR Controller Tracker: Retrieved controller LED model for device %u",
			         device->led_model.id);
			device->search_led_model = constellation_search_model_new(&device->led_model);
			device->have_led_model = true;
		}

		struct tracking_sample_device_state *dev_state = sample->devices + sample->n_devices;

		// Collect the controller prior pose
		struct xrt_space_relation xsr;
		if (!wmr_controller_tracker_connection_get_tracked_pose(device->connection, xf->timestamp, &xsr)) {
			WMR_DEBUG(wct, "Failed to retrieve tracked pose for device %u", device->led_model.id);
			continue; // Can't retrieve the pose: means the device was disconnected
		}

		// Incoming controller pose is in OpenXR. Flip it to OpenCV for all our operations
		pose_flip_YZ(&xsr.pose, &dev_state->P_world_obj_prior);

		//! @todo: Get actual error bounds from fusion
		dev_state->prior_pos_error.x = dev_state->prior_pos_error.y = dev_state->prior_pos_error.z =
		    MIN_POS_ERROR;
		dev_state->prior_rot_error.x = dev_state->prior_rot_error.y = dev_state->prior_rot_error.z =
		    MIN_ROT_ERROR;
		dev_state->gravity_error_rad = MIN_ROT_ERROR;

		dev_state->have_last_seen_pose = device->have_last_seen_pose;
		dev_state->last_seen_pose = device->last_seen_pose;

		dev_state->dev_index = d;
		dev_state->led_model = &device->led_model;

		sample->n_devices++;
	}
	os_mutex_unlock(&wct->tracked_controller_lock);

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
		struct wmr_controller_tracker_device *device = wct->controllers + dev_state->dev_index;

		WMR_DEBUG(wct, "Doing fast match search for device %d", device->led_model.id);

		if (device_try_global_pose(wct, dev_state, sample, &dev_state->P_world_obj_prior)) {
			WMR_DEBUG(wct, "Found fast match search for device %d in view %d from prior pose",
			          device->led_model.id, dev_state->found_pose_view_id);
			continue;
		}
		if (dev_state->have_last_seen_pose &&
		    device_try_global_pose(wct, dev_state, sample, &dev_state->last_seen_pose)) {
			WMR_DEBUG(wct, "Found fast match search for device %d in view %d from last_seen pose",
			          device->led_model.id, dev_state->found_pose_view_id);
			continue;
		}
		if (device_try_recover_pose(wct, dev_state, sample)) {
			WMR_DEBUG(wct, "Found fast match search for device %d in view %d from labelled blobs",
			          device->led_model.id, dev_state->found_pose_view_id);
			continue;
		}
		if (!dev_state->found_device_pose) {
			need_full_search = true;
		}
	}

	uint64_t fast_analysis_finish_ts = os_monotonic_get_ns();
	wct->last_fast_analysis_ms = (fast_analysis_finish_ts - fast_analysis_start_ts) / U_TIME_1MS_IN_NS;

	/* Send analysis results to debug view if needed */
	enum debug_draw_flag debug_flags = DEBUG_DRAW_FLAG_NONE;
	if (wct->debug_draw_normalise)
		debug_flags |= DEBUG_DRAW_FLAG_NORMALISE;
	if (wct->debug_draw_blob_tint)
		debug_flags |= DEBUG_DRAW_FLAG_BLOB_TINT;
	if (wct->debug_draw_blob_circles)
		debug_flags |= DEBUG_DRAW_FLAG_BLOB_CIRCLE;
	if (wct->debug_draw_blob_ids)
		debug_flags |= DEBUG_DRAW_FLAG_BLOB_IDS;
	if (wct->debug_draw_blob_unique_ids)
		debug_flags |= DEBUG_DRAW_FLAG_BLOB_UNIQUE_IDS;
	if (wct->debug_draw_leds)
		debug_flags |= DEBUG_DRAW_FLAG_LEDS;
	if (wct->debug_draw_prior_leds)
		debug_flags |= DEBUG_DRAW_FLAG_PRIOR_LEDS;
	if (wct->debug_draw_last_leds)
		debug_flags |= DEBUG_DRAW_FLAG_LAST_SEEN_LEDS;
	if (wct->debug_draw_pose_bounds)
		debug_flags |= DEBUG_DRAW_FLAG_POSE_BOUNDS;

	for (int i = 0; i < sample->n_views; i++) {
		struct wmr_controller_tracker_camera *cam = wct->cam + i;
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
	if (wct->do_full_search || need_full_search) {
		wct->do_full_search = false;

		/* Send the sample for long analysis */
		os_thread_helper_lock(&wct->long_analysis_thread);
		if (wct->long_analysis_pending_sample != NULL) {
			constellation_tracking_sample_free(wct->long_analysis_pending_sample);
		}
		wct->long_analysis_pending_sample = sample;
		os_thread_helper_signal_locked(&wct->long_analysis_thread);
		os_thread_helper_unlock(&wct->long_analysis_thread);
	} else {
		/* not sending for long analysis: free it */
		constellation_tracking_sample_free(sample);
	}
}

static void
wmr_controller_tracker_process_frame_long(struct wmr_controller_tracker *wct,
                                          struct constellation_tracking_sample *sample)
{
	WMR_DEBUG(wct, "Starting long analysis of frame TS %" PRIu64, sample->timestamp);

	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		struct wmr_controller_tracker_camera *cam = wct->cam + view_id;

		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue; // no blobs in this view
		}

		correspondence_search_set_blobs(cam->cs, view->bwobs->blobs, view->bwobs->num_blobs);

		for (int d = 0; d < sample->n_devices; d++) {
			struct tracking_sample_device_state *dev_state = sample->devices + d;
			struct wmr_controller_tracker_device *device = wct->controllers + dev_state->dev_index;

			//! The fast analysis thread must have retrieved the LED model before
			//  ever sending a device for long analysis
			assert(device->have_led_model == true);

			if (dev_state->found_device_pose)
				continue; /* This device was already found. No need for a long search */

			WMR_DEBUG(wct, "Doing full search for device %d in view %d", device->led_model.id, view_id);

			for (int pass = 0; pass < 2; pass++) {

				if (dev_state->found_device_pose)
					break; /* This device was already found on the previous pass */

				enum correspondence_search_flags search_flags =
				    CS_FLAG_STOP_FOR_STRONG_MATCH | CS_FLAG_HAVE_POSE_PRIOR | CS_FLAG_MATCH_GRAVITY;

				if (pass == 0) {
					/* 1st pass - quick search only */
					search_flags |= CS_FLAG_SHALLOW_SEARCH;
				} else {
					/* 2nd pass - do a deep search */
					search_flags |= CS_FLAG_DEEP_SEARCH;
				}

				struct xrt_pose P_cam_obj;
				math_pose_transform(&view->P_cam_world, &dev_state->P_world_obj_prior, &P_cam_obj);

				if (correspondence_search_find_one_pose(
				        cam->cs, device->search_led_model, search_flags, &P_cam_obj,
				        &dev_state->prior_pos_error, &dev_state->prior_rot_error,
				        &view->cam_gravity_vector, dev_state->gravity_error_rad, &dev_state->score)) {
					WMR_DEBUG(wct, "Found a pose on cam %u device %d long search pass %d", view_id,
					          device->led_model.id, pass);
					submit_device_pose(wct, dev_state, sample, view_id, &P_cam_obj);
					break;
				}
			}
		}
	}
}

static void *
wmr_controller_tracking_long_analysis_thread(void *ptr)
{
	U_TRACE_SET_THREAD_NAME("WMR: Controller long recovery thread");
	struct wmr_controller_tracker *wct = (struct wmr_controller_tracker *)(ptr);

	os_thread_helper_lock(&wct->long_analysis_thread);
	while (os_thread_helper_is_running_locked(&wct->long_analysis_thread)) {
		/* Wait for a sample to analyse, or for shutdown */
		if (wct->long_analysis_pending_sample == NULL) {
			os_thread_helper_wait_locked(&wct->long_analysis_thread);
			if (!os_thread_helper_is_running_locked(&wct->long_analysis_thread)) {
				break;
			}
		}

		/* Take ownership of any pending sample */
		struct constellation_tracking_sample *sample = wct->long_analysis_pending_sample;
		wct->long_analysis_pending_sample = NULL;

		os_thread_helper_unlock(&wct->long_analysis_thread);
		if (sample != NULL) {
			uint64_t long_analysis_start_ts = os_monotonic_get_ns();
			wmr_controller_tracker_process_frame_long(wct, sample);
			uint64_t long_analysis_finish_ts = os_monotonic_get_ns();

			constellation_tracking_sample_free(sample);

			wct->last_long_analysis_ms =
			    (long_analysis_finish_ts - long_analysis_start_ts) / U_TIME_1MS_IN_NS;
		}

		os_thread_helper_lock(&wct->long_analysis_thread);
	}
	os_thread_helper_unlock(&wct->long_analysis_thread);

	return NULL;
}

static void
wmr_controller_tracker_node_destroy(struct xrt_frame_node *node)
{
	struct wmr_controller_tracker *wct = container_of(node, struct wmr_controller_tracker, node);

	DRV_TRACE_MARKER();
	WMR_DEBUG(wct, "Destroying WMR controller tracker");

	/* Make sure the long analysis thread isn't running */
	os_thread_helper_stop_and_wait(&wct->long_analysis_thread);

	// Unlink the controller connections and release the models
	os_mutex_lock(&wct->tracked_controller_lock);
	for (int i = 0; i < wct->num_controllers; i++) {
		struct wmr_controller_tracker_device *device = wct->controllers + i;

		// Clean up the LED tracking model
		constellation_led_model_clear(&device->led_model);
		if (device->search_led_model) {
			constellation_search_model_free(device->search_led_model);
		}

		if (device->connection != NULL) {
			wmr_controller_tracker_connection_disconnect(device->connection);
			device->connection = NULL;
		}
	}
	os_mutex_unlock(&wct->tracked_controller_lock);
	os_mutex_destroy(&wct->tracked_controller_lock);

	/* Clean up any pending sample */
	os_thread_helper_lock(&wct->long_analysis_thread);
	if (wct->long_analysis_pending_sample != NULL) {
		constellation_tracking_sample_free(wct->long_analysis_pending_sample);
		wct->long_analysis_pending_sample = NULL;
	}
	os_thread_helper_unlock(&wct->long_analysis_thread);
	/* Then release the thread helper */
	os_thread_helper_destroy(&wct->long_analysis_thread);

	//! Clean up
	for (int i = 0; i < wct->cam_count; i++) {
		struct wmr_controller_tracker_camera *cam = wct->cam + i;

		u_sink_debug_destroy(&cam->debug_sink);

		if (cam->cs) {
			correspondence_search_free(cam->cs);
		}

		os_mutex_destroy(&cam->bw_lock);
		if (cam->bw) {
			blobwatch_free(cam->bw);
		}
	}

	u_var_remove_root(wct);
	free(wct);
}

static void
wct_full_search_btn_cb(void *wct_ptr)
{
	struct wmr_controller_tracker *wct = (struct wmr_controller_tracker *)wct_ptr;
	wct->do_full_search = true;
}

int
wmr_controller_tracker_create(struct xrt_frame_context *xfctx,
                              struct xrt_device *hmd_xdev,
                              struct wmr_hmd_config *hmd_cfg,
                              struct t_slam_calibration *slam_calib,
                              struct wmr_controller_tracker **out_tracker,
                              struct xrt_frame_sink **out_sink)
{
	DRV_TRACE_MARKER();

	int ret;
	struct wmr_controller_tracker *wct = calloc(1, sizeof(struct wmr_controller_tracker));

	wct->log_level = debug_get_log_option_wmr_log();
	wct->debug_draw_blob_tint = true;
	wct->debug_draw_blob_ids = true;
	wct->hmd_xdev = hmd_xdev;

	// Set up the per-camera constellation tracking pieces config and pose
	wct->cam_count = slam_calib->cam_count;

	struct xrt_pose P_imu_c0 = hmd_cfg->sensors.accel.pose;

	for (int i = 0; i < wct->cam_count; i++) {
		struct wmr_controller_tracker_camera *cam = wct->cam + i;
		struct wmr_camera_config *cam_cfg = hmd_cfg->tcams[i];
		struct t_slam_camera_calibration *cam_slam_cfg = slam_calib->cams + i;

#if 1
		if (i == 2 || i == 3) {
			//! @note The calibration json for the reverb G2v2 (the only 4-camera wmr
			//! headset we know about) has the HT2 and HT3 extrinsics flipped compared
			//! to the order the third and fourth camera images come from usb.
			cam_cfg = hmd_cfg->tcams[i == 2 ? 3 : 2];
		}
#endif

		cam->roi = cam_cfg->roi;

		struct xrt_pose P_ci_c0 = cam_cfg->pose;
		struct xrt_pose P_c0_ci;
		math_pose_invert(&P_ci_c0, &P_c0_ci);

		math_pose_transform(&P_imu_c0, &P_c0_ci, &cam->P_imu_cam);

		/* Init the camera model with size and distortion */
		cam->camera_model.width = cam_cfg->roi.extent.w;
		cam->camera_model.height = cam_cfg->roi.extent.h;
		t_camera_model_params_from_t_camera_calibration(&cam_slam_cfg->base, &cam->camera_model.calib);

		os_mutex_init(&cam->bw_lock);
		cam->bw = blobwatch_new(BLOB_PIXEL_THRESHOLD_WMR, BLOB_THRESHOLD_MIN_WMR);
		cam->cs = correspondence_search_new(&cam->camera_model);
	}

	// Set up frame receiver
	wct->base.push_frame = wmr_controller_tracker_receive_frame;

	// Setup node
	struct xrt_frame_node *xfn = &wct->node;
	xfn->break_apart = wmr_controller_tracker_node_break_apart;
	xfn->destroy = wmr_controller_tracker_node_destroy;

	ret = os_mutex_init(&wct->tracked_controller_lock);
	if (ret != 0) {
		WMR_ERROR(wct, "Failed to init Tracked Controller mutex!");
		wmr_controller_tracker_node_destroy(&wct->node);
		return -1;
	}

	ret = os_thread_helper_init(&wct->long_analysis_thread);
	if (ret != 0) {
		WMR_ERROR(wct, "WMR Controller tracking: Failed to init long analysis thread!");
		wmr_controller_tracker_node_destroy(&wct->node);
		return -1;
	}

	// Fast processing thread
	wct->fast_process_sink.push_frame = wmr_controller_tracker_process_frame_fast;

	if (!u_sink_queue_create(xfctx, MAX_FAST_QUEUE_SIZE, &wct->fast_process_sink, &wct->fast_q_sink)) {
		WMR_ERROR(wct, "Failed to init Tracked Controller mutex!");
		wmr_controller_tracker_node_destroy(&wct->node);
		return -1;
	}

	// Long match recovery thread
	ret = os_thread_helper_start(&wct->long_analysis_thread, wmr_controller_tracking_long_analysis_thread, wct);
	if (ret != 0) {
		WMR_ERROR(wct, "WMR Controller tracking: Failed to start long analysis thread!");
		wmr_controller_tracker_node_destroy(&wct->node);
		return -1;
	}

	// Debug UI
	wct->full_search_button.cb = wct_full_search_btn_cb;
	wct->full_search_button.ptr = wct;

	u_var_add_root(wct, "WMR Controller Tracker", false);
	u_var_add_log_level(wct, &wct->log_level, "Log Level");
	u_var_add_ro_i32(wct, &wct->num_controllers, "Num Controllers");
	u_var_add_ro_u64(wct, &wct->last_frame_timestamp, "Last Frame Timestamp");
	u_var_add_ro_u64(wct, &wct->last_blob_analysis_ms, "Blob tracking time (ms)");
	u_var_add_ro_u64(wct, &wct->last_fast_analysis_ms, "Fast analysis time (ms)");
	u_var_add_ro_u64(wct, &wct->last_long_analysis_ms, "Long analysis time (ms)");
	u_var_add_button(wct, &wct->full_search_button, "Trigger ab-initio search");
	u_var_add_bool(wct, &wct->debug_draw_normalise, "Debug: Normalise source frame");
	u_var_add_bool(wct, &wct->debug_draw_blob_tint, "Debug: Tint blobs by device assignment");
	u_var_add_bool(wct, &wct->debug_draw_blob_circles, "Debug: Draw circles around blobs");
	u_var_add_bool(wct, &wct->debug_draw_blob_ids, "Debug: Draw LED id labels for blobs");
	u_var_add_bool(wct, &wct->debug_draw_blob_unique_ids, "Debug: Draw blobs tracking ID");
	u_var_add_bool(wct, &wct->debug_draw_leds, "Debug: Draw LED position markers for found poses");
	u_var_add_bool(wct, &wct->debug_draw_prior_leds, "Debug: Draw LED markers for prior poses");
	u_var_add_bool(wct, &wct->debug_draw_last_leds, "Debug: Draw LED markers for last observed poses");
	u_var_add_bool(wct, &wct->debug_draw_pose_bounds, "Debug: Draw bounds rect for found poses");

	for (int i = 0; i < wct->cam_count; i++) {
		struct wmr_controller_tracker_camera *cam = wct->cam + i;
		u_var_add_ro_i32(wct, &cam->last_num_blobs, "Num Blobs");
		u_var_add_pose(wct, &cam->debug_last_pose, "Last view pose");
		u_var_add_vec3_f32(wct, &cam->debug_last_gravity_vector, "Last gravity vector");

		char cam_name[64];
		sprintf(cam_name, "Cam %u", i);
		u_sink_debug_init(&cam->debug_sink);
		u_var_add_sink_debug(wct, &cam->debug_sink, cam_name);
	}

	// Hand ownership to the frame context
	xrt_frame_context_add(xfctx, &wct->node);

	WMR_DEBUG(wct, "WMR Controller tracker created");

	*out_tracker = wct;
	*out_sink = &wct->base;

	return 0;
}

static void
wmr_controller_tracker_connection_destroy(struct wmr_controller_tracker_connection *wctc)
{
	DRV_TRACE_MARKER();

	os_mutex_destroy(&wctc->lock);
	free(wctc);
}

static struct wmr_controller_tracker_connection *
wmr_controller_tracker_connection_create(int id,
                                         struct wmr_controller_tracker *tracker,
                                         struct wmr_controller_base *wcb)
{
	DRV_TRACE_MARKER();

	struct wmr_controller_tracker_connection *wctc = calloc(1, sizeof(struct wmr_controller_tracker_connection));

	wctc->id = id;
	wctc->wcb = wcb;
	wctc->tracker = tracker;

	/* Init 2 references - one for the controller, one for the tracker */
	xrt_reference_inc(&wctc->ref);
	xrt_reference_inc(&wctc->ref);

	int ret = os_mutex_init(&wctc->lock);
	if (ret != 0) {
		WMR_ERROR(tracker, "WMR Controller Tracker connection: Failed to init mutex!");
		wmr_controller_tracker_connection_destroy(wctc);
		return NULL;
	}

	return wctc;
}

struct wmr_controller_tracker_connection *
wmr_controller_tracker_add_controller(struct wmr_controller_tracker *wct, struct wmr_controller_base *wcb)
{
	os_mutex_lock(&wct->tracked_controller_lock);
	assert(wct->num_controllers < WMR_MAX_CONTROLLERS);

	WMR_DEBUG(wct, "WMR Controller Tracker: Adding controller %d", wct->num_controllers);

	struct wmr_controller_tracker_connection *wctc =
	    wmr_controller_tracker_connection_create(wct->num_controllers, wct, wcb);
	if (wctc != NULL) {
		struct wmr_controller_tracker_device *device = wct->controllers + wct->num_controllers;
		device->connection = wctc;
		device->last_matched_cam = -1;
		wct->num_controllers++;

		char dev_name[64];
		sprintf(dev_name, "Controller %u - %s", wct->num_controllers,
		        wcb->base.device_type == XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER ? "Left" : "Right");
		u_var_add_ro_text(wct, "Device", dev_name);
		u_var_add_pose(wct, &device->last_seen_pose, "Last observed global pose");
		u_var_add_u64(wct, &device->last_seen_pose_ts, "Last observed pose");
		u_var_add_ro_i32(wct, &device->last_matched_blobs, "Last matched Blobs");
		u_var_add_ro_i32(wct, &device->last_matched_cam, "Last observed camera #");
		u_var_add_pose(wct, &device->last_matched_cam_pose, "Last observed camera pose");
	}

	os_mutex_unlock(&wct->tracked_controller_lock);
	return wctc;
}

void
wmr_controller_tracker_connection_disconnect(struct wmr_controller_tracker_connection *wctc)
{
	os_mutex_lock(&wctc->lock);
	wctc->disconnected = true;
	os_mutex_unlock(&wctc->lock);

	if (xrt_reference_dec_and_is_zero(&wctc->ref)) {
		wmr_controller_tracker_connection_destroy(wctc);
	}
}
