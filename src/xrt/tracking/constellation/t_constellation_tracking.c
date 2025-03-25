// Copyright 2023 Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Implementation of LED constellation tracking
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup constellation
 */
#include <inttypes.h>
#include <stdio.h>

#include "os/os_threading.h"

#include "tracking/t_led_models.h"
#include "tracking/t_constellation_tracking.h"

#include "util/u_debug.h"
#include "util/u_frame.h"
#include "util/u_logging.h"
#include "util/u_sink.h"
#include "util/u_trace_marker.h"
#include "util/u_var.h"

#include "internal/blobwatch.h"
#include "internal/camera_model.h"
#include "internal/correspondence_search.h"
#include "internal/debug_draw.h"
#include "internal/ransac_pnp.h"
#include "internal/sample.h"

#define MAX_TRACKED_DEVICES 4

DEBUG_GET_ONCE_LOG_OPTION(ct_log, "CONSTELLATION_LOG", U_LOGGING_INFO)

#define MIN_ROT_ERROR DEG_TO_RAD(30)
#define MIN_POS_ERROR 0.10

#define CT_TRACE(c, ...) U_LOG_IFL_T(c->log_level, __VA_ARGS__)
#define CT_DEBUG(c, ...) U_LOG_IFL_D(c->log_level, __VA_ARGS__)
#define CT_INFO(c, ...) U_LOG_IFL_I(c->log_level, __VA_ARGS__)
#define CT_WARN(c, ...) U_LOG_IFL_W(c->log_level, __VA_ARGS__)
#define CT_ERROR(c, ...) U_LOG_IFL_E(c->log_level, __VA_ARGS__)

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
	struct constellation_tracker_device devices[MAX_TRACKED_DEVICES];

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

static void
submit_device_pose(struct t_constellation_tracker *ct,
                   struct tracking_sample_device_state *dev_state,
                   struct constellation_tracking_sample *sample,
                   int view_id,
                   struct xrt_pose *P_cam_obj)
{
	struct constellation_tracker_camera_state *cam = ct->cam + view_id;
	struct tracking_sample_frame *view = sample->views + view_id;
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;

	struct pose_metrics *score = &dev_state->score;

	pose_metrics_match_pose_to_blobs(P_cam_obj, view->bwobs->blobs, view->bwobs->num_blobs, &device->led_model,
	                                 &cam->camera_model, &dev_state->blob_match_info);
	mark_matching_blobs(ct, P_cam_obj, view->bwobs, &device->led_model, &dev_state->blob_match_info);

	struct xrt_pose refine_pose = *P_cam_obj;

	int num_leds_out;
	int num_inliers;

	if (!ransac_pnp_pose(&refine_pose, view->bwobs->blobs, view->bwobs->num_blobs, &device->led_model,
	                     &cam->camera_model, &num_leds_out, &num_inliers)) {
		CT_DEBUG(ct, "Camera %d RANSAC-PnP refinement for device %d from %u blobs failed", view_id,
		         device->led_model.id, view->bwobs->num_blobs);
	} else {
		CT_DEBUG(ct,
		         "Camera %d RANSAC-PnP refinement for device %d from %u blobs had %d LEDs with %d inliers. "
		         "Produced pose %f,%f,%f,%f pos %f,%f,%f",
		         view_id, device->led_model.id, view->bwobs->num_blobs, num_leds_out, num_inliers,
		         refine_pose.orientation.x, refine_pose.orientation.y, refine_pose.orientation.z,
		         refine_pose.orientation.w, refine_pose.position.x, refine_pose.position.y,
		         refine_pose.position.z);
		// *P_cam_obj = refine_pose;
	}

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

		constellation_tracked_device_connection_notify_pose(device->connection, sample->timestamp,
		                                                    &P_xrworld_device);
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
			submit_device_pose(ct, dev_state, sample, view_id, &P_cam_obj_candidate);
			ret = true;
		}
	}

	return ret;
}

/* Try and recover the pose from labelled blobs */
static bool
device_try_recover_pose(struct t_constellation_tracker *ct,
                        struct tracking_sample_device_state *dev_state,
                        struct constellation_tracking_sample *sample)
{
	struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
	struct t_constellation_led_model *leds_model = &device->led_model;

	for (int view_id = 0; view_id < sample->n_views; view_id++) {
		struct tracking_sample_frame *view = sample->views + view_id;
		struct constellation_tracker_camera_state *cam = ct->cam + view_id;

		if (view->bwobs == NULL || view->bwobs->num_blobs == 0) {
			continue;
		}

		struct constellation_tracker_device *device = ct->devices + dev_state->dev_index;
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

		CT_DEBUG(ct,
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
			CT_DEBUG(ct, "Camera %d RANSAC-PnP for device %d from %u blobs failed", view_id, leds_model->id,
			         num_blobs);
			continue;
		}

		pose_metrics_evaluate_pose_with_prior(&dev_state->score, &P_cam_obj, true, &P_cam_obj_prior,
		                                      &dev_state->prior_pos_error, &dev_state->prior_rot_error,
		                                      bwobs->blobs, bwobs->num_blobs, &device->led_model,
		                                      &cam->camera_model, NULL);

		if (POSE_HAS_FLAGS(&dev_state->score, POSE_MATCH_GOOD)) {
			CT_DEBUG(ct, "Camera %d RANSAC-PnP recovered pose for device %d from %u blobs", view_id,
			         leds_model->id, num_blobs);
			submit_device_pose(ct, dev_state, sample, view_id, &P_cam_obj);
			return true;
		}
		CT_DEBUG(ct,
		         "Camera %d device %d had %d prior blobs, but failed match with flags 0x%x. "
		         "Yielded pose %f %f %f %f pos %f %f %f (match %d of %d visible) rot_error %f %f %f pos_error "
		         "%f %f %f",
		         view_id, leds_model->id, num_blobs, dev_state->score.match_flags, P_cam_obj.orientation.x,
		         P_cam_obj.orientation.y, P_cam_obj.orientation.z, P_cam_obj.orientation.w,
		         P_cam_obj.position.x, P_cam_obj.position.y, P_cam_obj.position.z,
		         dev_state->score.matched_blobs, dev_state->score.visible_leds, dev_state->score.orient_error.x,
		         dev_state->score.orient_error.y, dev_state->score.orient_error.z, dev_state->score.pos_error.x,
		         dev_state->score.pos_error.y, dev_state->score.pos_error.z);
	}

	return false;
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

		os_mutex_lock(&cam->bw_lock);
		blobwatch_process(cam->bw, view->vframe, &view->bwobs);
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

		// Collect the controller prior pose
		struct xrt_space_relation xsr;
		if (!constellation_tracked_device_connection_get_tracked_pose(device->connection, xf->timestamp,
		                                                              &xsr)) {
			CT_DEBUG(ct, "Failed to retrieve tracked pose for device %u", device->led_model.id);
			continue; // Can't retrieve the pose: means the device was disconnected
		}

		// Apply device -> LED model pose from xsr = P_world_device + P_device_model = P_world_model
		struct xrt_pose P_xrworld_model;
		math_pose_transform(&xsr.pose, &device->led_model.P_device_model, &P_xrworld_model);

		// Incoming controller pose is in OpenXR. Flip it to OpenCV for all our operations
		pose_flip_YZ(&P_xrworld_model, &dev_state->P_world_obj_prior);

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
			constellation_tracking_sample_free(ct->long_analysis_pending_sample);
		}
		ct->long_analysis_pending_sample = sample;
		os_thread_helper_signal_locked(&ct->long_analysis_thread);
		os_thread_helper_unlock(&ct->long_analysis_thread);
	} else {
		/* not sending for long analysis: free it */
		constellation_tracking_sample_free(sample);
	}
}

static void
constellation_tracker_process_frame_long(struct t_constellation_tracker *ct,
                                         struct constellation_tracking_sample *sample)
{
	CT_DEBUG(ct, "Starting long analysis of frame TS %" PRIu64, sample->timestamp);

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
					CT_DEBUG(ct, "Found a pose on cam %u device %d long search pass %d", view_id,
					         device->led_model.id, pass);
					submit_device_pose(ct, dev_state, sample, view_id, &P_cam_obj);
					break;
				}
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
                               struct xrt_frame_sink **out_sink)
{
	DRV_TRACE_MARKER();

	int ret;
	struct t_constellation_tracker *ct = calloc(1, sizeof(struct t_constellation_tracker));

	ct->log_level = debug_get_log_option_ct_log();
	ct->debug_draw_blob_tint = true;
	ct->debug_draw_blob_ids = true;
	ct->hmd_xdev = hmd_xdev;

	// Set up the per-camera constellation tracking pieces config and pose
	ct->cam_count = cams->cam_count;
	for (int i = 0; i < ct->cam_count; i++) {
		struct constellation_tracker_camera_state *cam = ct->cam + i;
		struct t_constellation_camera *cam_cfg = cams->cams + i;

		cam->roi = cam_cfg->roi;
		cam->P_imu_cam = cam_cfg->P_imu_cam;

		/* Init the camera model with size and distortion */
		cam->camera_model.width = cam_cfg->roi.extent.w;
		cam->camera_model.height = cam_cfg->roi.extent.h;
		t_camera_model_params_from_t_camera_calibration(&cam_cfg->calibration, &cam->camera_model.calib);

		os_mutex_init(&cam->bw_lock);
		cam->bw = blobwatch_new(cam_cfg->blob_min_threshold, cam_cfg->blob_detect_threshold);
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
	u_var_add_bool(ct, &ct->debug_draw_pose_bounds, "Debug: Draw bounds rect for found poses");

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
	assert(ct->num_devices < MAX_TRACKED_DEVICES);

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
