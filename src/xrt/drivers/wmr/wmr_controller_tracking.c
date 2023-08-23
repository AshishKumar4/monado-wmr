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
#include "util/u_trace_marker.h"

#include "constellation/blobwatch.h"
#include "constellation/camera_model.h"
#include "constellation/correspondence_search.h"
#include "constellation/debug_draw.h"
#include "constellation/led_models.h"

#include "wmr_controller_tracking.h"
#include "wmr_hmd.h"

DEBUG_GET_ONCE_LOG_OPTION(wmr_log, "WMR_LOG", U_LOGGING_INFO)

#define RAD_TO_DEG(RAD) ((RAD)*180. / M_PI)

#define WMR_TRACE(c, ...) U_LOG_IFL_T(c->log_level, __VA_ARGS__)
#define WMR_DEBUG(c, ...) U_LOG_IFL_D(c->log_level, __VA_ARGS__)
#define WMR_INFO(c, ...) U_LOG_IFL_I(c->log_level, __VA_ARGS__)
#define WMR_WARN(c, ...) U_LOG_IFL_W(c->log_level, __VA_ARGS__)
#define WMR_ERROR(c, ...) U_LOG_IFL_E(c->log_level, __VA_ARGS__)

/* WMR thresholds for min brightness and min-blob-required magnitude */
#define BLOB_PIXEL_THRESHOLD_WMR 0x8
#define BLOB_THRESHOLD_MIN_WMR 0x10

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
};

struct wmr_controller_tracker_camera
{
	struct wmr_camera_config wcfg;
	struct camera_model camera_model;

	//! Constellation tracking - fast tracking thread
	blobwatch *bw;
	int last_num_blobs;

	//! Full search / pose recovery thread
	struct correspondence_search *cs;

	//! Debug output
	struct u_sink_debug debug_sink;
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
	struct xrt_device *reference_xdev;

	struct os_mutex tracked_controller_lock;

	//! Controller communication connections
	int num_controllers;
	struct wmr_controller_tracker_device controllers[WMR_MAX_CONTROLLERS];

	//!< Tracking camera entries
	struct wmr_controller_tracker_camera cam[WMR_MAX_CAMERAS];
	int cam_count;

	/* Debug */
	enum u_logging_level log_level;

	uint64_t last_timestamp;
	uint64_t last_sequence;

	struct u_var_button full_search_button;
	bool do_full_search;

	// Fast recovery thread
	struct xrt_frame_sink *fast_q_sink;
	struct xrt_frame_sink fast_process_sink;
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
		xrt_device_get_tracked_pose(xdev, XRT_INPUT_GENERIC_STAGE_SPACE_POSE, timestamp_ns, xsr);
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

	/* The frame cadence is SLAM/controller/controller, and we need to pass the
	 * estimate of the next SLAM frame timestamp to the timesync */
	bool is_second_frame = wct->last_sequence + 1 == xf->source_sequence;
	wct->last_timestamp = xf->timestamp;
	wct->last_sequence = xf->source_sequence;

	os_mutex_lock(&wct->tracked_controller_lock);

	// Update the controller timesync estimate
	if (is_second_frame) {
		timepoint_ns next_slam_ts = (timepoint_ns)(xf->timestamp) + U_TIME_1S_IN_NS / 90;
		for (int i = 0; i < wct->num_controllers; i++) {
			wmr_controller_tracker_connection_notify_timesync(wct->controllers[i].connection, next_slam_ts);
		}
	}

	os_mutex_unlock(&wct->tracked_controller_lock);

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
                    struct camera_model *calib)
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

	struct pose_metrics_blob_match_info blob_match_info;
	pose_metrics_match_pose_to_blobs(pose, bwobs->blobs, bwobs->num_blobs, led_model, calib, &blob_match_info);

	/* Iterate the visible LEDs and mark matching blobs with this device ID and LED ID */
	for (i = 0; i < blob_match_info.num_visible_leds; i++) {
		struct pose_metrics_visible_led_info *led_info = blob_match_info.visible_leds + i;
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

// Fast frame processing: blob extraction and match to existing predictions
static void
wmr_controller_tracker_process_frame_fast(struct xrt_frame_sink *sink, struct xrt_frame *xf)
{
	struct wmr_controller_tracker *wct = container_of(sink, struct wmr_controller_tracker, fast_process_sink);

	struct xrt_frame *frames[WMR_MAX_CAMERAS] = {NULL};
	blobservation *frames_bwobs[WMR_MAX_CAMERAS] = {NULL};

	/* Split out camera views and collect blobs across all cameras */
	for (int i = 0; i < wct->cam_count; i++) {
		struct wmr_controller_tracker_camera *cam = wct->cam + i;
		blobservation *bwobs = NULL;

		u_frame_create_roi(xf, cam->wcfg.roi, &frames[i]);

		blobwatch_process(cam->bw, frames[i], &bwobs);
		frames_bwobs[i] = bwobs;

		if (bwobs == NULL) {
			cam->last_num_blobs = 0;
			continue;
		}

		cam->last_num_blobs = bwobs->num_blobs;

		printf("frame %" PRIu64 " TS %" PRIu64 " cam %d ROI %d,%d w/h %d,%d Blobs: %d\n", xf->source_sequence,
		       xf->timestamp, i, cam->wcfg.roi.offset.w, cam->wcfg.roi.offset.h, cam->wcfg.roi.extent.w,
		       cam->wcfg.roi.extent.h, bwobs->num_blobs);

		for (int index = 0; index < bwobs->num_blobs; index++) {
			printf("  Blob[%d]: %f,%f %dx%d id %d age %u\n", index, bwobs->blobs[index].x,
			       bwobs->blobs[index].y, bwobs->blobs[index].width, bwobs->blobs[index].height,
			       bwobs->blobs[index].led_id, bwobs->blobs[index].age);
		}
	}

	/* Grab the 'do_full_search' value in case the user clicked the
	 * button in the debug UI */
	bool do_full_search = wct->do_full_search;
	wct->do_full_search = false;

	// Ready to start processing device poses now. Make sure we have the
	// LED models and collect the best estimate of the current pose
	// for each target device
	os_mutex_lock(&wct->tracked_controller_lock);
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

		// Collect the controller prior pose
		struct xrt_space_relation xsr;
		if (!wmr_controller_tracker_connection_get_tracked_pose(device->connection, xf->timestamp, &xsr)) {
			continue; // Can't retrieve the pose: means the device was disconnected
		}

		if (do_full_search) {
			WMR_INFO(wct, "Doing full search for device %d", device->led_model.id);

			for (int c = 0; c < wct->cam_count; c++) {
				struct wmr_controller_tracker_camera *cam = wct->cam + c;

				if (frames_bwobs[c] == NULL)
					continue; // no blobs in this view

				correspondence_search_set_blobs(cam->cs, frames_bwobs[c]->blobs,
				                                frames_bwobs[c]->num_blobs);

				enum correspondence_search_flags search_flags =
				    CS_FLAG_STOP_FOR_STRONG_MATCH | CS_FLAG_MATCH_ALL_BLOBS | CS_FLAG_DEEP_SEARCH;
				struct pose_metrics score;
				struct xrt_pose obj_cam_pose;

				if (correspondence_search_find_one_pose(cam->cs, device->search_led_model, search_flags,
				                                        &obj_cam_pose, NULL, NULL, NULL, 0.0, &score)) {
					WMR_INFO(wct,
					         "Found a pose on cam %d device %d score match_flags 0x%x matched %u "
					         "blobs of %u visible LEDs",
					         c, device->led_model.id, score.match_flags, score.matched_blobs,
					         score.visible_leds);

					mark_matching_blobs(wct, &obj_cam_pose, frames_bwobs[c], &device->led_model,
					                    &cam->camera_model);

					blobwatch_update_labels(cam->bw, frames_bwobs[c], device->led_model.id);
				}
			}
		}
	}
	os_mutex_unlock(&wct->tracked_controller_lock);

	/* Release camera frames and blobs */
	for (int i = 0; i < wct->cam_count; i++) {
		struct wmr_controller_tracker_camera *cam = wct->cam + i;

		if (u_sink_debug_is_active(&cam->debug_sink)) {
			struct xrt_frame *xf_src = frames[i];
			struct xrt_frame *xf_dbg = NULL;

			u_frame_create_one_off(XRT_FORMAT_R8G8B8, xf_src->width, xf_src->height, &xf_dbg);
			xf_dbg->timestamp = xf_src->timestamp;

			if (frames_bwobs[i] != NULL) {
				debug_draw_blobs(xf_dbg, frames_bwobs[i], xf_src);
			}

			u_sink_debug_push_frame(&cam->debug_sink, xf_dbg);
			xrt_frame_reference(&xf_dbg, NULL);
		}

		xrt_frame_reference(&frames[i], NULL);

		if (frames_bwobs[i] != NULL) {
			blobwatch_release_observation(cam->bw, frames_bwobs[i]);
		}
	}
}

static void
wmr_controller_tracker_node_destroy(struct xrt_frame_node *node)
{
	struct wmr_controller_tracker *wct = container_of(node, struct wmr_controller_tracker, node);

	DRV_TRACE_MARKER();
	WMR_DEBUG(wct, "Destroying WMR controller tracker");

	// Unlink the controller connections first
	os_mutex_lock(&wct->tracked_controller_lock);
	for (int i = 0; i < wct->num_controllers; i++) {
		struct wmr_controller_tracker_device *device = wct->controllers + i;

		if (device->connection != NULL) {
			// Clean up the LED tracking model
			constellation_led_model_clear(&device->led_model);
			if (device->search_led_model) {
				constellation_search_model_free(device->search_led_model);
			}

			wmr_controller_tracker_connection_disconnect(device->connection);
			device->connection = NULL;
		}
	}
	os_mutex_unlock(&wct->tracked_controller_lock);
	os_mutex_destroy(&wct->tracked_controller_lock);

	//! Clean up
	for (int i = 0; i < wct->cam_count; i++) {
		struct wmr_controller_tracker_camera *cam = wct->cam + i;

		u_sink_debug_destroy(&cam->debug_sink);

		if (cam->cs) {
			correspondence_search_free(cam->cs);
		}

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

static void
camera_model_from_wmr(struct camera_model *cm, struct wmr_camera_config *cfg)
{
	struct t_camera_calibration cc;
	struct wmr_distortion_6KT *intr = &cfg->distortion6KT;

	cc.image_size_pixels.h = cfg->roi.extent.h;
	cc.image_size_pixels.w = cfg->roi.extent.w;
	cc.intrinsics[0][0] = intr->params.fx * (double)cfg->roi.extent.w;
	cc.intrinsics[1][1] = intr->params.fy * (double)cfg->roi.extent.h;
	cc.intrinsics[0][2] = intr->params.cx * (double)cfg->roi.extent.w;
	cc.intrinsics[1][2] = intr->params.cy * (double)cfg->roi.extent.h;
	cc.intrinsics[2][2] = 1.0;

	cc.distortion_model = T_DISTORTION_WMR;
	cc.wmr.k1 = intr->params.k[0];
	cc.wmr.k2 = intr->params.k[1];
	cc.wmr.p1 = intr->params.p1;
	cc.wmr.p2 = intr->params.p2;
	cc.wmr.k3 = intr->params.k[2];
	cc.wmr.k4 = intr->params.k[3];
	cc.wmr.k5 = intr->params.k[4];
	cc.wmr.k6 = intr->params.k[5];
	cc.wmr.codx = intr->params.dist_x;
	cc.wmr.cody = intr->params.dist_y;
	cc.wmr.rpmax = intr->params.metric_radius;

	cm->width = cfg->roi.extent.w;
	cm->height = cfg->roi.extent.h;
	t_camera_model_params_from_t_camera_calibration(&cc, &cm->calib);
}

int
wmr_controller_tracker_create(struct xrt_frame_context *xfctx,
                              struct xrt_device *reference_xdev,
                              struct wmr_hmd_config *hmd_cfg,
                              struct wmr_controller_tracker **out_tracker,
                              struct xrt_frame_sink **out_sink)
{
	int ret;
	struct wmr_controller_tracker *wct = calloc(1, sizeof(struct wmr_controller_tracker));

	wct->log_level = debug_get_log_option_wmr_log();

	DRV_TRACE_MARKER();

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

	// Set up the per-camera constellation tracking pieces
	wct->cam_count = hmd_cfg->tcam_count;

	for (int i = 0; i < hmd_cfg->tcam_count; i++) {
		struct wmr_controller_tracker_camera *cam = wct->cam + i;
		cam->wcfg = *hmd_cfg->tcams[i];
		camera_model_from_wmr(&cam->camera_model, &cam->wcfg);
		cam->bw = blobwatch_new(BLOB_PIXEL_THRESHOLD_WMR, BLOB_THRESHOLD_MIN_WMR);
		cam->cs = correspondence_search_new(&cam->camera_model);
	}

	// Fast processing thread
	wct->fast_process_sink.push_frame = wmr_controller_tracker_process_frame_fast;

	if (!u_sink_simple_queue_create(xfctx, &wct->fast_process_sink, &wct->fast_q_sink)) {
		WMR_ERROR(wct, "Failed to init Tracked Controller mutex!");
		wmr_controller_tracker_node_destroy(&wct->node);
		return -1;
	}

	// Debug UI

	wct->full_search_button.cb = wct_full_search_btn_cb;
	wct->full_search_button.ptr = wct;

	u_var_add_root(wct, "WMR Controller Tracker", false);
	u_var_add_log_level(wct, &wct->log_level, "Log Level");
	u_var_add_ro_i32(wct, &wct->num_controllers, "Num Controllers");
	u_var_add_ro_u64(wct, &wct->last_timestamp, "Last Frame Timestamp");
	u_var_add_button(wct, &wct->full_search_button, "Trigger ab-initio search");

	for (int i = 0; i < wct->cam_count; i++) {
		struct wmr_controller_tracker_camera *cam = wct->cam + i;
		u_var_add_ro_i32(wct, &cam->last_num_blobs, "Num Blobs");

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
		wct->num_controllers++;
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
