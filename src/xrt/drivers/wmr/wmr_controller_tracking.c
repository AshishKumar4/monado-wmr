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
#include "constellation/sample.h"

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

/* Maximum number of frames to permit waiting in the fast-processing queue */
#define MAX_FAST_QUEUE_SIZE 2

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
	//! Distortion params
	struct camera_model camera_model;
	//! ROI in the full frame mosaic
	struct xrt_rect roi;
	//! Camera's pose relative to the HMD GENERIC_HEAD_POSE
	struct xrt_pose imu_cam_pose;

	//! Constellation tracking - fast tracking thread
	struct os_mutex bw_lock; /* Protects blobwatch process vs release from long thread */
	blobwatch *bw;
	int last_num_blobs;

	//! Full search / pose recovery thread
	struct correspondence_search *cs;

	//! Debug output
	struct u_sink_debug debug_sink;
	struct xrt_pose debug_last_pose;
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

	//! Controller LED timesync tracking
	uint64_t avg_frame_duration;
	uint64_t next_slam_frame_ts;

	/* Debug */
	enum u_logging_level log_level;

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

	/* The frame cadence is SLAM/controller/controller, and we need to pass the
	 * estimate of the next SLAM frame timestamp to the timesync. Only
	 * controller frames get passed to here, so the 2nd sequential controller
	 * frame is the one right before the next SLAM frame */
	bool is_second_frame = (wct->last_frame_sequence + 1) == xf->source_sequence;

	os_mutex_lock(&wct->tracked_controller_lock);

	// Update the controller timesync estimate
	timepoint_ns next_slam_ts;
	timepoint_ns frame_duration = 0;
	if (is_second_frame) {
		frame_duration = xf->timestamp - wct->last_frame_timestamp;
		next_slam_ts = (timepoint_ns)(xf->timestamp) + wct->avg_frame_duration;
	} else if (wct->last_frame_sequence != 0) {
		frame_duration = (xf->timestamp - wct->last_frame_timestamp) / 2;
		next_slam_ts = (timepoint_ns)(xf->timestamp) + 2 * wct->avg_frame_duration;
	}
	if (wct->last_frame_timestamp != 0 && frame_duration > 0 && frame_duration < (2 * U_TIME_1S_IN_NS / 90)) {
		wct->avg_frame_duration = (frame_duration + (29 * wct->avg_frame_duration)) / 30;
	}

	/* If the estimate of the next SLAM frame moves by more than 1ms, update the controllers */
	int64_t slam_ts_diff = next_slam_ts - wct->next_slam_frame_ts;
	if (llabs(slam_ts_diff) > U_TIME_1MS_IN_NS) {
		for (int i = 0; i < wct->num_controllers; i++) {
			wmr_controller_tracker_connection_notify_timesync(wct->controllers[i].connection, next_slam_ts);
		}
	}
	wct->next_slam_frame_ts = next_slam_ts;

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
	struct xrt_space_relation xsr_base_pose;

	/* Allocate a tracking sample for everything we're about to process */
	struct constellation_tracking_sample *sample = constellation_tracking_sample_new();
	uint64_t fast_analysis_start_ts = os_monotonic_get_ns();

	/* Get the HMD's pose so we can calculate the camera view poses */
	xrt_device_get_tracked_pose(wct->hmd_xdev, XRT_INPUT_GENERIC_HEAD_POSE, xf->timestamp, &xsr_base_pose);

	/* Split out camera views and collect blobs across all cameras */
	assert(wct->cam_count <= CONSTELLATION_MAX_CAMERAS);
	sample->n_views = wct->cam_count;

	for (int i = 0; i < wct->cam_count; i++) {
		struct wmr_controller_tracker_camera *cam = wct->cam + i;
		struct tracking_sample_frame *view = sample->views + i;

		//! Calculate the view pose in this sample, from the HMD pose + IMU_camera pose
		math_pose_transform(&xsr_base_pose.pose, &cam->imu_cam_pose, &view->pose);
		cam->debug_last_pose = view->pose;

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

		printf("frame %" PRIu64 " TS %" PRIu64 " cam %d ROI %d,%d w/h %d,%d Blobs: %d\n", xf->source_sequence,
		       xf->timestamp, i, cam->roi.offset.w, cam->roi.offset.h, cam->roi.extent.w, cam->roi.extent.h,
		       bwobs->num_blobs);

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
			continue; // Can't retrieve the pose: means the device was disconnected
		}

		dev_state->dev_index = d;
		sample->n_devices++;
	}
	os_mutex_unlock(&wct->tracked_controller_lock);

	uint64_t fast_analysis_finish_ts = os_monotonic_get_ns();
	wct->last_fast_analysis_ms = (fast_analysis_finish_ts - fast_analysis_start_ts) / U_TIME_1MS_IN_NS;

	/* Send analysis results to debug view if needed */
	for (int i = 0; i < sample->n_views; i++) {
		struct wmr_controller_tracker_camera *cam = wct->cam + i;

		if (u_sink_debug_is_active(&cam->debug_sink)) {
			struct tracking_sample_frame *view = sample->views + i;
			struct xrt_frame *xf_src = view->vframe;
			struct xrt_frame *xf_dbg = NULL;

			u_frame_create_one_off(XRT_FORMAT_R8G8B8, xf_src->width, xf_src->height, &xf_dbg);
			xf_dbg->timestamp = xf_src->timestamp;

			if (view->bwobs != NULL) {
				debug_draw_blobs(xf_dbg, view->bwobs, xf_src);
			}

			u_sink_debug_push_frame(&cam->debug_sink, xf_dbg);
			xrt_frame_reference(&xf_dbg, NULL);
		}
	}

	/* Grab the 'do_full_search' value in case the user clicked the
	 * button in the debug UI */
	if (wct->do_full_search) {
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
	for (int c = 0; c < sample->n_views; c++) {
		struct tracking_sample_frame *view = sample->views + c;
		struct wmr_controller_tracker_camera *cam = wct->cam + c;

		if (view->bwobs == NULL)
			continue; // no blobs in this view

		correspondence_search_set_blobs(cam->cs, view->bwobs->blobs, view->bwobs->num_blobs);

		for (int d = 0; d < sample->n_devices; d++) {
			struct tracking_sample_device_state *dev_state = sample->devices + sample->n_devices;
			struct wmr_controller_tracker_device *device = wct->controllers + dev_state->dev_index;

			//! The fast analysis thread must have retrieved the LED model before
			//  ever sending a device for long analysis
			assert(device->have_led_model == true);

			enum correspondence_search_flags search_flags =
			    CS_FLAG_STOP_FOR_STRONG_MATCH | CS_FLAG_MATCH_ALL_BLOBS | CS_FLAG_DEEP_SEARCH;
			struct pose_metrics score;
			struct xrt_pose obj_cam_pose;

			WMR_INFO(wct, "Doing full search for device %d", device->led_model.id);
			if (correspondence_search_find_one_pose(cam->cs, device->search_led_model, search_flags,
			                                        &obj_cam_pose, NULL, NULL, NULL, 0.0, &score)) {
				WMR_INFO(wct,
				         "Found a pose on cam %d device %d score match_flags 0x%x matched %u "
				         "blobs of %u visible LEDs",
				         c, device->led_model.id, score.match_flags, score.matched_blobs,
				         score.visible_leds);

				mark_matching_blobs(wct, &obj_cam_pose, view->bwobs, &device->led_model,
				                    &cam->camera_model);

				os_mutex_lock(&cam->bw_lock);
				blobwatch_update_labels(cam->bw, view->bwobs, device->led_model.id);
				os_mutex_unlock(&cam->bw_lock);
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

	/* Make sure the long analysis thread isn't running */
	os_thread_helper_stop_and_wait(&wct->long_analysis_thread);
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
	wct->hmd_xdev = hmd_xdev;

	/* Init frame duration to 90fps */
	wct->avg_frame_duration = U_TIME_1S_IN_NS / 90;

	// Set up the per-camera constellation tracking pieces config and pose
	wct->cam_count = slam_calib->cam_count;

	for (int i = 0; i < wct->cam_count; i++) {
		struct wmr_controller_tracker_camera *cam = wct->cam + i;
		struct wmr_camera_config *cam_cfg = hmd_cfg->tcams[i];
		struct t_slam_camera_calibration *cam_slam_cfg = slam_calib->cams + i;

		cam->roi = cam_cfg->roi;
		math_pose_from_isometry(&cam_slam_cfg->T_imu_cam, &cam->imu_cam_pose);

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
	u_var_add_ro_u64(wct, &wct->avg_frame_duration, "Average frame duration (ns)");
	u_var_add_ro_u64(wct, &wct->last_blob_analysis_ms, "Blob tracking time (ms)");
	u_var_add_ro_u64(wct, &wct->last_fast_analysis_ms, "Fast analysis time (ms)");
	u_var_add_ro_u64(wct, &wct->last_long_analysis_ms, "Long analysis time (ms)");
	u_var_add_button(wct, &wct->full_search_button, "Trigger ab-initio search");

	for (int i = 0; i < wct->cam_count; i++) {
		struct wmr_controller_tracker_camera *cam = wct->cam + i;
		u_var_add_ro_i32(wct, &cam->last_num_blobs, "Num Blobs");
		u_var_add_pose(wct, &cam->debug_last_pose, "Last view pose");

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
