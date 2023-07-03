// Copyright 2023 Jan Schmidt
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Implementation of tunnelled controller connection,
 * that translates messages passing via an HP G2 or Sasmung Odyssey+ HMD
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup drv_wmr
 */
#include "os/os_threading.h"
#include "util/u_trace_marker.h"

#include "wmr_controller_tracking.h"
#include "wmr_hmd.h"

DEBUG_GET_ONCE_LOG_OPTION(wmr_log, "WMR_LOG", U_LOGGING_INFO)

#define WMR_TRACE(c, ...) U_LOG_IFL_T(c->log_level, __VA_ARGS__)
#define WMR_DEBUG(c, ...) U_LOG_IFL_D(c->log_level, __VA_ARGS__)
#define WMR_INFO(c, ...) U_LOG_IFL_I(c->log_level, __VA_ARGS__)
#define WMR_WARN(c, ...) U_LOG_IFL_W(c->log_level, __VA_ARGS__)
#define WMR_ERROR(c, ...) U_LOG_IFL_E(c->log_level, __VA_ARGS__)

struct wmr_controller_tracker_connection
{
	/* Controller and tracker each hold a reference. It's
	 * only cleaned up once both release it. */
	struct xrt_reference ref;

	/* Protect access when around API calls and disconnects */
	struct os_mutex lock;
	bool disconnected; /* Set to true once disconnect() is called */

	struct wmr_controller_base *wcb;        //! Controller instance
	struct wmr_controller_tracker *tracker; //! Tracker instance
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
	struct wmr_controller_tracker_connection *controllers[WMR_MAX_CONTROLLERS];

	enum u_logging_level log_level;

	uint64_t last_timestamp;
	uint64_t last_sequence;
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
wmr_controller_tracker_receive_frame(struct xrt_frame_sink *sink, struct xrt_frame *xf)
{
	struct wmr_controller_tracker *wct = container_of(sink, struct wmr_controller_tracker, base);

	/* The frame cadence is SLAM/controller/controller, and we need to pass the estimate of the
	 * next SLAM frame timestamp to the timesync */
	bool is_second_frame = wct->last_sequence + 1 == xf->source_sequence;
	wct->last_timestamp = xf->timestamp;
	wct->last_sequence = xf->source_sequence;

	if (is_second_frame) {
		timepoint_ns next_slam_ts = (timepoint_ns)(xf->timestamp) + U_TIME_1S_IN_NS / 90;
		os_mutex_lock(&wct->tracked_controller_lock);
		for (int i = 0; i < wct->num_controllers; i++) {
			wmr_controller_tracker_connection_notify_timesync(wct->controllers[i], next_slam_ts);
		}
		os_mutex_unlock(&wct->tracked_controller_lock);
	}
}

static void
wmr_controller_tracker_node_break_apart(struct xrt_frame_node *node)
{
	DRV_TRACE_MARKER();
}

static void
wmr_controller_tracker_node_destroy(struct xrt_frame_node *node)
{
	struct wmr_controller_tracker *wct = container_of(node, struct wmr_controller_tracker, node);

	DRV_TRACE_MARKER();
	WMR_DEBUG(wct, "Destroying WMR controller tracker");

	os_mutex_lock(&wct->tracked_controller_lock);
	for (int i = 0; i < wct->num_controllers; i++) {
		if (wct->controllers[i] != NULL) {
			wmr_controller_tracker_connection_disconnect(wct->controllers[i]);
			wct->controllers[i] = NULL;
		}
	}
	os_mutex_unlock(&wct->tracked_controller_lock);
	os_mutex_destroy(&wct->tracked_controller_lock);

	u_var_remove_root(wct);
	free(wct);
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

	u_var_add_root(wct, "WMR Controller Tracker", false);
	u_var_add_log_level(wct, &wct->log_level, "Log Level");
	u_var_add_ro_i32(wct, &wct->num_controllers, "Num Controllers");
	u_var_add_ro_u64(wct, &wct->last_timestamp, "Last Frame Timestamp");

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
wmr_controller_tracker_connection_create(struct wmr_controller_tracker *tracker, struct wmr_controller_base *wcb)
{
	DRV_TRACE_MARKER();

	struct wmr_controller_tracker_connection *wctc = calloc(1, sizeof(struct wmr_controller_tracker_connection));

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

	struct wmr_controller_tracker_connection *wctc = wmr_controller_tracker_connection_create(wct, wcb);
	if (wctc != NULL) {
		wct->controllers[wct->num_controllers] = wctc;
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
