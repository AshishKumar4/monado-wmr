// Copyright 2022-2023, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  @ref drv_wmr driver builder.
 * @author Jakob Bornecrantz <jakob@collabora.com>
 * @ingroup xrt_iface
 */

#include "xrt/xrt_config_drivers.h"
#include "xrt/xrt_prober.h"

#include "util/u_misc.h"
#include "util/u_debug.h"
#include "util/u_logging.h"
#include "util/u_builders.h"
#include "util/u_config_json.h"
#include "util/u_pretty_print.h"
#include "util/u_space_overseer.h"
#include "util/u_system_helpers.h"

#include "target_builder_interface.h"

#include "wmr/wmr_common.h"
#include "wmr/wmr_interface.h"
#include "wmr/wmr_controller_base.h"
#include "wmr/wmr_hmd.h"

#include <assert.h>

#ifndef XRT_BUILD_DRIVER_WMR
#error "Must only be built with XRT_BUILD_DRIVER_WMR set"
#endif

DEBUG_GET_ONCE_LOG_OPTION(wmr_log, "WMR_LOG", U_LOGGING_INFO)


/*
 *
 * Various helper functions and lists.
 *
 */

static const char *driver_list[] = {
    "wmr",
};

static void
print_hmd(u_pp_delegate_t dg,
          const char *prefix,
          enum wmr_headset_type type,
          struct xrt_prober_device *xpdev_holo,
          struct xrt_prober_device *xpdev_companion)
{
	u_pp(dg, "\n\t%s: ", prefix);

	if (xpdev_holo == NULL || xpdev_companion == NULL) {
		u_pp(dg, "None");
		return;
	}

	struct xrt_prober_device *c = xpdev_companion;

	switch (type) {
	case WMR_HEADSET_GENERIC: u_pp(dg, "Generic"); break;
	case WMR_HEADSET_HP_VR1000: u_pp(dg, "HP VR1000"); break;
	case WMR_HEADSET_REVERB_G1: u_pp(dg, "Reverb G1"); break;
	case WMR_HEADSET_REVERB_G2: u_pp(dg, "Reverb G2"); break;
	case WMR_HEADSET_SAMSUNG_XE700X3AI: u_pp(dg, "Samsung XE700X3AI"); break;
	case WMR_HEADSET_SAMSUNG_800ZAA: u_pp(dg, "Samsung 800ZAA"); break;
	case WMR_HEADSET_LENOVO_EXPLORER: u_pp(dg, "Lenovo Explorer"); break;
	case WMR_HEADSET_MEDION_ERAZER_X1000: u_pp(dg, "Medion Erazer X1000"); break;
	case WMR_HEADSET_FUJITSU_FMVHDS1: u_pp(dg, "Fujitsu FMVHDS1"); break;
	default: u_pp(dg, "Unknown (VID: %04x, PID: 0x%04x)", c->vendor_id, c->product_id); break;
	}
}

static void
print_ctrl(u_pp_delegate_t dg, const char *prefix, struct xrt_prober_device *xpdev)
{
	u_pp(dg, "\n\t%s: ", prefix);

	if (xpdev == NULL) {
		u_pp(dg, "None");
		return;
	}

	switch (xpdev->product_id) {
	case WMR_CONTROLLER_PID: u_pp(dg, "WinMR Controller"); break;
	case ODYSSEY_CONTROLLER_PID: u_pp(dg, "Odyssey Controller"); break;
	default: u_pp(dg, "Unknown (VID: %04x, PID: 0x%04x)", xpdev->vendor_id, xpdev->product_id); break;
	}
}


/*
 *
 * Member functions.
 *
 */

static xrt_result_t
wmr_estimate_system(struct xrt_builder *xb,
                    cJSON *config,
                    struct xrt_prober *xp,
                    struct xrt_builder_estimate *out_estimate)
{
	enum u_logging_level log_level = debug_get_log_option_wmr_log();
	struct wmr_bt_controllers_search_results bt_ctrls = {0};
	struct wmr_headset_search_results whsr = {0};
	struct xrt_builder_estimate estimate = {0};
	struct xrt_prober_device **xpdevs = NULL;
	size_t xpdev_count = 0;
	xrt_result_t xret = XRT_SUCCESS;

	/*
	 * Pre device looking stuff.
	 */

	// Lock the device list
	xret = xrt_prober_lock_list(xp, &xpdevs, &xpdev_count);
	if (xret != XRT_SUCCESS) {
		return xret;
	}


	/*
	 * Search for devices.
	 */

	wmr_find_headset(xp, xpdevs, xpdev_count, log_level, &whsr);

	wmr_find_bt_controller_pair(xp, xpdevs, xpdev_count, log_level, &bt_ctrls);

	if (log_level >= U_LOGGING_DEBUG) {
		struct u_pp_sink_stack_only sink;
		u_pp_delegate_t dg = u_pp_sink_stack_only_init(&sink);
		u_pp(dg, "Found:");
		print_hmd(dg, "head", whsr.type, whsr.xpdev_holo, whsr.xpdev_companion);
		print_ctrl(dg, "left", bt_ctrls.left);
		print_ctrl(dg, "right", bt_ctrls.right);

		U_LOG_IFL_D(log_level, "%s", sink.buffer);
	}


	/*
	 * Tidy.
	 */

	xret = xrt_prober_unlock_list(xp, &xpdevs);
	assert(xret == XRT_SUCCESS);


	/*
	 * Fill out estimate.
	 */

	if (whsr.xpdev_holo != NULL && whsr.xpdev_companion != NULL) {
		estimate.certain.head = true;

		if (whsr.type == WMR_HEADSET_REVERB_G2) {
			estimate.maybe.left = true;
			estimate.maybe.right = true;
		}
	}

	if (bt_ctrls.left != NULL) {
		estimate.certain.left = true;
	}

	if (bt_ctrls.right != NULL) {
		estimate.certain.right = true;
	}

	*out_estimate = estimate;

	return XRT_SUCCESS;
}

static xrt_result_t
wmr_open_system_impl(struct xrt_builder *xb,
                     cJSON *config,
                     struct xrt_prober *xp,
                     struct xrt_tracking_origin *origin,
                     struct xrt_system_devices *xsysd,
                     struct xrt_frame_context *xfctx,
                     struct u_builder_roles_helper *ubrh)
{
	enum u_logging_level log_level = debug_get_log_option_wmr_log();
	struct wmr_bt_controllers_search_results bt_ctrls = {0};
	struct wmr_headset_search_results whsr = {0};
	struct xrt_prober_device **xpdevs = NULL;
	size_t xpdev_count = 0;
	xrt_result_t xret_unlock = XRT_SUCCESS;
	xrt_result_t xret = XRT_SUCCESS;

	struct wmr_hmd *head = NULL;
	struct wmr_controller_base *ctrl_left = NULL;
	struct wmr_controller_base *ctrl_right = NULL;

	/*
	 * Pre device looking stuff.
	 */

	// Lock the device list
	xret = xrt_prober_lock_list(xp, &xpdevs, &xpdev_count);
	if (xret != XRT_SUCCESS) {
		return xret;
	}


	/*
	 * Search for devices.
	 */

	wmr_find_headset(xp, xpdevs, xpdev_count, log_level, &whsr);

	wmr_find_bt_controller_pair(xp, xpdevs, xpdev_count, log_level, &bt_ctrls);


	/*
	 * Validation.
	 */

	if (whsr.xpdev_holo == NULL || whsr.xpdev_companion == NULL) {
		U_LOG_IFL_E(log_level, "Could not find headset devices! (holo %p, companion %p)",
		            (void *)whsr.xpdev_holo, (void *)whsr.xpdev_companion);

		xret = XRT_ERROR_DEVICE_CREATION_FAILED;
		goto error;
	}


	/*
	 * Creation.
	 */

	struct xrt_device *ht_left = NULL;
	struct xrt_device *ht_right = NULL;

	xret = wmr_create_headset( //
	    xp,                    //
	    whsr.xpdev_holo,       //
	    whsr.xpdev_companion,  //
	    whsr.type,             //
	    log_level,             //
	    &head,                 //
	    &ctrl_left,            //
	    &ctrl_right,           //
	    &ht_left,              //
	    &ht_right);            //
	if (xret != XRT_SUCCESS) {
		goto error;
	}

	if (ctrl_left == NULL && bt_ctrls.left != NULL) {
		xret = wmr_create_bt_controller(xp, bt_ctrls.left, log_level, &ctrl_left);
		if (xret != XRT_SUCCESS) {
			goto error;
		}
	}

	if (ctrl_right == NULL && bt_ctrls.right != NULL) {
		xret = wmr_create_bt_controller(xp, bt_ctrls.right, log_level, &ctrl_right);
		if (xret != XRT_SUCCESS) {
			goto error;
		}
	}

	if (head != NULL) {
		if (ctrl_left) {
			wmr_controller_attach_to_hmd(ctrl_left, head);
		}
		if (ctrl_right) {
			wmr_controller_attach_to_hmd(ctrl_right, head);
		}
	}

	/*
	 * Tidy
	 */

	xret_unlock = xrt_prober_unlock_list(xp, &xpdevs);
	assert(xret_unlock == XRT_SUCCESS);
	(void)xret_unlock;

	/* We'll give everyone a shared tracking origin using the system allocated one */
	origin->type = XRT_TRACKING_TYPE_OTHER;
	origin->initial_offset.orientation.w = 1.0f;
	origin->initial_offset.position.y = 1.6;
	snprintf(origin->name, XRT_TRACKING_NAME_LEN, "%s", "WMR SLAM Tracking");

	struct xrt_device *head_xdev = wmr_hmd_to_xrt_device(head);
	struct xrt_device *left_xdev = wmr_controller_base_to_xrt_device(ctrl_left);
	struct xrt_device *right_xdev = wmr_controller_base_to_xrt_device(ctrl_right);

	head_xdev->tracking_origin = origin;

	xsysd->xdevs[xsysd->xdev_count++] = head_xdev;
	if (left_xdev != NULL) {
		left_xdev->tracking_origin = origin;
		xsysd->xdevs[xsysd->xdev_count++] = left_xdev;
	}
	if (right_xdev != NULL) {
		right_xdev->tracking_origin = origin;
		xsysd->xdevs[xsysd->xdev_count++] = right_xdev;
	}
	if (ht_left != NULL) {
		ht_left->tracking_origin = origin;
		xsysd->xdevs[xsysd->xdev_count++] = ht_left;
	}
	if (ht_right != NULL) {
		ht_right->tracking_origin = origin;
		xsysd->xdevs[xsysd->xdev_count++] = ht_right;
	}

	// Use hand tracking if no controllers.
	if (left_xdev == NULL) {
		left_xdev = ht_left;
	}
	if (right_xdev == NULL) {
		right_xdev = ht_right;
	}


	// Assign to role(s).
	ubrh->head = head_xdev;
	ubrh->left = left_xdev;
	ubrh->right = right_xdev;
	ubrh->hand_tracking.unobstructed.left = ht_left;
	ubrh->hand_tracking.unobstructed.right = ht_right;

	return XRT_SUCCESS;

error : {
	struct xrt_device *head_xdev = wmr_hmd_to_xrt_device(head);
	struct xrt_device *left_xdev = wmr_controller_base_to_xrt_device(ctrl_left);
	struct xrt_device *right_xdev = wmr_controller_base_to_xrt_device(ctrl_right);

	xrt_device_destroy(&head_xdev);
	xrt_device_destroy(&left_xdev);
	xrt_device_destroy(&right_xdev);
}

	xret_unlock = xrt_prober_unlock_list(xp, &xpdevs);
	assert(xret_unlock == XRT_SUCCESS);

	return xret;
}

static void
wmr_destroy(struct xrt_builder *xb)
{
	free(xb);
}


/*
 *
 * 'Exported' functions.
 *
 */

struct xrt_builder *
t_builder_wmr_create(void)
{
	struct u_builder *ub = U_TYPED_CALLOC(struct u_builder);

	// xrt_builder fields.
	ub->base.estimate_system = wmr_estimate_system;
	ub->base.open_system = u_builder_open_system_static_roles;
	ub->base.destroy = wmr_destroy;
	ub->base.identifier = "wmr";
	ub->base.name = "Windows Mixed Reality";
	ub->base.driver_identifiers = driver_list;
	ub->base.driver_identifier_count = ARRAY_SIZE(driver_list);

	// u_builder fields.
	ub->open_system_static_roles = wmr_open_system_impl;

	return &ub->base;
}
