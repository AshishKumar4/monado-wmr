/*
 * Blob detection
 * Copyright 2014-2015 Philipp Zabel
 * Copyright 2019-2023 Jan Schmidt
 * SPDX-License-Identifier:	LGPL-2.0+ or BSL-1.0
 */
/*!
 * @file
 * @brief  Blob thresholding and tracking in camera frames
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup constellation
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>

#include "xrt/xrt_defines.h"
#include "util/u_frame.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_BLOBS_PER_FRAME 100
#define LED_INVALID_ID ((uint16_t)(-1))
#define LED_NOISE_ID ((uint16_t)(-2))
#define LED_LOCAL_ID(l) (((l) == LED_INVALID_ID) ? (l) : (l) & 0xFF)
#define LED_OBJECT_ID(l) (((l) == LED_INVALID_ID) ? (l) : (l) >> 8)
#define LED_MAKE_ID(o, n) ((uint16_t)((uint16_t)(o)) << 8 | ((uint16_t)(n)))

/* 0x24 works much better for Rift CV1, but the threshold needs
 * to be higher for DK2 which has more background bleed and bigger
 * tracking LEDs */
#define BLOB_PIXEL_THRESHOLD_CV1 0x24
#define BLOB_PIXEL_THRESHOLD_DK2 0x7f
#define BLOB_THRESHOLD_MIN_OCULUS 0x40

struct blob
{
	/* Each new blob is assigned a unique ID and used
	 * to match between frames when updating blob labels
	 * from a delayed long-analysis. 4 billion
	 * ought to be enough before it wraps
	 */
	uint32_t blob_id;

	/* Weighted greysum / sub-pixel refined centre of blob */
	float x;
	float y;

	/* Centroid measurement variance (R), in px^2: how uncertain (x,y) is as an
	 * LED centre. Isotropic 1-DoF (the spot is roughly round); a downstream
	 * matcher can use it as the per-blob measurement noise to Mahalanobis-weight
	 * reprojection cost (cost = (dx^2+dy^2)/pos_var_px2). Grows with blob spatial
	 * spread, saturation fraction, frame-edge truncation and relative dimness, so
	 * a fat/clipped/edge/faint blob is a less certain centre. */
	float pos_var_px2;

	/* Motion vector from previous blob */
	float vx;
	float vy;

	/* The max brightness we see in the blob */
	uint8_t brightness;

	/* bounding box */
	uint16_t top;
	uint16_t left;

	uint16_t width;
	uint16_t height;
	uint32_t area;
	uint32_t age;
	int16_t track_index;

	uint32_t id_age;
	uint16_t led_id;
	uint16_t prev_led_id;
};

/*
 * Stores all blobs observed in a single frame.
 */
struct blobservation
{
	int num_blobs;
	struct blob blobs[MAX_BLOBS_PER_FRAME];
	uint8_t tracked[MAX_BLOBS_PER_FRAME];

	int dropped_dark_blobs;
	int dropped_shape_blobs;
};

typedef struct blobwatch blobwatch;
typedef struct blobservation blobservation;

blobwatch *
blobwatch_new(uint8_t pixel_threshold, uint8_t blob_required_threshold, uint8_t cam_id);
void
blobwatch_free(blobwatch *bw);
void
blobwatch_process(blobwatch *bw, struct xrt_frame *frame, uint16_t exposure, uint16_t gain, blobservation **output);
//! Predictive-ROI variant: scans only the rectangle [roi_x, roi_x+roi_w) x [roi_y, roi_y+roi_h) of the
//! frame, instead of the full image. Convergent with the Oasis driver's `ConnectedComponent::Locate`
//! which restricts blob search to a bounding box around the ESKF-predicted LED positions padded by
//! PredictivePatchSize/2 = 8 px per side. A 0-area or fully-covering ROI degrades cleanly to the
//! full-frame `blobwatch_process` behaviour, so the caller can pass a generous ROI without harm.
//! ROI is clamped to the frame bounds internally; the integral image for the adaptive threshold is
//! computed over the ROI only (the local-background lookup needs the same window the scanner uses).
void
blobwatch_process_roi(blobwatch *bw,
                      struct xrt_frame *frame,
                      uint16_t exposure,
                      uint16_t gain,
                      int roi_x,
                      int roi_y,
                      int roi_w,
                      int roi_h,
                      blobservation **output);

//! Predictive-ROI variant with caller-supplied lower thresholds. A zero threshold keeps the blobwatch
//! default for that component.
void
blobwatch_process_roi_lowthresh(blobwatch *bw,
                                struct xrt_frame *frame,
                                uint16_t exposure,
                                uint16_t gain,
                                int roi_x,
                                int roi_y,
                                int roi_w,
                                int roi_h,
                                uint8_t roi_pixel_threshold,
                                uint8_t roi_adapt_margin,
                                uint8_t roi_required_threshold,
                                blobservation **output);
void
blobwatch_update_labels(blobwatch *bw, blobservation *ob, uint8_t device_id);
void
blobwatch_release_observation(blobwatch *bw, blobservation *ob);

#ifdef __cplusplus
}
#endif
