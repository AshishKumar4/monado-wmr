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
#define LED_LOCAL_ID(l) (((l) == LED_INVALID_ID) ? (l) : (l)&0xFF)
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

	/* Weighted greysum centre of blob */
	float x;
	float y;

	/* Motion vector from previous blob */
	float vx;
	float vy;

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
};

typedef struct blobwatch blobwatch;
typedef struct blobservation blobservation;

blobwatch *
blobwatch_new(uint8_t pixel_threshold, uint8_t blob_required_threshold);
void
blobwatch_free(blobwatch *bw);
void
blobwatch_process(blobwatch *bw, struct xrt_frame *frame, blobservation **output);
void
blobwatch_update_labels(blobwatch *bw, blobservation *ob, uint8_t device_id);
void
blobwatch_release_observation(blobwatch *bw, blobservation *ob);
struct blob *
blobwatch_find_blob_at(blobwatch *bw, int x, int y);

#ifdef __cplusplus
}
#endif
