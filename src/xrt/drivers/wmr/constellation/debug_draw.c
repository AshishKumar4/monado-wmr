/*
 * constellation tracking debug visualisation functions
 * Copyright 2020-2023, Jan Schmidt <thaytan@noraisin.net>
 * SPDX-License-Identifier: BSL-1.0
 */
/*!
 * @file
 * @brief  Debug visualisation for constellation tracking
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup drv_wmr
 */
#include <assert.h>
#include <math.h>
#include <string.h>

#include "debug_draw.h"

/* This will draw RGB flipped on big-endian */
#define WRITE_UINT24_BE(dest, colour)                                                                                  \
	dest[2] = colour & 0xff;                                                                                       \
	dest[1] = colour >> 8 & 0xff;                                                                                  \
	dest[0] = colour >> 16 & 0xff

#define MIN(_a, _b) ((_a) < (_b) ? (_a) : (_b))
#define MAX(_a, _b) ((_a) > (_b) ? (_a) : (_b))

static void
draw_rgb_marker(uint8_t *pixels,
                int width,
                int stride,
                int height,
                int x_pos,
                int y_pos,
                int mark_width,
                int mark_height,
                uint32_t colour)
{
	int x, y;
	int min_x = MAX(0, x_pos - mark_width / 2);
	int max_x = MIN(width, x_pos + mark_width / 2 + 1.5);
	int min_y = MAX(0, y_pos - mark_height / 2);
	int max_y = MIN(height, y_pos + mark_height / 2 + 1.5);

	if (y_pos < 0 || y_pos >= height)
		return;
	if (x_pos < 0 || x_pos >= width)
		return;

	/* Horizontal line */
	uint8_t *dest = pixels + stride * y_pos + 3 * min_x;
	for (x = 0; x < max_x - min_x; x++) {
		WRITE_UINT24_BE(dest, colour);
		dest += 3;
	}

	/* Vertical line */
	dest = pixels + stride * min_y + 3 * x_pos;
	for (y = 0; y < max_y - min_y; y++) {
		WRITE_UINT24_BE(dest, colour);
		dest += stride;
	}
}

static void
clamp(int *val, int max)
{
	if (*val < 0)
		*val = 0;
	if (*val >= max)
		*val = max - 1;
}

static void
clamp_rect(int *x, int *y, int *rw, int *rh, int width, int height)
{
	clamp(x, width);
	clamp(y, height);
	clamp(rw, width - *x);
	clamp(rh, height - *y);
}

#if 0
/* Draw a single-pixel rect outline */
static void
draw_rgb_rect (uint8_t * pixels, int width, int stride, int height,
    int start_x, int start_y, int box_width, int box_height, uint32_t colour)
{
  clamp_rect (&start_x, &start_y, &box_width, &box_height, width, height);

  int x, y;
  uint8_t *dest = pixels + stride * start_y + 3 * start_x;
  for (x = 0; x < box_width; x++) {
    WRITE_UINT24_BE (dest, colour);
    dest += 3;
  }

  for (y = 1; y < box_height - 1; y++) {
    dest = pixels + stride * (start_y + y) + 3 * start_x;

    WRITE_UINT24_BE (dest, colour);
    dest += 3 * (box_width - 1);
    WRITE_UINT24_BE (dest, colour);
  }

  dest = pixels + stride * (start_y + box_height - 1) + 3 * start_x;
  for (x = 0; x < box_width; x++) {
    WRITE_UINT24_BE (dest, colour);
    dest += 3;
  }
}
#endif

static void
colour_rgb_rect(uint8_t *pixels,
                int width,
                int stride,
                int height,
                int start_x,
                int start_y,
                int box_width,
                int box_height,
                uint32_t mask_colour)
{
	clamp_rect(&start_x, &start_y, &box_width, &box_height, width, height);

	int x, y;
	uint8_t *dest;
	/* This will draw RGB flipped on big-endian */
	int r = mask_colour >> 16;
	int g = (mask_colour >> 8) & 0xFF;
	int b = mask_colour & 0xFF;

	for (y = 0; y < box_height; y++) {
		dest = pixels + stride * (start_y + y) + 3 * start_x;
		for (x = 0; x < box_width; x++) {
			dest[0] = (r * dest[0]) >> 8;
			dest[1] = (g * dest[1]) >> 8;
			dest[2] = (b * dest[2]) >> 8;
			dest += 3;
		}
	}
}

void
debug_draw_blobs(struct xrt_frame *rgb_out, struct blobservation *bwobs, struct xrt_frame *gray_in)
{
	int x, y;

	uint8_t *src = gray_in->data;
	int width = gray_in->width;
	int height = gray_in->height;
	int in_stride = gray_in->stride;

	uint8_t *dest = rgb_out->data;
	int out_stride = rgb_out->stride;

	uint8_t *dest_line = dest;
	for (y = 0; y < height; y++) {
		/* Expand to RGB and copy the source to the dest, then
		 * paint blob markers */
		uint8_t *d = dest_line;
		for (x = 0; x < width; x++) {
			/* Expand GRAY8 to RGB */
			d[0] = d[1] = d[2] = src[x];
			d += 3;
		}

		dest_line += out_stride;
		src += in_stride;
	}

	const int colours[] = {0xFF0000, 0x00FF00, 0x0000FF};

	if (bwobs) {
		/* Draw the blobs in the video */
		for (int index = 0; index < bwobs->num_blobs; index++) {
			struct blob *b = bwobs->blobs + index;
			int start_x, start_y, w, h;

			start_x = b->left;
			start_y = b->top;
			w = b->width;
			h = b->height;
			clamp_rect(&start_x, &start_y, &w, &h, width, height);

			/* Tint known blobs by their device ID in the image */
			if (b->led_id != LED_INVALID_ID) {
				int dtype = LED_OBJECT_ID(b->led_id);
				int d = -1;
				switch (dtype) {
				case XRT_DEVICE_TYPE_LEFT_HAND_CONTROLLER: d = 1; break;
				case XRT_DEVICE_TYPE_RIGHT_HAND_CONTROLLER: d = 2; break;
				default: break;
				}
				assert(d != -1);

				colour_rgb_rect(dest, width, out_stride, height, start_x, start_y, b->width, b->height,
				                colours[d]);

				/* Draw a dot at the weighted center */
				draw_rgb_marker(dest, width, out_stride, height, round(b->x), round(b->y), 0, 0,
				                colours[d]);
			} else {
				/* Draw a dot at the weighted center in purple */
				draw_rgb_marker(dest, width, out_stride, height, round(b->x), round(b->y), 0, 0,
				                0xFF00FF);
			}
		}
	}
}
