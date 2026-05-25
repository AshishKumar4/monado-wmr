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
#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/u_logging.h"
#include "util/u_g2_telemetry.h"

#include "blobwatch.h"

/* Keep enough history frames that the caller can keep hold
 * of a previous blobservation and pass it to the long term
 * tracker while we still have 2 left to ping-pong between */
#define NUM_FRAMES_HISTORY 5
#define MAX_EXTENTS_PER_LINE 30

/* Set to 1 to do extra array tracking consistency checks */
#define CONSISTENCY_CHECKS 0

/* --- Blob-detection tuning ---
 * Few knobs, all derived where possible. Values calibrated against real G2
 * controller captures (short LED-flash exposure: dim spots peak ~24..200;
 * fill ~0.7..0.9; peak/mean ~1.1..2.6, lower for dimmer LEDs). */

/* Local adaptive threshold: a pixel only counts as blob if it is both above
 * the global floor AND at least this many counts above its local background.
 * This separates a dim LED next to a bright window and stops a uniformly
 * bright background becoming blobs, without a full second pass. Kept small so
 * it lifts the floor without eroding real LED spots (validated on captures:
 * ~99% of dark-frame LEDs retained, bright-room false blobs cut >10x). */
#define ADAPT_MARGIN 6
/* Half-size of the local-background box (window = 2*r+1). Large enough to
 * straddle an LED spot so the LED itself barely lifts its own background. */
#define ADAPT_BG_RADIUS 12

/* A pixel >= this is treated as saturated/clipped. 255 is the 8-bit physical ceiling; the 5-count guard
 * band also catches near-clipped pixels whose greysum centroid is already biased. */
#define SATURATION_LEVEL (255 - 5)

/* Blob qualification (reject reflections / streaks / window edges). */
#define MIN_FILL_RATIO 0.40f /* area / bbox-area; round spot is high, streak low */
/* Peaked LED spot vs flat bright patch. Validated on real captures: across 209 certain LEDs (peak>=32)
 * the ratio is always >=1.61 (0% dropped at 1.30), while room-noise blobs cluster <=1.26 (93% rejected). */
#define MIN_PEAK_TO_MEAN 1.30f
#define MIN_BLOB_AREA 3 /* drop salt-and-pepper specks */

/* Centroid-uncertainty (R) inflation factors. A fully clipped or frame-edge-truncated spot has lost the
 * symmetry information its centre relies on, so its variance is inflated up to these multiples. Derived as
 * worst-case multipliers on the measured spread, not absolute pixel magic numbers: at full saturation the
 * usable signal is the contour alone (~SAT_R_INFLATE_MAX wider), and a contour cut by the frame border is
 * one-sided (~EDGE_R_INFLATE wider). A blob far dimmer than the frame's brightest credible spot is a weaker
 * LED candidate (possible faint reflection), so its R is scaled up to DIM_R_INFLATE_MAX as it approaches the
 * detect floor. All are linear blends keyed to a measured fraction in [0,1] -> no abrupt cull. */
#define SAT_R_INFLATE_MAX 4.0f
#define EDGE_R_INFLATE 3.0f
#define DIM_R_INFLATE_MAX 4.0f

#define QUEUE_ENTRIES (NUM_FRAMES_HISTORY + 1)

#define abs(x) ((x) >= 0 ? (x) : -(x))
#define min(x, y) ((x) < (y) ? (x) : (y))
#define max(x, y) ((x) > (y) ? (x) : (y))

struct extent
{
	uint16_t start;
	uint16_t end;
	/* inherited parameters */
	uint16_t top;
	uint16_t left;
	uint16_t right;
	uint32_t area;

	/* Maximum pixel colour detected */
	uint8_t max_pixel;

	/* Accumulated over the whole (merged) blob for qualification, so no
	 * second pass over the pixels is needed. */
	uint32_t intensity_sum; /* sum of pixel values (for peak/mean) */
	uint32_t sat_count;     /* number of saturated pixels */
};

struct extent_line
{
	struct extent extents[MAX_EXTENTS_PER_LINE];
	uint16_t num;
	uint16_t padding[3];
};

typedef struct blobservation_queue blobservation_queue;

struct blobservation_queue
{
	blobservation *data[QUEUE_ENTRIES];
	unsigned int head, tail;
};

#define INIT_QUEUE(q) (q)->head = (q)->tail = 0;

#define PUSH_QUEUE(q, b)                                                                                               \
	do {                                                                                                           \
		unsigned int next = ((q)->tail + 1) % QUEUE_ENTRIES;                                                   \
		assert(next != (q)->head); /* Check there's room */                                                    \
		assert((b) != NULL);                                                                                   \
		(q)->data[(q)->tail] = (b);                                                                            \
		(q)->tail = next;                                                                                      \
	} while (0)

static blobservation *
POP_QUEUE(blobservation_queue *q)
{
	blobservation *b;
	unsigned int next_head = (q->head + 1) % QUEUE_ENTRIES;

	if ((q)->tail == (q)->head) /* Check there's something in the queue */
		return NULL;

	b = q->data[q->head];
	q->head = next_head;

	return b;
}

/*
 * Blob detector internal state
 */
struct blobwatch
{
	uint32_t next_blob_id;
	uint8_t pixel_threshold;         /* Minimum pixel magnitude considered non-black */
	uint8_t blob_required_threshold; /* Minimum pixel magnitude a blob must contain somewhere to be retained */
	uint8_t cam_id;                  /* Camera index this blobwatch processes (for telemetry) */
	int blob_max_wh;

	/* Blob qualification: reject elongated non-LED shapes (window edges, reflections, streaks). An LED
	 * images as a small, roughly round spot. Fill-ratio and peak-to-mean are compile-time shape priors
	 * (MIN_FILL_RATIO / MIN_PEAK_TO_MEAN, data-validated); aspect stays a field as it is the one shape
	 * cut a per-unit/bright-room calibration may want to vary. Set 0 to disable. */
	float blob_min_aspect; /* min(w,h)/max(w,h): ~1 for an LED, low for a streak */

	bool debug;

	blobservation observations[NUM_FRAMES_HISTORY];

	blobservation_queue observation_q;

	blobservation *last_observation;

	/* Integral image of the frame, used for an O(1) local-background box mean
	 * per pixel (adaptive threshold). Lazily (re)allocated to frame size. */
	uint32_t *integral;
	int integral_w, integral_h; /* allocated dims = frame (w+1) x (h+1) */
};

/*
 * Allocates and initializes blobwatch structure.
 *
 * Returns the newly allocated blobwatch structure.
 */
blobwatch *
blobwatch_new(uint8_t pixel_threshold, uint8_t blob_required_threshold, uint8_t cam_id)
{
	blobwatch *bw = malloc(sizeof(*bw));
	int i;

	if (!bw)
		return NULL;

	memset(bw, 0, sizeof(*bw));
	bw->next_blob_id = 1;
	bw->cam_id = cam_id;

	/* Minimum pixel magnitude to be included in a blob at all */
	bw->pixel_threshold = pixel_threshold;

	/* Require at least 1 pixel over this threshold in a blob -
	 * allows for collecting fainter blobs, as long as they have a bright
	 * point somewhere, and helps to eliminate generally faint background noise */
	bw->blob_required_threshold = blob_required_threshold;

	/* Don't store blobs that are too big to be LEDs sensibly
	 * (arbitrary 35 pixel cut-off. FIXME: revisit this number) */
	bw->blob_max_wh = 35;
	bw->blob_min_aspect = 0.30f;

	bw->last_observation = NULL;
	bw->debug = true;

	INIT_QUEUE(&bw->observation_q);
	/* Push all observations into the available queue */
	for (i = 0; i < NUM_FRAMES_HISTORY; i++)
		PUSH_QUEUE(&bw->observation_q, bw->observations + i);

	return bw;
}

void
blobwatch_free(blobwatch *bw)
{
	free(bw->integral);
	free(bw);
}

/*
 * Builds an integral image of the frame (single pass), so the mean of any
 * box can be read in O(1). Used for the local adaptive threshold.
 */
static void
build_integral(blobwatch *bw, struct xrt_frame *frame)
{
	const int w = frame->width, h = frame->height;
	const int iw = w + 1, ih = h + 1;

	if (bw->integral == NULL || bw->integral_w != iw || bw->integral_h != ih) {
		free(bw->integral);
		bw->integral = malloc((size_t)iw * ih * sizeof(uint32_t));
		if (bw->integral == NULL) {
			/* OOM on the realtime camera thread: degrade to the global-threshold-only path
			 * (local_bg_mean returns 0 when integral==NULL) instead of dereferencing NULL; the
			 * zeroed dims make the next frame retry the allocation. */
			bw->integral_w = bw->integral_h = 0;
			return;
		}
		bw->integral_w = iw;
		bw->integral_h = ih;
	}

	/* uint32 holds the running sum: max value = w*h*255; safe for any sensor up to ~16.8 Mpx
	 * (UINT32_MAX/255). The G2 cameras are 640x480, far inside that bound. */
	uint32_t *I = bw->integral;
	memset(I, 0, (size_t)iw * sizeof(uint32_t)); /* top border row */

	for (int y = 0; y < h; y++) {
		const uint8_t *src = frame->data + (size_t)frame->stride * y;
		uint32_t *row = I + (size_t)(y + 1) * iw;
		const uint32_t *prev = I + (size_t)y * iw;
		uint32_t run = 0;
		row[0] = 0; /* left border column */
		for (int x = 0; x < w; x++) {
			run += src[x];
			row[x + 1] = prev[x + 1] + run;
		}
	}
}

/* Local background mean around (x,y) from the integral image. Returns 0 if the integral image is
 * unavailable (OOM fallback), which makes the adaptive threshold degrade to the global floor. */
static inline uint32_t
local_bg_mean(const blobwatch *bw, int x, int y, int w, int h)
{
	if (bw->integral == NULL) {
		return 0;
	}
	const int iw = bw->integral_w;
	int x0 = x - ADAPT_BG_RADIUS, x1 = x + ADAPT_BG_RADIUS;
	int y0 = y - ADAPT_BG_RADIUS, y1 = y + ADAPT_BG_RADIUS;
	if (x0 < 0)
		x0 = 0;
	if (y0 < 0)
		y0 = 0;
	if (x1 >= w)
		x1 = w - 1;
	if (y1 >= h)
		y1 = h - 1;

	const uint32_t *I = bw->integral;
	uint32_t sum = I[(size_t)(y1 + 1) * iw + (x1 + 1)] - I[(size_t)y0 * iw + (x1 + 1)] -
	               I[(size_t)(y1 + 1) * iw + x0] + I[(size_t)y0 * iw + x0];
	uint32_t cnt = (uint32_t)(x1 - x0 + 1) * (uint32_t)(y1 - y0 + 1);
	return sum / cnt;
}

/*
 * Saturation-contour centre of one axis.
 *
 * A clipped spot has a flat plateau, so the intensity-weighted centroid is
 * driven by the (possibly asymmetric) skirt rather than the true centre. The
 * saturated region itself, however, is symmetric about the spot centre for a
 * symmetric PSF regardless of plateau width: its centre is the midpoint of the
 * first and last saturated samples projected onto the axis. @p lo / @p hi are
 * the smallest/largest saturated coordinate on this axis (inclusive, in the
 * extent's local 0-based frame); @p origin places them in absolute pixel coords.
 * The midpoint origin + (lo+hi)/2 is the sub-pixel centre under the same
 * pixel-index = pixel-centre convention as the greysum centroid (which is why no
 * extra half-pixel offset is added). Returns false if no saturated sample seen.
 */
static inline bool
sat_contour_center(int lo, int hi, float origin, float *out)
{
	if (lo > hi) {
		return false;
	}
	*out = origin + (lo + hi) * 0.5f;
	return true;
}

/*
 * Sub-pixel centroid of one blob, plus its centroid measurement variance (R, px^2).
 *
 * Unsaturated: intensity-weighted greysum centre, then a parabolic (3-point)
 * peak fit on the row/column profiles through the brightest pixel, which lowers
 * centroid noise on a roughly Gaussian spot.
 *
 * Saturated: the clipped plateau biases the greysum, so the centre is taken from
 * the saturation contour (the midpoint of the saturated span on each axis), which
 * is symmetric about the true centre independent of the plateau width or skirt
 * asymmetry. Falls back to the bbox centre only if the contour is somehow empty.
 *
 * The variance is derived from the spot's own intensity-weighted spatial spread
 * (second central moment): a fat spot localises its centre less precisely. It is
 * the source-of-truth per-blob R that a downstream matcher can use to weight the
 * reprojection cost; @ref store_blob inflates it further for saturation, frame-edge
 * truncation and relative dimness.
 */
static void
compute_greysum(blobwatch *bw,
                struct xrt_frame *frame,
                struct extent *e,
                int end_y,
                bool saturated,
                float *led_x,
                float *led_y,
                float *pos_var_px2)
{
	const uint16_t width = e->right - e->left + 1;
	const uint16_t height = end_y - e->top + 1;
	const uint8_t weight_cap = saturated ? (SATURATION_LEVEL - 1) : 255;
	uint8_t *pixels;
	uint16_t x, y;
	uint32_t x_pos, y_pos;
	uint64_t greysum_total = 0, greysum_x = 0, greysum_y = 0;
	/* Second moments (for the centroid variance), accumulated about the extent origin. */
	uint64_t greysum_xx = 0, greysum_yy = 0;

	/* Brightest-pixel location and its row/column neighbour values, for the
	 * parabolic peak fit (only meaningful when not saturated). */
	uint32_t peak_val = 0;
	int peak_x = 0, peak_y = 0;

	/* Saturation-contour bounds (local 0-based extent coords) for the saturated centre. */
	int sat_x_lo = INT_MAX, sat_x_hi = -1, sat_y_lo = INT_MAX, sat_y_hi = -1;

	/* Point to top left pixel of the extent */
	pixels = frame->data + frame->stride * e->top + e->left;

	for (y = 0; y < height; y++) {
		/* Use 1...frame_width/height as coords, otherwise the first column never contributes anything */
		y_pos = e->top + y + 1;
		x_pos = e->left + 1;

		for (x = 0; x < width; x++) {
			uint32_t pix = pixels[x];

			if (pix > peak_val) {
				peak_val = pix;
				peak_x = e->left + x;
				peak_y = e->top + y;
			}
			if (pix >= SATURATION_LEVEL) {
				if (x < sat_x_lo)
					sat_x_lo = x;
				if (x > sat_x_hi)
					sat_x_hi = x;
				if (y < sat_y_lo)
					sat_y_lo = y;
				if (y > sat_y_hi)
					sat_y_hi = y;
			}

			/* Drop the clipped plateau from the weighted sums for saturated blobs: their centre
			 * comes from the saturation contour below, and these moments then measure only the
			 * informative skirt spread (the variance). Unsaturated blobs weight every pixel. */
			uint32_t w = pix <= weight_cap ? pix : 0;
			greysum_total += w;
			greysum_x += (uint64_t)x_pos * w;
			greysum_y += (uint64_t)y_pos * w;
			greysum_xx += (uint64_t)x_pos * x_pos * w;
			greysum_yy += (uint64_t)y_pos * y_pos * w;
			x_pos++;
		}

		pixels += frame->stride;
	}

	/* Centroid variance from the intensity-weighted spatial spread: Var = E[r^2] - E[r]^2 per axis
	 * (the spot's second central moment), averaged over both axes for an isotropic 1-DoF R. A larger,
	 * more spread-out spot pins its centre less tightly. Floor at sub-pixel so a clean tight spot still
	 * carries a small, finite R (never zero -> downstream division stays well-posed). */
	if (greysum_total > 0) {
		const double mx = (double)greysum_x / greysum_total;
		const double my = (double)greysum_y / greysum_total;
		double vx = (double)greysum_xx / greysum_total - mx * mx;
		double vy = (double)greysum_yy / greysum_total - my * my;
		if (vx < 0.0)
			vx = 0.0;
		if (vy < 0.0)
			vy = 0.0;
		*pos_var_px2 = (float)(0.5 * (vx + vy));
	} else {
		/* Fully-saturated spot: no weighted spread, fall back to the bbox half-extent as the spread. */
		const float r = 0.5f * (width + height) * 0.5f;
		*pos_var_px2 = r * r;
	}
	if (*pos_var_px2 < 0.25f)
		*pos_var_px2 = 0.25f; /* >= (0.5 px)^2 */

	if (saturated) {
		/* Saturation-contour centre (robust to clipped plateau + skirt asymmetry). */
		float cx, cy;
		bool okx = sat_contour_center(sat_x_lo, sat_x_hi, (float)e->left, &cx);
		bool oky = sat_contour_center(sat_y_lo, sat_y_hi, (float)e->top, &cy);
		*led_x = okx ? cx : e->left + (width - 1) * 0.5f;
		*led_y = oky ? cy : e->top + (height - 1) * 0.5f;
		return;
	}

	if (greysum_total == 0) {
		/* Tiny unsaturated blob whose only pixels were clipped (shouldn't happen here, but stay safe) */
		*led_x = e->left + (width - 1) * 0.5f;
		*led_y = e->top + (height - 1) * 0.5f;
		return;
	}

	*led_x = (float)greysum_x / greysum_total - 1;
	*led_y = (float)greysum_y / greysum_total - 1;

	/* Parabolic peak refinement: blend the greysum centre toward the
	 * sub-pixel peak of the 3-point profile through the brightest pixel.
	 * Skipped for saturated blobs (no single peak) and at frame edges. */
	if (peak_x > 0 && peak_x < (int)frame->width - 1 && peak_y > 0 && peak_y < (int)frame->height - 1) {
		uint8_t *p = frame->data + frame->stride * peak_y + peak_x;
		float c = peak_val;
		float lx = p[-1], rx = p[1];
		float ux = p[-frame->stride], dx = p[frame->stride];

		float denx = lx - 2.0f * c + rx;
		float deny = ux - 2.0f * c + dx;
		/* Only accept a concave (true peak) fit; clamp the shift to ±0.5 px */
		if (denx < 0.0f) {
			float dxs = 0.5f * (lx - rx) / denx;
			if (dxs > -0.5f && dxs < 0.5f)
				*led_x = peak_x + dxs;
		}
		if (deny < 0.0f) {
			float dys = 0.5f * (ux - dx) / deny;
			if (dys > -0.5f && dys < 0.5f)
				*led_y = peak_y + dys;
		}
	}
}

/*
 * Stores blob information collected in the last extent e into the blob
 * array b at the given index.
 */
static inline void
store_blob(struct extent *e,
           int index,
           int end_y,
           struct blob *b,
           uint32_t blob_id,
           float led_x,
           float led_y,
           float pos_var_px2,
           uint8_t brightness)
{
	b += index;
	b->blob_id = blob_id;
	b->x = led_x;
	b->y = led_y;
	b->pos_var_px2 = pos_var_px2;
	b->vx = 0;
	b->vy = 0;

	b->left = e->left;
	b->top = e->top;
	b->width = e->right - e->left + 1;
	b->height = end_y - e->top + 1;
	b->area = e->area;
	b->age = 0;
	b->track_index = -1;
	b->id_age = 0;
	b->prev_led_id = b->led_id = LED_INVALID_ID;
	b->brightness = brightness;
}

static void
extent_to_blobs(blobwatch *bw, blobservation *ob, struct extent *e, int y, struct xrt_frame *frame)
{
	const int max_blobs = MAX_BLOBS_PER_FRAME;
	struct blob *blobs = ob->blobs;

	/* Don't store unless there was at least one "bright enough" pixel in the blob */
	if (e->max_pixel < bw->blob_required_threshold) {
		ob->dropped_dark_blobs++;
		return;
	}

	/* Don't store 1x1 blobs */
	if (e->top == y && e->left == e->right) {
		ob->dropped_shape_blobs++;
		return;
	}

	/* Check width and height against the blob "maximum size" */
	if (y - e->top > bw->blob_max_wh || e->right - e->left > bw->blob_max_wh) {
		ob->dropped_shape_blobs++;
		return;
	}

	const uint32_t bb_w = e->right - e->left + 1;
	const uint32_t bb_h = y - e->top + 1;

	/* Drop tiny specks (sensor noise that survived the adaptive threshold). */
	if (e->area < MIN_BLOB_AREA) {
		ob->dropped_shape_blobs++;
		return;
	}

	/* Reject non-LED shapes (window edges / reflections / streaks): an LED images as a small, roughly
	 * round, well-filled, peaked spot. Aspect-ratio rejects elongated streaks/edges; fill-ratio rejects
	 * sparse blobs; peak-to-mean rejects flat bright patches — before they pollute the matcher. */
	{
		const uint32_t lo = (bb_w < bb_h) ? bb_w : bb_h;
		const uint32_t hi = (bb_w > bb_h) ? bb_w : bb_h;
		if (bw->blob_min_aspect > 0.0f && hi > 0 && (float)lo / (float)hi < bw->blob_min_aspect) {
			ob->dropped_shape_blobs++;
			return;
		}
	}
	const float fill_ratio = (float)e->area / (float)(bb_w * bb_h);
	const float mean = (float)e->intensity_sum / (float)e->area;
	const float peak_to_mean = (float)e->max_pixel / (mean > 0.0f ? mean : 1.0f);
	const bool saturated = e->sat_count > 0;
	if (fill_ratio < MIN_FILL_RATIO || peak_to_mean < MIN_PEAK_TO_MEAN) {
		ob->dropped_shape_blobs++; // a shape/quality reject, not a brightness (dark) one
		return;
	}

	/* In the future we could generate multiple blobs from one extent if we detect
	 * it as multiple LEDs */
	while (ob->num_blobs < max_blobs) {
		float led_x, led_y, pos_var_px2;

		compute_greysum(bw, frame, e, y, saturated, &led_x, &led_y, &pos_var_px2);

		/* Inflate R for centroid information lost to clipping and to frame-border truncation. The
		 * saturated fraction (clipped pixels / area) interpolates toward SAT_R_INFLATE_MAX; touching
		 * the frame border multiplies by EDGE_R_INFLATE (its skirt/contour is one-sided). The remaining
		 * dimness term is applied in process_frame once the brightest blob is known. */
		const float sat_frac = (float)e->sat_count / (float)e->area;
		float infl = 1.0f + sat_frac * (SAT_R_INFLATE_MAX - 1.0f);
		const bool on_edge = e->left == 0 || e->top == 0 || (int)e->right == (int)frame->width - 1 ||
		                     y == (int)frame->height - 1;
		if (on_edge)
			infl *= EDGE_R_INFLATE;
		pos_var_px2 *= infl;

		store_blob(e, ob->num_blobs++, y, blobs, bw->next_blob_id++, led_x, led_y, pos_var_px2,
		           e->max_pixel);
		break;
	}
}

/*
 * Collects contiguous ranges of pixels with values larger than a threshold of
 * THRESHOLD in a given scanline and stores them in extents. Processing stops after
 * num_extents.
 * Extents are marked with the same index as overlapping extents of the previous
 * scanline, and properties of the formed blobs are accumulated.
 *
 * Returns the number of extents found.
 */
static void
process_scanline(uint8_t *line,
                 blobwatch *bw,
                 uint32_t y,
                 struct extent_line *el,
                 struct extent_line *prev_el,
                 struct xrt_frame *frame,
                 blobservation *ob)
{
	struct extent *le_end = prev_el->extents;
	struct extent *le = prev_el->extents;
	struct extent *extent = el->extents;
	int num_extents = MAX_EXTENTS_PER_LINE;
	float center;
	uint32_t x;
	int e = 0;

	if (prev_el)
		le_end += prev_el->num;

	for (x = 0; x < frame->width; x++) {
		int start, end;
		bool is_new_extent = true;
		uint8_t max_pixel = 0;
		uint32_t intensity_sum = 0;
		uint32_t sat_count = 0;

		/* Adaptive threshold: a pixel must clear the global floor AND stand
		 * out from its local background. Cheap (one integral-image lookup),
		 * separates dim LEDs from bright reflections and stops a uniformly
		 * bright background from forming blobs. */
		if (line[x] <= bw->pixel_threshold ||
		    line[x] < local_bg_mean(bw, x, y, frame->width, frame->height) + ADAPT_MARGIN)
			continue;

		start = x;
		max_pixel = line[x];
		intensity_sum += line[x];
		if (line[x] >= SATURATION_LEVEL)
			sat_count++;
		x++;

		/* Loop until pixel value falls below threshold */
		while (x < frame->width && line[x] > bw->pixel_threshold &&
		       line[x] >= local_bg_mean(bw, x, y, frame->width, frame->height) + ADAPT_MARGIN) {
			if (line[x] > max_pixel)
				max_pixel = line[x];
			intensity_sum += line[x];
			if (line[x] >= SATURATION_LEVEL)
				sat_count++;
			x++;
		}

		end = x - 1;

		center = (start + end) / 2.0;

		extent->start = start;
		extent->end = end;
		extent->area = x - start;
		extent->max_pixel = max_pixel;
		extent->intensity_sum = intensity_sum;
		extent->sat_count = sat_count;

		if (prev_el) {
			/*
			 * Previous extents without significant overlap are the
			 * bottom of finished blobs. Store them into an array.
			 */
			while (le < le_end && le->end < center) {
				extent_to_blobs(bw, ob, le, y, frame);
				le++;
			}

			/*
			 * A previous extent with significant overlap is
			 * considered to be part of the same blob.
			 */
			if (le < le_end && le->start <= center && le->end >= center) {
				extent->top = le->top;
				extent->left = min(extent->start, le->left);
				extent->right = max(extent->end, le->right);
				if (le->max_pixel > extent->max_pixel)
					extent->max_pixel = le->max_pixel;
				extent->area += le->area;
				extent->intensity_sum += le->intensity_sum;
				extent->sat_count += le->sat_count;
				is_new_extent = false;
				le++;
			}
		}

		/*
		 * If this extent is not part of a previous blob, increment the
		 * blob index.
		 */
		if (is_new_extent) {
			extent->top = y;
			extent->left = extent->start;
			extent->right = extent->end;
		}

		if (++e == num_extents)
			break;
		extent++;
	}

	if (prev_el) {
		/*
		 * If there are no more extents on this line, all remaining
		 * extents in the previous line are finished blobs. Store them.
		 */
		while (le < le_end) {
			extent_to_blobs(bw, ob, le, y, frame);
			le++;
		}
	}

	el->num = e;

	if (y == frame->height - 1) {
		/* All extents of the last line are finished blobs, too. */
		for (extent = el->extents; extent < el->extents + el->num; extent++) {
			extent_to_blobs(bw, ob, extent, y, frame);
		}
	}
}

/*
 * Processes extents from all scanlines in a frame and stores the
 * resulting blobs in ob->blobs.
 */
static void
process_frame(blobwatch *bw, blobservation *ob, struct xrt_frame *frame)
{
	struct extent_line el1;
	struct extent_line el2;

	ob->num_blobs = 0;
	ob->dropped_dark_blobs = 0;
	ob->dropped_shape_blobs = 0;

	/* Integral image for the per-pixel local-background lookup (adaptive threshold) */
	build_integral(bw, frame);

	uint8_t *line = frame->data;
	process_scanline(line, bw, 0, &el1, NULL, frame, ob);

	for (uint32_t y = 1; y < frame->height; y++) {
		process_scanline(line, bw, y, y & 1 ? &el2 : &el1, y & 1 ? &el1 : &el2, frame, ob);
		line += frame->stride;
	}

	/* Relative-dimness R inflation (within-frame intensity rank): the brightest blob is the strongest LED
	 * candidate; a much dimmer blob is more likely a faint reflection, so its centroid R is scaled up
	 * toward DIM_R_INFLATE_MAX as its peak falls from the frame max toward the detect floor. Uncertainty-
	 * aware (no cull) and self-referenced (no absolute brightness constant). Single blob -> no rank, no-op. */
	if (ob->num_blobs >= 2) {
		uint8_t bmax = 0;
		for (int i = 0; i < ob->num_blobs; i++)
			bmax = max(bmax, ob->blobs[i].brightness);
		const float span = (float)bmax - (float)bw->blob_required_threshold;
		if (span > 0.0f) {
			for (int i = 0; i < ob->num_blobs; i++) {
				struct blob *b = &ob->blobs[i];
				float dim = ((float)bmax - (float)b->brightness) / span; /* 0 at peak, 1 at floor */
				if (dim < 0.0f)
					dim = 0.0f;
				b->pos_var_px2 *= 1.0f + dim * (DIM_R_INFLATE_MAX - 1.0f);
			}
		}
	}
}

/*
 * Finds the first free tracking slot.
 */
static int
find_free_track(uint8_t *tracked)
{
	int i;

	for (i = 0; i < MAX_BLOBS_PER_FRAME; i++) {
		if (tracked[i] == 0)
			return i;
	}

	return -1;
}

void
copy_matching_blob(struct blob *to, struct blob *from)
{
	to->blob_id = from->blob_id;
	to->vx = to->x - from->x;
	to->vy = to->y - from->y;
	to->id_age = from->id_age;
	to->led_id = from->led_id;
	to->age = from->age + 1;
}

/*
 * Detects blobs in the current frame and compares them with the observation
 * history. The returned blobservation in the `output` variable must be returned
 * to the blobwatch via blobwatch_release_observation()
 */
void
blobwatch_process(blobwatch *bw, struct xrt_frame *frame, uint16_t exposure, uint16_t gain, blobservation **output)
{
	blobservation *ob = POP_QUEUE(&bw->observation_q);
	assert(ob != NULL);

	process_frame(bw, ob, frame);

	/* Optional: dump the actual controller-tracking frame (what the constellation tracker sees) as
	 * PGM — the real short-exposure LED images, which the EuRoC recorder does NOT capture (it taps the
	 * SLAM-exposure stream). Enabled by G2_DUMP_FRAMES=<dir>. Default dumps EVERY frame (needed for a
	 * faithful offline VIO replay); set G2_DUMP_FRAMES_STRIDE=N to dump 1-in-N (e.g. 15 for sparse
	 * inspection, less capture-time I/O). */
	{
		static const char *dump_dir = NULL;
		static bool dump_init = false;
		static uint32_t dump_ctr = 0;
		static uint32_t dump_stride = 1;
		if (!dump_init) {
			dump_dir = getenv("G2_DUMP_FRAMES");
			const char *stride_s = getenv("G2_DUMP_FRAMES_STRIDE");
			if (stride_s != NULL && stride_s[0] != '\0' && atoi(stride_s) > 0) {
				dump_stride = (uint32_t)atoi(stride_s);
			}
			dump_init = true;
		}
		if (dump_dir != NULL && dump_dir[0] != '\0' && (dump_ctr++ % dump_stride) == 0 &&
		    frame->data != NULL && frame->width > 0 && frame->stride >= frame->width) {
			char path[512];
			// Encode cam, frame timestamp (ns, for IMU alignment in offline replay), exposure (the real
			// controller exposure the matcher needs), source seq (groups the 4 cams of one frame), blobs.
			snprintf(path, sizeof(path), "%s/cam%u_t%020lld_e%u_s%010lu_n%u.pgm", dump_dir, bw->cam_id,
			         (long long)frame->timestamp, (unsigned)exposure,
			         (unsigned long)frame->source_sequence, (unsigned)ob->num_blobs);
			FILE *fp = fopen(path, "wb");
			if (fp != NULL) {
				fprintf(fp, "P5\n%u %u\n255\n", frame->width, frame->height);
				for (uint32_t y = 0; y < frame->height; y++) {
					fwrite(frame->data + (size_t)y * frame->stride, 1, frame->width, fp);
				}
				fclose(fp);
			}
		}
	}

	/* Telemetry: one row per processed frame per camera. n_blobs is what we just
	 * produced; frame_seq comes from the source frame. hw_ts_ns uses frame->timestamp,
	 * the monotonic-converted frame time (NOT the raw device-clock source_timestamp),
	 * so it shares the common clock with every other stream's t_mono_ns.
	 * exposure is the authoritative per-frame value read from the camera's pixel header
	 * (plumbed in by the caller); gain is the commanded value (0 if unknown). */
	g2_telem_frame(bw->cam_id, (uint64_t)frame->timestamp, (uint32_t)frame->source_sequence,
	               (uint16_t)ob->num_blobs, exposure, gain, 0);

	/* Return observed blobs */
	if (output) {
		*output = ob;
	}

	/* If there is no previous observation, our work is done here - no need to match
	 * blobs against the prior */
	if (bw->last_observation == NULL) {
		bw->last_observation = ob;
		return;
	}

	blobservation *last_ob = bw->last_observation;
	int closest_ob[MAX_BLOBS_PER_FRAME];             // index of last_ob that is closest to each ob
	int closest_last_ob[MAX_BLOBS_PER_FRAME];        // index of ob that is closest to each last_ob
	int closest_last_ob_distsq[MAX_BLOBS_PER_FRAME]; // distsq of ob that is closest to each last_ob
	int i, j;

	/* Clear closest_* */
	for (i = 0; i < MAX_BLOBS_PER_FRAME; i++) {
		closest_ob[i] = -1;
		closest_last_ob[i] = -1;
		closest_last_ob_distsq[i] = 1000000;
	}

	int scan_again = 1;
	int scan_times = 0;
	while (scan_again) {
		scan_again = 0;

		/* Try to match each blob with the closest blob from the previous frame. */
		for (i = 0; i < ob->num_blobs; i++) {
			if (closest_ob[i] != -1)
				continue; // already has a match

			struct blob *b2 = &ob->blobs[i];
			int closest_j = -1;
			int closest_distsq = -1;

			for (j = 0; j < last_ob->num_blobs; j++) {
				struct blob *b1 = &last_ob->blobs[j];
				int x, y, dx, dy, distsq;

				/* Estimate b1's next position */
				x = b1->x + b1->vx;
				y = b1->y + b1->vy;

				/* Absolute distance */
				dx = abs(x - b2->x);
				dy = abs(y - b2->y);
				distsq = dx * dx + dy * dy;

				if (closest_distsq < 0 || distsq < closest_distsq) {
					if (closest_last_ob[j] != -1 && closest_last_ob_distsq[j] <= distsq) {
						// some blob already claimed this one as closest
						// don't usurp if previous one is closer
						continue;
					}
					closest_j = j;
					closest_distsq = distsq;
				}
			}

			closest_ob[i] = closest_j;

			// didn't find any matching blobs
			if (closest_j < 0)
				continue;

			if (closest_last_ob[closest_j] != -1) {
				// we are usurping some other blob because we are closer
				closest_ob[closest_last_ob[closest_j]] = -1;
				scan_again++;
			}

			closest_last_ob[closest_j] = i;
			closest_last_ob_distsq[closest_j] = closest_distsq;
		}

		if (scan_times++ > 100) {
			U_LOG_W("blob matching looped excessively. scan_times: %d", scan_times);
		}
	}

	/* Copy blobs that found a closest match */
	for (i = 0; i < ob->num_blobs; i++) {
		if (closest_ob[i] < 0)
			continue; // no match

		struct blob *b2 = &ob->blobs[i];
		struct blob *b1 = &last_ob->blobs[closest_ob[i]];

		if (b1->track_index >= 0 && ob->tracked[b1->track_index] == 0) {
			/* Only overwrite tracks that are not already set */
			b2->track_index = b1->track_index;
			ob->tracked[b2->track_index] = i + 1;
		}
		copy_matching_blob(b2, b1);
	}

	/*
	 * Clear the tracking array where blobs have gone missing.
	 */
	for (i = 0; i < MAX_BLOBS_PER_FRAME; i++) {
		int t = ob->tracked[i];
		if (t > 0 && ob->blobs[t - 1].track_index != i)
			ob->tracked[i] = 0;
	}

	/*
	 * Associate newly tracked blobs with a free space in the
	 * tracking array.
	 */
	for (i = 0; i < ob->num_blobs; i++) {
		struct blob *b2 = &ob->blobs[i];

		if (b2->age > 0 && b2->track_index < 0)
			b2->track_index = find_free_track(ob->tracked);
		if (b2->track_index >= 0)
			ob->tracked[b2->track_index] = i + 1;
	}

#if CONSISTENCY_CHECKS
	/* Check blob <-> tracked array links for consistency */
	for (i = 0; i < ob->num_blobs; i++) {
		struct blob *b = &ob->blobs[i];

		if (b->track_index >= 0 && ob->tracked[b->track_index] != i + 1) {
			U_LOG_W("Inconsistency! %d != %d", ob->tracked[b->track_index], i + 1);
		}
	}
#endif

	bw->last_observation = ob;
}

struct blob *
blobwatch_find_blob_at(blobwatch *bw, int x, int y)
{
	blobservation *ob = bw->last_observation;
	int i;

	if (ob == NULL) {
		/* No blobs to match against yet */
		return NULL;
	}

	for (i = 0; i < ob->num_blobs; i++) {
		struct blob *b = ob->blobs + i;
		int dx = abs(x - b->x);
		int dy = abs(y - b->y);

		/* Check if the target is outside the bounding box */
		if (2 * dx > b->width || 2 * dy > b->height)
			continue;
		return b;
	}

	return NULL;
}

void
blobwatch_update_labels(blobwatch *bw, blobservation *ob, uint8_t device_id)
{
	/* Take the observation in ob and replace any labels for the given device in
	 * the most recent observation with labels from this observation */
	/* FIXME: This is an n^2 match for simplicity. Filtering out blobs with no LED
	 * label and sorting the blobs by ID might make things quicker for larger numbers
	 * of blobs - needs testing. */
	blobservation *last_ob = bw->last_observation;
	int i, l;

	if (last_ob == NULL || last_ob == ob) {
		/* No label transfers needed, increment the ID age */
		for (i = 0; i < ob->num_blobs; i++) {
			struct blob *b = ob->blobs + i;
			if (b->led_id != LED_INVALID_ID && b->led_id == b->prev_led_id)
				b->id_age++;
			else
				b->id_age = 0;
		}
		return;
	}

	/* Clear all labels for the indicated device */
	for (l = 0; l < last_ob->num_blobs; l++) {
		struct blob *new_b = last_ob->blobs + l;
		if (LED_OBJECT_ID(new_b->led_id) == device_id) {
			new_b->prev_led_id = new_b->led_id;
			new_b->led_id = LED_INVALID_ID;
		}
	}

	/* Transfer all labels for matching blob ids */
	for (i = 0; i < ob->num_blobs; i++) {
		struct blob *b = ob->blobs + i;
		if (LED_OBJECT_ID(b->led_id) != device_id)
			continue; /* Not for this device */

		for (l = 0; l < last_ob->num_blobs; l++) {
			struct blob *new_b = last_ob->blobs + l;
			if (new_b->blob_id == b->blob_id) {
				if (bw->debug) {
					U_LOG_D("Found matching blob %u - labelled with LED id %x\n", b->blob_id,
					        b->led_id);
				}
				new_b->led_id = b->led_id;

				if (new_b->led_id == new_b->prev_led_id) {
					new_b->id_age++;
				} else {
					new_b->id_age = 0;
				}
			}
		}
	}
}

void
blobwatch_release_observation(blobwatch *bw, blobservation *ob)
{
	PUSH_QUEUE(&bw->observation_q, ob);
}
