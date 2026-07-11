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
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "os/os_threading.h"

#include "math/m_api.h"
#include "util/u_logging.h"
#include "util/u_g2_telemetry.h"

#include "blobwatch.h"

/* AVX2 row kernels for the two per-pixel passes (integral-image accumulation
 * and the dark-pixel scan): pure integer ops computing exactly the same values
 * element-for-element as the scalar loops, so the output is bit-identical to
 * the scalar path by construction. Selected once per process at library load
 * via __builtin_cpu_supports("avx2") — pre-Haswell x86-64 and non-x86 builds
 * run the scalar loops. */
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define BW_HAVE_AVX2 1
#define BW_TARGET_AVX2 __attribute__((target("avx2")))

/* Written exactly once, by the load-time constructor below, before any frame
 * thread can exist; read (unsynchronized, safely) by the row kernels. */
static bool bw_use_avx2 = false;

__attribute__((constructor)) static void
bw_select_row_kernels(void)
{
	bw_use_avx2 = __builtin_cpu_supports("avx2");
}
#else
#define BW_HAVE_AVX2 0
#define BW_TARGET_AVX2
#endif

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
#define BLOB_ADMISSION_NOISE_FLOOR 8
/* Half-size of the local-background box (window = 2*r+1). Large enough to
 * straddle an LED spot so the LED itself barely lifts its own background. */
#define ADAPT_BG_RADIUS 12

/* A pixel >= this is treated as saturated/clipped. 255 is the 8-bit physical ceiling; the 5-count guard
 * band also catches near-clipped pixels whose greysum centroid is already biased. */
#define SATURATION_LEVEL (255 - 5)

/* Shape qualification rejects clutter without throwing away dim LED spots. */
#define MAX_BLOB_WH 12
#define MIN_BLOB_AREA 2
#define MIN_BLOB_ASPECT 0.34f
#define MIN_FILL_FLOOR 0.34f
#define MIN_FILL_BRIGHT 0.45f
#define MIN_PEAK_TO_MEAN 1.30f
#define SHAPE_BRIGHT_REF 40.0f

/* Bounded recovery for dim compact local maxima missed by the scanline gates. */
#define RECOVER_NMS_RADIUS 2
#define RECOVER_MAX_WH 5
#define RECOVER_MIN_AREA 2
#define RECOVER_MAX_AREA 30
#define RECOVER_MARGIN_LO 3.0f
#define RECOVER_MIN_ASPECT 0.55f
#define RECOVER_MIN_FILL 0.55f
#define RECOVER_MIN_PEAK_TO_MEAN 1.15f
#define RECOVER_MIN_SNR 4.0f
#define RECOVER_CENTER_EXCESS 4.0f
#define RECOVER_RING_STD_FLOOR 1.0f
#define RECOVER_DEDUP_R 3.0f
#define RECOVER_RIDGE_RING_MAX 1
#define RECOVER_FRAME_MAX 4

/* Centroid-uncertainty (R) inflation factors. A fully clipped or frame-edge-truncated spot has lost the
 * symmetry information its centre relies on, so its variance is inflated up to these multiples. Derived as
 * worst-case multipliers on the measured spread, not absolute pixel magic numbers: at full saturation the
 * usable signal is the contour alone (~SAT_R_INFLATE_MAX wider), and a contour cut by the frame border is
 * one-sided (~EDGE_R_INFLATE wider). Dim blobs carry the measured brightness-noise inflation below,
 * capped at DIM_R_INFLATE_MAX. All factors are smooth in the blob's own measurements -> no abrupt cull. */
#define SAT_R_INFLATE_MAX 4.0f
#define EDGE_R_INFLATE 3.0f
#define DIM_R_INFLATE_MAX 4.0f

/* Measured centroid-noise law (results/h6-rmodel-20260612, 161k held-out LED<->blob residuals
 * across four captures, DOF-corrected): sigma^2(brightness) = S0 + A/brightness^2 px^2 — the
 * photometric-centroid noise scales as pixel-noise / peak-signal on a near-zero short-exposure
 * background, on top of a brightness-independent floor. R therefore carries the variance ratio
 * g(b) = 1 + (K/b)^2, K = sqrt(A/S0): a pure function of the blob (H5 invariant: no per-frame
 * anchor, no room-content coupling), capped at DIM_R_INFLATE_MAX (the long-validated worst case;
 * binds below b ~ 23, where the fit under-estimates anyway). The law form replicates on every
 * capture and device independently; K is calibrated at the production gain-16 operating point
 * (K = 40.5 / 39.0 per device on the felt-validation capture; the calmer May-era captures give
 * K = 21..27, i.e. the same law with less per-pixel noise). At the gain-16 dim band (b ~ 22..45)
 * this yields x1.8..4.0 — the measured truth, matching the long-validated frame-anchored dim-band
 * operating point without its room-content coupling, and well below the saturation-anchored flat
 * x3.7..4.0 that deflated acceptance nats on dim clutter and bright LED spots alike.
 *
 * K under commanded analog gain (B3, results/b3-signal-design-20260704 §1a + analysis-d §1): the
 * per-pixel noise splits into a pre-gain share (read + dark + photon shot + scene-IR, electron-referred,
 * so it scales with the brightness multiplier m = gain/16 in DN) and a post-gain share (ADC read +
 * quantization >= 1/sqrt(12) DN, fixed in DN — measured dark sigma 0.276 DN ~= the quantizer floor, so
 * this share is real). Centroid noise A ~ sigma_pix^2 and S0 is gain-invariant (bright-blob geometric
 * floor in px^2, constant across captures whose A varies 3x), hence
 *     K(m) = K16 * sqrt(f_pre * m^2 + f_post),  f_pre + f_post = 1  (variance shares at gain 16).
 * f_pre below is analyst D's felt-room point estimate (band 0.5..0.8), expressed binary-exactly so that
 * K(2) = 65.0 (the predicted 60..70 band) and K(1) = K16 exactly — gain-16 (or unknown-gain) replays
 * stay bit-identical. It is a provisional physical constant: the S3 gain-sweep re-fit (dual-model
 * S0 + A1/b + A2/b^2, f_pre backed out as (A'/A - 1)/(m^2 - 1)) replaces it with the measured split. */
#define DIM_NOISE_BRIGHTNESS_K16 40.0f
#define DIM_NOISE_GAIN_F_PRE 0.546875f

/* Cap-pressure retention: all qualifying extents are staged (bound below), and when more than
 * MAX_BLOBS_PER_FRAME qualify the survivors are chosen by priority (device-near class first,
 * then non-static-clutter, by descending local contrast) instead of silently dropping
 * everything past the cap in scanline order — which evicted bottom-of-frame (hand-height)
 * LEDs exactly in the clutter-storm frames that hit the cap. */
#define BLOB_STAGE_MAX 256
/* Radius for transferring the previous frame's retention class onto a staged blob (the same
 * spot, allowing for centroid noise — static clutter does not move in the image between
 * consecutive frames). Matches the static map's image-space match gate. */
#define STAGE_CLASS_MATCH_PX 3.0f

#define QUEUE_ENTRIES (NUM_FRAMES_HISTORY + 1)

#define G2_PGM_DUMP_QUEUE_CAP 64

struct g2_pgm_dump_job
{
	char path[512];
	struct xrt_frame *frame;
};

struct g2_pgm_dump_queue
{
	struct os_thread_helper helper;
	bool initialized;
	bool enabled;
	bool blocking;
	const char *dir;
	uint32_t stride;
	struct g2_pgm_dump_job jobs[G2_PGM_DUMP_QUEUE_CAP];
	uint32_t head;
	uint32_t tail;
	uint32_t count;
	uint64_t dropped;
};

static struct g2_pgm_dump_queue g_pgm_dump = {0};
static pthread_once_t g_pgm_dump_once = PTHREAD_ONCE_INIT;
static atomic_uint g_pgm_dump_stride_ctr = ATOMIC_VAR_INIT(0);

static void
g2_pgm_dump_write_job(struct g2_pgm_dump_job *job)
{
	FILE *fp = fopen(job->path, "wb");
	if (fp != NULL && job->frame != NULL && job->frame->data != NULL) {
		fprintf(fp, "P5\n%u %u\n255\n", job->frame->width, job->frame->height);
		for (uint32_t y = 0; y < job->frame->height; y++) {
			const uint8_t *row = job->frame->data + (size_t)y * job->frame->stride;
			fwrite(row, 1, job->frame->width, fp);
		}
		fclose(fp);
	}
	xrt_frame_reference(&job->frame, NULL);
}

static void *
g2_pgm_dump_thread(void *ptr)
{
	struct g2_pgm_dump_queue *queue = (struct g2_pgm_dump_queue *)ptr;

	for (;;) {
		struct g2_pgm_dump_job job = {0};

		os_thread_helper_lock(&queue->helper);
		while (queue->count == 0 && os_thread_helper_is_running_locked(&queue->helper)) {
			os_thread_helper_wait_locked(&queue->helper);
		}
		if (queue->count == 0 && !os_thread_helper_is_running_locked(&queue->helper)) {
			os_thread_helper_unlock(&queue->helper);
			break;
		}
		job = queue->jobs[queue->head];
		queue->jobs[queue->head] = (struct g2_pgm_dump_job){0};
		queue->head = (queue->head + 1) % G2_PGM_DUMP_QUEUE_CAP;
		queue->count--;
		os_thread_helper_signal_locked(&queue->helper);
		os_thread_helper_unlock(&queue->helper);

		g2_pgm_dump_write_job(&job);
	}

	os_thread_helper_signal_stop(&queue->helper);
	return NULL;
}

static void
g2_pgm_dump_shutdown(void)
{
	if (!g_pgm_dump.initialized) {
		return;
	}
	os_thread_helper_destroy(&g_pgm_dump.helper);
	g_pgm_dump.initialized = false;
}

static void
g2_pgm_dump_init_once(void)
{
	const char *dump_dir = getenv("G2_DUMP_FRAMES");
	if (dump_dir == NULL || dump_dir[0] == '\0') {
		return;
	}

	g_pgm_dump.enabled = true;
	g_pgm_dump.blocking = getenv("G2_DUMP_FRAMES_BLOCKING") != NULL;
	g_pgm_dump.dir = dump_dir;
	g_pgm_dump.stride = 1;
	const char *stride_s = getenv("G2_DUMP_FRAMES_STRIDE");
	if (stride_s != NULL && stride_s[0] != '\0' && atoi(stride_s) > 0) {
		g_pgm_dump.stride = (uint32_t)atoi(stride_s);
	}
	if (os_thread_helper_init(&g_pgm_dump.helper) != 0) {
		g_pgm_dump.enabled = false;
		return;
	}
	g_pgm_dump.initialized = true;
	if (os_thread_helper_start(&g_pgm_dump.helper, g2_pgm_dump_thread, &g_pgm_dump) != 0) {
		os_thread_helper_destroy(&g_pgm_dump.helper);
		g_pgm_dump.initialized = false;
		g_pgm_dump.enabled = false;
		return;
	}
	os_thread_helper_name(&g_pgm_dump.helper, "G2 PGM dump");
	atexit(g2_pgm_dump_shutdown);
}

static bool
g2_pgm_dump_enabled(void)
{
	pthread_once(&g_pgm_dump_once, g2_pgm_dump_init_once);
	return g_pgm_dump.enabled && g_pgm_dump.initialized;
}

static void
g2_pgm_dump_enqueue(const char *path, struct xrt_frame *frame)
{
	if (!g2_pgm_dump_enabled() || frame == NULL || frame->data == NULL || frame->width == 0 ||
	    frame->height == 0 || frame->stride < frame->width) {
		return;
	}

	struct g2_pgm_dump_job job = {0};
	snprintf(job.path, sizeof(job.path), "%s", path);

	os_thread_helper_lock(&g_pgm_dump.helper);
	while (g_pgm_dump.blocking && g_pgm_dump.count == G2_PGM_DUMP_QUEUE_CAP &&
	       os_thread_helper_is_running_locked(&g_pgm_dump.helper)) {
		os_thread_helper_wait_locked(&g_pgm_dump.helper);
	}
	if (g_pgm_dump.count == G2_PGM_DUMP_QUEUE_CAP || !os_thread_helper_is_running_locked(&g_pgm_dump.helper)) {
		g_pgm_dump.dropped++;
		if (g_pgm_dump.dropped == 1 || (g_pgm_dump.dropped % 1000) == 0) {
			g2_telem_event(0, g2_telem_now_ns(), G2_TELEM_EV_FRAME_DUMP_DROPPED,
			               (float)g_pgm_dump.dropped);
			U_LOG_W("G2_DUMP_FRAMES async queue full; dropped %" PRIu64
			        " frame(s). Set G2_DUMP_FRAMES_BLOCKING=1 for lossless capture.",
			        g_pgm_dump.dropped);
		}
		os_thread_helper_unlock(&g_pgm_dump.helper);
		return;
	}
	xrt_frame_reference(&job.frame, frame);
	g_pgm_dump.jobs[g_pgm_dump.tail] = job;
	g_pgm_dump.tail = (g_pgm_dump.tail + 1) % G2_PGM_DUMP_QUEUE_CAP;
	g_pgm_dump.count++;
	os_thread_helper_signal_locked(&g_pgm_dump.helper);
	os_thread_helper_unlock(&g_pgm_dump.helper);
}

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
	uint8_t pixel_threshold; /* Minimum pixel magnitude considered non-black */
	uint8_t adapt_margin;    /* Minimum contrast above the local adaptive background */
	uint8_t cam_id;          /* Camera index this blobwatch processes (for telemetry) */
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

	/* Per-frame staging for cap-pressure priority retention (single producer; reset by
	 * process_frame_roi, consumed by finalize_staged_blobs before any other ob consumer). */
	struct blob stage[BLOB_STAGE_MAX];
	int num_staged;
	/* Cached priority class (0..2) of each staged blob, filled lazily on the first staging
	 * overflow of a frame so the overflow eviction need not re-derive the retention class per
	 * candidate. Reset with num_staged each frame. */
	uint8_t stage_priority[BLOB_STAGE_MAX];
	bool stage_priority_cached;
};

/*
 * Allocates and initializes blobwatch structure.
 *
 * Returns the newly allocated blobwatch structure.
 */
blobwatch *
blobwatch_new(uint8_t pixel_threshold, uint8_t cam_id)
{
	blobwatch *bw = malloc(sizeof(*bw));
	int i;

	if (!bw)
		return NULL;

	{
		static bool kernels_logged = false;
		if (!kernels_logged) {
			kernels_logged = true;
#if BW_HAVE_AVX2
			U_LOG_I("blobwatch: row kernels: %s", bw_use_avx2 ? "AVX2 (runtime-selected)" : "scalar");
#else
			U_LOG_I("blobwatch: row kernels: scalar");
#endif
		}
	}

	memset(bw, 0, sizeof(*bw));
	bw->next_blob_id = 1;
	bw->cam_id = cam_id;

	/* Minimum pixel magnitude to be included in a blob at all */
	bw->pixel_threshold = pixel_threshold;
	bw->adapt_margin = ADAPT_MARGIN;

	bw->blob_max_wh = MAX_BLOB_WH;
	bw->blob_min_aspect = MIN_BLOB_ASPECT;

	bw->last_observation = NULL;
	bw->debug = false;

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
 * First index in [x, x_max) whose pixel value exceeds @p threshold, or x_max.
 * The pixels skipped over have no side effects in any caller, so vectorizing
 * this scan cannot change any output.
 */
#if BW_HAVE_AVX2
BW_TARGET_AVX2 static uint32_t
next_pixel_above_avx2(const uint8_t *line, uint32_t x, uint32_t x_max, uint8_t threshold)
{
	const __m256i thr = _mm256_set1_epi8((char)threshold);
	const __m256i zero = _mm256_setzero_si256();
	while (x + 32 <= x_max) {
		const __m256i v = _mm256_loadu_si256((const __m256i *)(line + x));
		/* No unsigned byte compare in AVX2: v > thr  <=>  saturating v - thr != 0. */
		const __m256i le = _mm256_cmpeq_epi8(_mm256_subs_epu8(v, thr), zero);
		const uint32_t above = ~(uint32_t)_mm256_movemask_epi8(le);
		if (above != 0) {
			return x + (uint32_t)__builtin_ctz(above);
		}
		x += 32;
	}
	while (x < x_max && line[x] <= threshold) {
		x++;
	}
	return x;
}
#endif

static uint32_t
next_pixel_above(const uint8_t *line, uint32_t x, uint32_t x_max, uint8_t threshold)
{
#if BW_HAVE_AVX2
	if (bw_use_avx2) {
		return next_pixel_above_avx2(line, x, x_max, threshold);
	}
#endif
	while (x < x_max && line[x] <= threshold) {
		x++;
	}
	return x;
}

/*
 * One row of the integral image: row[x+1] = prev[x+1] + sum(src[0..x]).
 * uint32 addition is associative, so the in-register prefix scan produces
 * exactly the scalar running sum in every element.
 */
#if BW_HAVE_AVX2
BW_TARGET_AVX2 static void
integral_row_avx2(const uint8_t *src, const uint32_t *prev, uint32_t *row, int w)
{
	uint32_t run = 0;
	int x = 0;
	row[0] = 0; /* left border column */
	const __m128i zero128 = _mm_setzero_si128();
	__m256i vrun = _mm256_setzero_si256(); /* running total, broadcast in all 8 lanes */
	for (; x + 16 <= w; x += 16) {
		const __m128i b = _mm_loadu_si128((const __m128i *)(src + x));
		__m256i v0 = _mm256_cvtepu8_epi32(b);
		__m256i v1 = _mm256_cvtepu8_epi32(_mm_srli_si128(b, 8));
		/* psadbw gives the exact byte sums of each 8-byte half; the
		 * loop-carried total advances through this short chain, kept in
		 * the vector domain, independent of the two prefix scans below. */
		const __m128i sad = _mm_sad_epu8(b, zero128); /* u64[0]=sum(low 8), u64[1]=sum(high 8) */
		const __m128i tot16 = _mm_add_epi64(sad, _mm_srli_si128(sad, 8));
		/* Inclusive prefix scan of 8 lanes: two shift-add steps within each
		 * 128-bit half, then add the low half's total into every high-half lane. */
		v0 = _mm256_add_epi32(v0, _mm256_slli_si256(v0, 4));
		v1 = _mm256_add_epi32(v1, _mm256_slli_si256(v1, 4));
		v0 = _mm256_add_epi32(v0, _mm256_slli_si256(v0, 8));
		v1 = _mm256_add_epi32(v1, _mm256_slli_si256(v1, 8));
		const __m256i t0 = _mm256_shuffle_epi32(v0, _MM_SHUFFLE(3, 3, 3, 3));
		const __m256i t1 = _mm256_shuffle_epi32(v1, _MM_SHUFFLE(3, 3, 3, 3));
		v0 = _mm256_add_epi32(v0, _mm256_permute2x128_si256(t0, t0, 0x08)); /* low half = 0 */
		v1 = _mm256_add_epi32(v1, _mm256_permute2x128_si256(t1, t1, 0x08));
		/* v1 additionally carries v0's full total (sad u32[0] = sum of the low 8 bytes). */
		v1 = _mm256_add_epi32(v1, _mm256_broadcastd_epi32(sad));
		v0 = _mm256_add_epi32(v0, vrun);
		v1 = _mm256_add_epi32(v1, vrun);
		vrun = _mm256_add_epi32(vrun, _mm256_broadcastd_epi32(tot16));
		_mm256_storeu_si256((__m256i *)(row + x + 1),
		                    _mm256_add_epi32(v0, _mm256_loadu_si256((const __m256i *)(prev + x + 1))));
		_mm256_storeu_si256((__m256i *)(row + x + 9),
		                    _mm256_add_epi32(v1, _mm256_loadu_si256((const __m256i *)(prev + x + 9))));
	}
	run = (uint32_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(vrun));
	for (; x < w; x++) {
		run += src[x];
		row[x + 1] = prev[x + 1] + run;
	}
}
#endif

static void
integral_row(const uint8_t *src, const uint32_t *prev, uint32_t *row, int w)
{
#if BW_HAVE_AVX2
	if (bw_use_avx2) {
		integral_row_avx2(src, prev, row, w);
		return;
	}
#endif
	uint32_t run = 0;
	row[0] = 0; /* left border column */
	for (int x = 0; x < w; x++) {
		run += src[x];
		row[x + 1] = prev[x + 1] + run;
	}
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
		integral_row(src, prev, row, w);
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
 * Parabolic (3-point) sub-pixel peak refinement on one axis: fit through the
 * neighbour/peak/neighbour values (@p lo, @p c, @p hi) around the integer peak
 * coordinate @p peak. Writes the refined coordinate to @p out only for a
 * concave fit (a true peak) whose shift stays within +/-0.5 px.
 */
static inline void
parabolic_subpixel_refine(float lo, float c, float hi, int peak, float *out)
{
	const float den = lo - 2.0f * c + hi;
	if (den < 0.0f) {
		const float s = 0.5f * (lo - hi) / den;
		if (s > -0.5f && s < 0.5f) {
			*out = peak + s;
		}
	}
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
 * truncation and the measured brightness-noise law.
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

		parabolic_subpixel_refine(lx, c, rx, peak_x, led_x);
		parabolic_subpixel_refine(ux, c, dx, peak_y, led_y);
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
           uint8_t brightness,
           float contrast)
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
	b->contrast = contrast;
	b->retention_class = BLOB_RETENTION_FRESH;
	b->static_dwell_s = 0.0f;
}

/* The retention class of a spot, transferred from the PREVIOUS frame by same-spot proximity (the
 * tracker's static map writes retention_class after extraction, so the current frame's is not yet
 * known here). One frame stale is fine for budget ordering; a miss returns FRESH (never demoted).
 * This is the class both cap stages -- staging-overflow eviction and the final MAX_BLOBS_PER_FRAME
 * cap -- rank by, so they can never disagree. Matches the static map's image-space match gate. */
static uint8_t
stale_retention_class(const blobservation *last_ob, float x, float y)
{
	if (last_ob == NULL) {
		return BLOB_RETENTION_FRESH;
	}
	float best_distsq = STAGE_CLASS_MATCH_PX * STAGE_CLASS_MATCH_PX;
	uint8_t cls = BLOB_RETENTION_FRESH;
	for (int i = 0; i < last_ob->num_blobs; i++) {
		const struct blob *b = &last_ob->blobs[i];
		const float dx = (b->x + b->vx) - x;
		const float dy = (b->y + b->vy) - y;
		const float distsq = dx * dx + dy * dy;
		if (distsq <= best_distsq) {
			best_distsq = distsq;
			cls = b->retention_class;
		}
	}
	return cls;
}

/* Priority class for cap retention: DEVICE_NEAR (0) outranks non-static clutter (1) outranks static
 * clutter (2). Lower is kept first. */
static inline uint8_t
stage_priority_class(const blobwatch *bw, float x, float y)
{
	const uint8_t cls = stale_retention_class(bw->last_observation, x, y);
	return cls == BLOB_RETENTION_DEVICE_NEAR ? 0 : cls != BLOB_RETENTION_STATIC_CLUTTER ? 1 : 2;
}

/* Strict (class, contrast) priority order: higher priority = lower class rank, ties broken by higher
 * local contrast. The one ordering primitive shared by the final-cap sort (stage_rank_cmp) and the
 * staging-overflow eviction, so scanline order can never evict a blob the ranked cap would keep. */
static inline bool
stage_key_higher_priority(uint8_t class_a, float contrast_a, uint8_t class_b, float contrast_b)
{
	if (class_a != class_b) {
		return class_a < class_b;
	}
	return contrast_a > contrast_b;
}

static void
extent_to_blobs(blobwatch *bw, blobservation *ob, struct extent *e, int y, struct xrt_frame *frame)
{
	const uint8_t admission_threshold = bw->pixel_threshold > BLOB_ADMISSION_NOISE_FLOOR
	                                        ? bw->pixel_threshold
	                                        : BLOB_ADMISSION_NOISE_FLOOR;
	if (e->max_pixel <= admission_threshold) {
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

	if (e->area < MIN_BLOB_AREA) {
		ob->dropped_shape_blobs++;
		return;
	}

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
	float t = ((float)e->max_pixel - (float)bw->pixel_threshold) / (SHAPE_BRIGHT_REF - (float)bw->pixel_threshold);
	if (t < 0.0f) {
		t = 0.0f;
	}
	if (t > 1.0f) {
		t = 1.0f;
	}
	const float fill_gate = MIN_FILL_FLOOR + (MIN_FILL_BRIGHT - MIN_FILL_FLOOR) * t;
	const float p2m_gate = 1.0f + (MIN_PEAK_TO_MEAN - 1.0f) * t;
	if (fill_ratio < fill_gate || peak_to_mean < p2m_gate) {
		ob->dropped_shape_blobs++; // a shape/quality reject, not a brightness (dark) one
		return;
	}

	/* In the future we could generate multiple blobs from one extent if we detect
	 * it as multiple LEDs */
	{
		float led_x, led_y, pos_var_px2;

		compute_greysum(bw, frame, e, y, saturated, &led_x, &led_y, &pos_var_px2);

		/* Inflate R for centroid information lost to clipping and to frame-border truncation. The
		 * saturated fraction (clipped pixels / area) interpolates toward SAT_R_INFLATE_MAX; touching
		 * the frame border multiplies by EDGE_R_INFLATE (its skirt/contour is one-sided). The
		 * brightness-noise term is applied in process_frame_roi over the final blob set. */
		const float sat_frac = (float)e->sat_count / (float)e->area;
		float infl = 1.0f + sat_frac * (SAT_R_INFLATE_MAX - 1.0f);
		const bool on_edge = e->left == 0 || e->top == 0 || (int)e->right == (int)frame->width - 1 ||
		                     y == (int)frame->height - 1;
		if (on_edge)
			infl *= EDGE_R_INFLATE;
		pos_var_px2 *= infl;

		const float contrast =
		    (float)e->max_pixel -
		    (float)local_bg_mean(bw, (int)led_x, (int)led_y, (int)frame->width, (int)frame->height);

		if (bw->num_staged < BLOB_STAGE_MAX) {
			store_blob(e, bw->num_staged++, y, bw->stage, bw->next_blob_id++, led_x, led_y,
			           pos_var_px2, e->max_pixel, contrast);
			return;
		}

		/* Staging is full. Do NOT drop this extent just because it completed late in scanline
		 * order -- a bottom-of-frame (hand-height) LED ring can complete after a top-of-frame
		 * clutter storm has already claimed all BLOB_STAGE_MAX slots. Keep the highest-priority
		 * BLOB_STAGE_MAX blobs: evict the lowest-priority staged blob when this one strictly
		 * outranks it, by the same (device-near, contrast) key the final cap ranks by. Below
		 * BLOB_STAGE_MAX this branch never runs, so normal frames stage in byte-identical
		 * scanline order. */
		ob->dropped_capacity++;
		if (!bw->stage_priority_cached) {
			for (int i = 0; i < BLOB_STAGE_MAX; i++) {
				bw->stage_priority[i] =
				    stage_priority_class(bw, bw->stage[i].x, bw->stage[i].y);
			}
			bw->stage_priority_cached = true;
		}
		int victim = 0;
		for (int i = 1; i < BLOB_STAGE_MAX; i++) {
			if (stage_key_higher_priority(bw->stage_priority[victim], bw->stage[victim].contrast,
			                              bw->stage_priority[i], bw->stage[i].contrast)) {
				victim = i;
			}
		}
		const uint8_t new_priority = stage_priority_class(bw, led_x, led_y);
		if (stage_key_higher_priority(new_priority, contrast, bw->stage_priority[victim],
		                              bw->stage[victim].contrast)) {
			store_blob(e, victim, y, bw->stage, bw->next_blob_id++, led_x, led_y, pos_var_px2,
			           e->max_pixel, contrast);
			bw->stage_priority[victim] = new_priority;
		}
	}
}

/* Priority retention at the frame blob cap: keep the highest-priority MAX_BLOBS_PER_FRAME staged
 * blobs (device-near class first, then non-static clutter, then descending contrast; staging order
 * breaks ties) instead of the old silent scanline-order truncation. Survivors keep staging order
 * so under-cap frames are byte-identical. */
struct stage_rank
{
	uint8_t class_rank;
	float contrast;
	int idx;
};

static int
stage_rank_cmp(const void *va, const void *vb)
{
	const struct stage_rank *a = va, *b = vb;
	if (a->class_rank != b->class_rank || a->contrast != b->contrast) {
		return stage_key_higher_priority(a->class_rank, a->contrast, b->class_rank, b->contrast) ? -1 : 1;
	}
	return a->idx < b->idx ? -1 : 1;
}

static void
finalize_staged_blobs(blobwatch *bw, blobservation *ob)
{
	if (bw->num_staged <= MAX_BLOBS_PER_FRAME) {
		memcpy(ob->blobs, bw->stage, (size_t)bw->num_staged * sizeof(struct blob));
		ob->num_blobs = bw->num_staged;
		return;
	}

	struct stage_rank rank[BLOB_STAGE_MAX];
	for (int i = 0; i < bw->num_staged; i++) {
		rank[i].class_rank = stage_priority_class(bw, bw->stage[i].x, bw->stage[i].y);
		rank[i].contrast = bw->stage[i].contrast;
		rank[i].idx = i;
	}
	qsort(rank, (size_t)bw->num_staged, sizeof(rank[0]), stage_rank_cmp);

	bool keep[BLOB_STAGE_MAX] = {false};
	for (int k = 0; k < MAX_BLOBS_PER_FRAME; k++) {
		keep[rank[k].idx] = true;
	}
	ob->num_blobs = 0;
	for (int i = 0; i < bw->num_staged; i++) {
		if (keep[i]) {
			ob->blobs[ob->num_blobs++] = bw->stage[i];
		}
	}
	ob->dropped_capacity += bw->num_staged - MAX_BLOBS_PER_FRAME;
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
                 blobservation *ob,
                 uint32_t x_min,
                 uint32_t x_max)
{
	struct extent *le_end = prev_el ? prev_el->extents : NULL;
	struct extent *le = prev_el ? prev_el->extents : NULL;
	struct extent *extent = el->extents;
	int num_extents = MAX_EXTENTS_PER_LINE;
	float center;
	uint32_t x;
	int e = 0;

	if (prev_el)
		le_end += prev_el->num;

	for (x = x_min; x < x_max; x++) {
		int start, end;
		bool is_new_extent = true;
		uint8_t max_pixel = 0;
		uint32_t intensity_sum = 0;
		uint32_t sat_count = 0;

		/* Adaptive threshold: a pixel must clear the global floor (SIMD
		 * skip-scan over the dark majority) AND stand out from its local
		 * background. Cheap (one integral-image lookup), separates dim LEDs
		 * from bright reflections and stops a uniformly bright background
		 * from forming blobs. */
		x = next_pixel_above(line, x, x_max, bw->pixel_threshold);
		if (x >= x_max)
			break;
		if (line[x] < local_bg_mean(bw, x, y, frame->width, frame->height) + bw->adapt_margin)
			continue;

		start = x;
		max_pixel = line[x];
		intensity_sum += line[x];
		if (line[x] >= SATURATION_LEVEL)
			sat_count++;
		x++;

		/* Loop until pixel value falls below threshold (bounded by the ROI's x_max). */
		while (x < x_max && line[x] > bw->pixel_threshold &&
		       line[x] >= local_bg_mean(bw, x, y, frame->width, frame->height) + bw->adapt_margin) {
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
				extent->left = MIN(extent->start, le->left);
				extent->right = MAX(extent->end, le->right);
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

static void
copy_matching_blob(struct blob *to, struct blob *from)
{
	to->blob_id = from->blob_id;
	to->vx = to->x - from->x;
	to->vy = to->y - from->y;
	to->id_age = from->id_age;
	to->led_id = from->led_id;
	to->age = from->age + 1;
	/* One-frame-stale retention evidence; the tracker's static map rewrites it post-extraction. */
	to->retention_class = from->retention_class;
	to->static_dwell_s = from->static_dwell_s;
}

static void
recover_dim_blobs(blobwatch *bw,
                  blobservation *ob,
                  struct xrt_frame *frame,
                  uint32_t roi_x,
                  uint32_t roi_y,
                  uint32_t roi_x_end,
                  uint32_t roi_y_end)
{
	const int w = (int)frame->width, h = (int)frame->height;
	const int nms = RECOVER_NMS_RADIUS;
	const int x_lo = (int)roi_x < 2 ? 2 : (int)roi_x;
	const int y_lo = (int)roi_y < 2 ? 2 : (int)roi_y;
	const int x_hi = (int)roi_x_end > w - 2 ? w - 2 : (int)roi_x_end;
	const int y_hi = (int)roi_y_end > h - 2 ? h - 2 : (int)roi_y_end;
	const int num_blobs_before = ob->num_blobs;

	for (int y = y_lo; y < y_hi; y++) {
		const uint8_t *row = frame->data + (size_t)frame->stride * y;
		for (int x = x_lo; x < x_hi; x++) {
			/* SIMD skip-scan to the next candidate; checking the frame cap only
			 * at candidates is equivalent, as dark pixels never add blobs. */
			x = (int)next_pixel_above(row, (uint32_t)x, (uint32_t)x_hi, bw->pixel_threshold);
			if (x >= x_hi) {
				break;
			}
			if (ob->num_blobs >= MAX_BLOBS_PER_FRAME) {
				goto done;
			}

			const int peak = row[x];
			const double contrast = (double)peak - local_bg_mean(bw, x, y, w, h);
			if (contrast < RECOVER_MARGIN_LO || contrast > (double)bw->adapt_margin) {
				continue;
			}

			bool is_peak = true;
			for (int dy = -nms; dy <= nms && is_peak; dy++) {
				const uint8_t *r2 = frame->data + (size_t)frame->stride * (y + dy);
				for (int dx = -nms; dx <= nms; dx++) {
					if (dx == 0 && dy == 0) {
						continue;
					}
					const int xx = x + dx;
					const int yy = y + dy;
					const int v = r2[xx];
					const double other_contrast = (double)v - local_bg_mean(bw, xx, yy, w, h);
					if (other_contrast > contrast ||
					    (other_contrast == contrast && (dy < 0 || (dy == 0 && dx < 0)))) {
						is_peak = false;
						break;
					}
				}
			}
			if (!is_peak) {
				continue;
			}

			bool covered = false;
			for (int i = 0; i < ob->num_blobs && !covered; i++) {
				const float ddx = ob->blobs[i].x - (float)x;
				const float ddy = ob->blobs[i].y - (float)y;
				covered = ddx * ddx + ddy * ddy <= RECOVER_DEDUP_R * RECOVER_DEDUP_R;
			}
			if (covered) {
				continue;
			}

			double ring_sum = 0.0, ring_sqsum = 0.0;
			int ring_n = 0;
			for (int dy = -2; dy <= 2; dy++) {
				const uint8_t *r2 = frame->data + (size_t)frame->stride * (y + dy);
				for (int dx = -2; dx <= 2; dx++) {
					if (dx >= -1 && dx <= 1 && dy >= -1 && dy <= 1) {
						continue;
					}
					const double v = r2[x + dx];
					ring_sum += v;
					ring_sqsum += v * v;
					ring_n++;
				}
			}
			const double ring_mean = ring_sum / ring_n;
			double ring_var = ring_sqsum / ring_n - ring_mean * ring_mean;
			if (ring_var < 0.0) {
				ring_var = 0.0;
			}
			const double ring_std = sqrt(ring_var);
			const double center_above_ring = (double)peak - ring_mean;
			if (center_above_ring < RECOVER_CENTER_EXCESS) {
				continue;
			}
			const double denom = ring_std > RECOVER_RING_STD_FLOOR ? ring_std : RECOVER_RING_STD_FLOOR;
			if (center_above_ring / denom < RECOVER_MIN_SNR) {
				continue;
			}

			const int half = RECOVER_MAX_WH;
			const int wx0 = x - half < 0 ? 0 : x - half;
			const int wy0 = y - half < 0 ? 0 : y - half;
			const int wx1 = x + half >= w ? w - 1 : x + half;
			const int wy1 = y + half >= h ? h - 1 : y + half;
			const int ww = wx1 - wx0 + 1, wh = wy1 - wy0 + 1;
			bool seen[(2 * RECOVER_MAX_WH + 1) * (2 * RECOVER_MAX_WH + 1)] = {false};
			int stack[(2 * RECOVER_MAX_WH + 1) * (2 * RECOVER_MAX_WH + 1)];
			int sp = 0;
			int area = 0, bb_l = x, bb_r = x, bb_t = y, bb_b = y;
			double wsum = 0.0, wx = 0.0, wy = 0.0, contrast_sum = 0.0;

			stack[sp++] = (y - wy0) * ww + (x - wx0);
			seen[(y - wy0) * ww + (x - wx0)] = true;
			while (sp > 0) {
				const int idx = stack[--sp];
				const int lyy = idx / ww, lxx = idx % ww;
				const int yy = wy0 + lyy, xx = wx0 + lxx;
				const int v = frame->data[(size_t)frame->stride * yy + xx];
				area++;
				if (xx < bb_l) {
					bb_l = xx;
				}
				if (xx > bb_r) {
					bb_r = xx;
				}
				if (yy < bb_t) {
					bb_t = yy;
				}
				if (yy > bb_b) {
					bb_b = yy;
				}
				const int b0 = (int)local_bg_mean(bw, xx, yy, w, h);
				const double wgt = (double)(v - b0);
				wsum += wgt;
				wx += wgt * xx;
				wy += wgt * yy;
				contrast_sum += wgt;

				for (int dy = -1; dy <= 1; dy++) {
					for (int dx = -1; dx <= 1; dx++) {
						if (dx == 0 && dy == 0) {
							continue;
						}
						const int nlx = lxx + dx, nly = lyy + dy;
						if (nlx < 0 || nly < 0 || nlx >= ww || nly >= wh) {
							continue;
						}
						const int nidx = nly * ww + nlx;
						if (seen[nidx]) {
							continue;
						}
						const int nyy = wy0 + nly, nxx = wx0 + nlx;
						const int nv = frame->data[(size_t)frame->stride * nyy + nxx];
						seen[nidx] = true;
						if (nv <= bw->pixel_threshold) {
							continue;
						}
						const double neighbor_contrast =
						    (double)nv - local_bg_mean(bw, nxx, nyy, w, h);
						if (neighbor_contrast >= RECOVER_MARGIN_LO) {
							stack[sp++] = nidx;
						}
					}
				}
			}
			if (area < RECOVER_MIN_AREA || area > RECOVER_MAX_AREA) {
				continue;
			}
			const int blob_w = bb_r - bb_l + 1;
			const int blob_h = bb_b - bb_t + 1;
			if (blob_w > RECOVER_MAX_WH || blob_h > RECOVER_MAX_WH) {
				continue;
			}
			const float aspect = (float)MIN(blob_w, blob_h) / (float)MAX(blob_w, blob_h);
			if (aspect < RECOVER_MIN_ASPECT) {
				continue;
			}
			const float fill = (float)area / (float)(blob_w * blob_h);
			if (fill < RECOVER_MIN_FILL) {
				continue;
			}
			const double mean_contrast = contrast_sum / (double)area;
			if (mean_contrast <= 0.0 || contrast / mean_contrast < RECOVER_MIN_PEAK_TO_MEAN) {
				continue;
			}

			static const int ring_dx[16] = {3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3};
			static const int ring_dy[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
			int ring_bright = 0;
			for (int k = 0; k < 16; k++) {
				const int rx = x + ring_dx[k], ry = y + ring_dy[k];
				if (rx < 0 || ry < 0 || rx >= w || ry >= h) {
					continue;
				}
				const int rv = frame->data[(size_t)frame->stride * ry + rx];
				if (rv - (int)local_bg_mean(bw, rx, ry, w, h) >= (int)bw->adapt_margin) {
					ring_bright++;
				}
			}
			if (ring_bright >= RECOVER_RIDGE_RING_MAX) {
				continue;
			}

			float led_x = wsum > 0.0 ? (float)(wx / wsum) : (float)x;
			float led_y = wsum > 0.0 ? (float)(wy / wsum) : (float)y;
			const float c = (float)peak;
			const float lx = row[x - 1], rx = row[x + 1];
			const float ux = frame->data[(size_t)frame->stride * (y - 1) + x];
			const float dx = frame->data[(size_t)frame->stride * (y + 1) + x];
			parabolic_subpixel_refine(lx, c, rx, x, &led_x);
			parabolic_subpixel_refine(ux, c, dx, y, &led_y);

			const float spread = 0.25f * (float)((bb_r - bb_l + 1) + (bb_b - bb_t + 1));
			float pos_var_px2 = spread * spread;
			if (pos_var_px2 < 0.25f) {
				pos_var_px2 = 0.25f;
			}
			pos_var_px2 *= DIM_R_INFLATE_MAX;

			struct blob *b = &ob->blobs[ob->num_blobs];
			b->blob_id = bw->next_blob_id++;
			b->x = led_x;
			b->y = led_y;
			b->pos_var_px2 = pos_var_px2;
			b->vx = 0;
			b->vy = 0;
			b->left = (uint16_t)bb_l;
			b->top = (uint16_t)bb_t;
			b->width = (uint16_t)(bb_r - bb_l + 1);
			b->height = (uint16_t)(bb_b - bb_t + 1);
			b->area = (uint32_t)area;
			b->age = 0;
			b->track_index = -1;
			b->id_age = 0;
			b->prev_led_id = b->led_id = LED_INVALID_ID;
			b->brightness = (uint8_t)peak;
			b->contrast = (float)contrast;
			b->retention_class = BLOB_RETENTION_FRESH;
			b->static_dwell_s = 0.0f;
			ob->num_blobs++;
		}
	}

done:
	if (ob->num_blobs - num_blobs_before > RECOVER_FRAME_MAX) {
		ob->num_blobs = num_blobs_before;
	}
}

/*
 * Predictive-ROI variant of process_frame: scans only rows in [roi_y, roi_y_end) and clips each row
 * scan to columns [roi_x, roi_x_end). Convergent with Oasis driver's PredictiveROI: blob search is
 * restricted to a bounding box around the ESKF-predicted LED positions (PredictivePatchSize/2 = 8 px
 * per side per LED, expanded here to a single bounding rect over all predictions).
 *
 * The integral image for the adaptive threshold is built over the full frame (the local-background
 * lookup is correct for any pixel inside the ROI); the cost saving is from not scanning rows or
 * columns outside the ROI. The brightness-noise inflation is applied only over the blobs found.
 */
static void
process_frame_roi(blobwatch *bw,
                  blobservation *ob,
                  struct xrt_frame *frame,
                  uint16_t gain,
                  uint32_t roi_x,
                  uint32_t roi_y,
                  uint32_t roi_x_end,
                  uint32_t roi_y_end)
{
	struct extent_line el1;
	struct extent_line el2;

	ob->num_blobs = 0;
	ob->dropped_dark_blobs = 0;
	ob->dropped_shape_blobs = 0;
	ob->dropped_capacity = 0;
	bw->num_staged = 0;
	bw->stage_priority_cached = false;

	build_integral(bw, frame);

	/* Buffer alternation: the original `y & 1 ? &el2 : &el1` pattern works only when roi_y=0 (so the
	 * first loop iter at y=1 reads from el1 which the prologue just filled). For arbitrary roi_y the
	 * parity flips and the loop reads UNINITIALIZED stack memory -> segfault. Explicit prev/cur swap
	 * is correct regardless of roi_y. */
	uint8_t *line = frame->data + (size_t)roi_y * frame->stride;
	struct extent_line *prev = &el1, *cur = &el2;
	process_scanline(line, bw, roi_y, prev, NULL, frame, ob, roi_x, roi_x_end);
	line += frame->stride;

	for (uint32_t y = roi_y + 1; y < roi_y_end; y++) {
		process_scanline(line, bw, y, cur, prev, frame, ob, roi_x, roi_x_end);
		line += frame->stride;
		struct extent_line *tmp = prev;
		prev = cur;
		cur = tmp;
	}

	/* Finalize extents at the last ROI row. process_scanline's internal "if (y == frame->height-1)"
	 * fires only when ROI covers to the bottom of the frame; for a partial ROI the last row's extents
	 * would otherwise be lost. Skip when ROI extends to frame->height (the internal check did fire). */
	if (roi_y_end < frame->height) {
		const uint32_t last_y = roi_y_end - 1;
		for (struct extent *e_iter = prev->extents; e_iter < prev->extents + prev->num; e_iter++) {
			extent_to_blobs(bw, ob, e_iter, (int)last_y, frame);
		}
	}

	finalize_staged_blobs(bw, ob);

	const bool is_full_frame = roi_x == 0 && roi_y == 0 && roi_x_end == frame->width && roi_y_end == frame->height;
	if (!is_full_frame) {
		recover_dim_blobs(bw, ob, frame, roi_x, roi_y, roi_x_end, roi_y_end);
	}

	/* Brightness-honest R inflation: the MEASURED centroid-noise variance ratio
	 * g(b) = 1 + (K/b)^2 (see DIM_NOISE_BRIGHTNESS_K16; K follows the commanded gain via
	 * blobwatch_dim_noise_k). Never anchored at the frame's brightest
	 * blob — the brightest blob is room clutter (a window/lamp) in ~1/5 of cluttered frames,
	 * which inflated every LED's R by x3.5 median and coupled the association margins to room
	 * content. A pure function of the blob's own peak: brightness is the peak signal the
	 * photometric centroid is estimated from, so its noise contribution falls as 1/b^2 onto the
	 * brightness-independent floor (g -> 1 for bright blobs). The cap keeps the worst-case dim
	 * blob at the long-validated DIM_R_INFLATE_MAX operating point. */
	const float dim_noise_k = blobwatch_dim_noise_k(gain);
	for (int i = 0; i < ob->num_blobs; i++) {
		struct blob *b = &ob->blobs[i];
		const float k = dim_noise_k / (float)b->brightness;
		float g = 1.0f + k * k;
		if (g > DIM_R_INFLATE_MAX)
			g = DIM_R_INFLATE_MAX;
		b->pos_var_px2 *= g;
	}
}

float
blobwatch_dim_noise_k(uint16_t gain)
{
	const float m = blobwatch_gain_multiplier(gain);
	return DIM_NOISE_BRIGHTNESS_K16 * sqrtf(DIM_NOISE_GAIN_F_PRE * m * m + (1.0f - DIM_NOISE_GAIN_F_PRE));
}

/*
 * Detects blobs in the current frame and compares them with the observation
 * history. The returned blobservation in the `output` variable must be returned
 * to the blobwatch via blobwatch_release_observation().
 *
 * The full-frame variant is implemented as a 0-area-clamped delegation to
 * blobwatch_process_roi, so the dump / telemetry / history-match post-processing
 * lives in one place and the predictive-ROI path goes through the same pipeline.
 */
void
blobwatch_process(blobwatch *bw, struct xrt_frame *frame, uint16_t exposure, uint16_t gain, blobservation **output)
{
	blobwatch_process_roi(bw, frame, exposure, gain, 0, 0, (int)frame->width, (int)frame->height, output);
}

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
                                blobservation **output)
{
	const uint8_t saved_pixel = bw->pixel_threshold;
	const uint8_t saved_adapt = bw->adapt_margin;

	if (roi_pixel_threshold != 0 && roi_pixel_threshold < saved_pixel) {
		bw->pixel_threshold = roi_pixel_threshold;
	}
	if (roi_adapt_margin != 0 && roi_adapt_margin < saved_adapt) {
		bw->adapt_margin = roi_adapt_margin;
	}

	blobwatch_process_roi(bw, frame, exposure, gain, roi_x, roi_y, roi_w, roi_h, output);

	bw->pixel_threshold = saved_pixel;
	bw->adapt_margin = saved_adapt;
}

void
blobwatch_process_roi(blobwatch *bw,
                      struct xrt_frame *frame,
                      uint16_t exposure,
                      uint16_t gain,
                      int roi_x,
                      int roi_y,
                      int roi_w,
                      int roi_h,
                      blobservation **output)
{
	blobservation *ob = POP_QUEUE(&bw->observation_q);
	assert(ob != NULL);

	/* Clamp the ROI to the frame; a 0-area or fully-covering rect degrades to the full-frame path. */
	uint32_t rx = roi_x < 0 ? 0 : (uint32_t)roi_x;
	uint32_t ry = roi_y < 0 ? 0 : (uint32_t)roi_y;
	uint32_t rx_end = (uint32_t)(roi_x + roi_w);
	uint32_t ry_end = (uint32_t)(roi_y + roi_h);
	if (rx_end > frame->width) rx_end = frame->width;
	if (ry_end > frame->height) ry_end = frame->height;
	if (rx >= rx_end || ry >= ry_end) {
		/* Caller passed a degenerate ROI -- fall back to full frame so the search isn't silently empty. */
		rx = 0; ry = 0; rx_end = frame->width; ry_end = frame->height;
	}

	process_frame_roi(bw, ob, frame, gain, rx, ry, rx_end, ry_end);

	/* Optional: dump the actual controller-tracking frame (what the constellation tracker sees) as
	 * PGM — the real short-exposure LED images, which the EuRoC recorder does NOT capture (it taps the
	 * SLAM-exposure stream). Enabled by G2_DUMP_FRAMES=<dir>. Frames are copied into a bounded async writer
	 * queue so capture I/O cannot stall controller tracking. Default dumps every frame but drops if storage
	 * cannot keep up; set G2_DUMP_FRAMES_BLOCKING=1 for lossless offline-corpus capture, or
	 * G2_DUMP_FRAMES_STRIDE=N for sparse visual inspection. */
	if (g2_pgm_dump_enabled() && frame->data != NULL && frame->width > 0 && frame->stride >= frame->width) {
		const uint32_t dump_ctr = atomic_fetch_add_explicit(&g_pgm_dump_stride_ctr, 1, memory_order_relaxed);
		if ((dump_ctr % g_pgm_dump.stride) == 0) {
			char path[512];
			// Encode cam, frame timestamp (ns, for IMU alignment in offline replay), exposure (the real
			// controller exposure the matcher needs), source seq (groups the 4 cams of one frame), blobs.
			snprintf(path, sizeof(path), "%s/cam%u_t%020" PRId64 "_e%u_s%010" PRIu64 "_n%u.pgm",
			         g_pgm_dump.dir, bw->cam_id, frame->timestamp, (unsigned)exposure,
			         frame->source_sequence, (unsigned)ob->num_blobs);
			g2_pgm_dump_enqueue(path, frame);
		}
	}

	/* Telemetry: one row per processed frame per camera. n_blobs is what we just
	 * produced; frame_seq comes from the source frame. hw_ts_ns uses frame->timestamp,
	 * the monotonic-converted frame time (NOT the raw device-clock source_timestamp),
	 * so it shares the common clock with every other stream's t_mono_ns.
	 * exposure is the authoritative per-frame value read from the camera's pixel header
	 * (plumbed in by the caller); gain is the commanded value (0 if unknown). */
	g2_telem_frame(bw->cam_id, (uint64_t)frame->timestamp, (uint32_t)frame->source_sequence,
	               (uint16_t)ob->num_blobs, exposure, gain, 0, (uint16_t)ob->dropped_capacity);

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
				dx = fabsf(x - b2->x);
				dy = fabsf(y - b2->y);
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
