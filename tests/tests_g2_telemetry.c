// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Standalone stress + correctness test for u_g2_telemetry.
 * @ingroup aux_util
 *
 * Self-contained: compile with the few Monado deps (os_time / os_threading are
 * header-only inline; u_log* are stubbed below) instead of a full Monado build.
 *
 *   gcc -std=c11 -O2 -pthread -D_GNU_SOURCE \
 *     -I src/xrt/include -I src/xrt/auxiliary \
 *     tests/tests_g2_telemetry.c src/xrt/auxiliary/util/u_g2_telemetry.c \
 *     -o /tmp/g2_telem_test && /tmp/g2_telem_test /tmp/g2_telem_out
 *
 * Checks:
 *   (a) peak rate for a few seconds -> ZERO overflow, .bin counts == emitted
 *   (b) tiny ring + overrun -> overflow counted, logged, event_type=5 row present
 *   (c) t_mono_ns non-decreasing within a stream
 *   (d) manifest.json field offsets == real struct offsets (offsetof)
 *
 * The packed row structs live inside u_g2_telemetry.c (not the public header),
 * so this test keeps shadow copies that MUST byte-match them for the offsetof /
 * sizeof checks in test (d) to mean anything.
 */

#include "util/u_g2_telemetry.h"
#include "os/os_time.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>


/*
 *
 * Shadow row structs + enums (byte-identical to u_g2_telemetry.c internals).
 *
 */

#if defined(_MSC_VER)
#define G2_PACKED
#pragma pack(push, 1)
#else
#define G2_PACKED __attribute__((packed))
#endif

struct g2_telem_imu
{
	uint64_t t_mono_ns;
	uint64_t hw_ts_ns;
	uint8_t device_id;
	float ax, ay, az;
	float gx, gy, gz;
} G2_PACKED;

struct g2_telem_frame
{
	uint64_t t_mono_ns;
	uint64_t hw_ts_ns;
	uint8_t cam_id;
	uint32_t frame_seq;
	uint16_t n_blobs;
	uint16_t exposure;
	uint16_t gain;
	uint16_t led_intensity;
} G2_PACKED;

struct g2_telem_pose_attempt
{
	uint64_t t_mono_ns;
	uint64_t hw_ts_ns;
	uint8_t device_id;
	uint8_t cam_id;
	uint8_t leds_visible;
	uint8_t blobs_matched;
	uint8_t inliers;
	uint8_t outcome;
	float reproj_err_px;
	float px, py, pz;
	float qx, qy, qz, qw;
} G2_PACKED;

struct g2_telem_candidate
{
	uint64_t t_mono_ns;
	uint64_t hw_ts_ns;
	uint8_t device_id;
	uint8_t cam_id;
	uint8_t stage;
	uint8_t candidate;
	uint8_t selected;
	uint8_t had_twin;
	uint8_t outcome;
	uint8_t prior_tilt_trusted;
	uint8_t leds_visible;
	uint8_t blobs_matched;
	uint8_t unmatched_blobs;
	uint8_t inliers;
	uint32_t match_flags;
	float reproj_err_px;
	float prior_cost;
	float total_cost;
	float yaw_sigma_rad;
	float tilt_err_rad;
	float yaw_err_rad;
	float prior_pos_err_x, prior_pos_err_y, prior_pos_err_z;
	float prior_rot_err_x, prior_rot_err_y, prior_rot_err_z;
	float blob_var_mean_px2;
	float blob_brightness_mean;
	float blob_area_mean;
	float px, py, pz;
	float qx, qy, qz, qw;
} G2_PACKED;

struct g2_telem_search
{
	uint64_t t_mono_ns;
	uint64_t hw_ts_ns;
	uint8_t device_id;
	uint8_t cam_id;
	uint8_t pass;
	uint8_t result;
	uint16_t search_flags;
	uint8_t prior_tilt_trusted;
	uint32_t input_blobs;
	uint32_t searchable_anchors;
	uint32_t filtered_anchors;
	uint32_t anchors_with_3_neighbours;
	uint32_t neighbour_links;
	uint32_t num_trials;
	uint32_t num_pose_checks;
	uint32_t num_pose_checks_pruned;
	uint8_t min_led_depth;
	uint8_t max_led_depth;
	uint8_t max_blob_depth;
	uint8_t best_blob_depth;
	uint8_t best_led_depth;
	uint32_t match_flags;
	uint8_t leds_visible;
	uint8_t blobs_matched;
	uint8_t unmatched_blobs;
	float reproj_err_px;
	uint32_t bng_reason_flags;
	float reproj_err_per_match;
	float unmatched_per_match;
	float matched_visible_ratio;
} G2_PACKED;

struct g2_telem_fusion
{
	uint64_t t_mono_ns;
	uint8_t device_id;
	uint8_t outcome;
	float pos_residual_m;
	float rot_residual_deg;
	float opt_px, opt_py, opt_pz, opt_qx, opt_qy, opt_qz, opt_qw;
	float pred_px, pred_py, pred_pz, pred_qx, pred_qy, pred_qz, pred_qw;
} G2_PACKED;

struct g2_telem_event
{
	uint64_t t_mono_ns;
	uint8_t device_id;
	uint16_t event_type;
	float value;
} G2_PACKED;

#if defined(_MSC_VER)
#pragma pack(pop)
#endif
#undef G2_PACKED

/*
 *
 * Logging stubs (so we don't need to link aux_util's u_logging.c).
 *
 */

#include "util/u_logging.h"

enum u_logging_level
u_log_get_global_level(void)
{
	return U_LOGGING_TRACE;
}

void
u_log(const char *file, int line, const char *func, enum u_logging_level level, const char *format, ...)
{
	(void)file;
	(void)line;
	(void)level;
	va_list args;
	va_start(args, format);
	fprintf(stderr, "[%s] ", func);
	vfprintf(stderr, format, args);
	fprintf(stderr, "\n");
	va_end(args);
}


/*
 *
 * Tiny harness.
 *
 */

static int g_fail = 0;
#define CHECK(cond, ...)                                                                                                \
	do {                                                                                                            \
		if (!(cond)) {                                                                                          \
			fprintf(stderr, "FAIL: " __VA_ARGS__);                                                          \
			fprintf(stderr, "\n  at %s:%d (%s)\n", __FILE__, __LINE__, #cond);                              \
			g_fail++;                                                                                       \
		}                                                                                                       \
	} while (0)

static long
file_size(const char *dir, const char *name)
{
	char path[1100];
	snprintf(path, sizeof(path), "%s/%s", dir, name);
	FILE *f = fopen(path, "rb");
	if (!f) {
		return -1;
	}
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fclose(f);
	return n;
}


/*
 *
 * Manifest field-offset verification (minimal JSON probing, no parser dep).
 *
 */

static long
find_offset(const char *json, const char *stream, const char *field)
{
	// Locate the stream block, then within it the field's offset.
	char streamkey[128];
	snprintf(streamkey, sizeof(streamkey), "\"%s\":", stream);
	const char *sp = strstr(json, streamkey);
	if (!sp) {
		return -1;
	}
	char fieldkey[128];
	snprintf(fieldkey, sizeof(fieldkey), "\"name\":\"%s\"", field);
	const char *fp = strstr(sp, fieldkey);
	if (!fp) {
		return -1;
	}
	const char *op = strstr(fp, "\"offset\":");
	if (!op) {
		return -1;
	}
	return strtol(op + strlen("\"offset\":"), NULL, 10);
}

static long
find_row_size(const char *json, const char *stream)
{
	char streamkey[128];
	snprintf(streamkey, sizeof(streamkey), "\"%s\":", stream);
	const char *sp = strstr(json, streamkey);
	if (!sp) {
		return -1;
	}
	const char *rp = strstr(sp, "\"row_size\":");
	if (!rp) {
		return -1;
	}
	return strtol(rp + strlen("\"row_size\":"), NULL, 10);
}

static char *
slurp(const char *dir, const char *name)
{
	char path[1100];
	snprintf(path, sizeof(path), "%s/%s", dir, name);
	FILE *f = fopen(path, "rb");
	if (!f) {
		return NULL;
	}
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *buf = malloc((size_t)n + 1);
	if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
		free(buf);
		fclose(f);
		return NULL;
	}
	buf[n] = '\0';
	fclose(f);
	return buf;
}


/*
 *
 * Test (a): peak rate, multi-threaded, zero overflow, counts match.
 *
 */

// One IMU producer thread per device (mirrors reality: each device's stream has
// a single producer). 3 devices x 200k rows = 600k IMU rows total.
#define N_IMU_THREADS 3
#define IMU_PER_THREAD 200000
#define N_FRAME 50000
#define N_POSE 60000
#define N_FUSION 60000

// Emit in chunks well below each ring's capacity, with a yield between chunks so
// the continuously-draining writer always frees slots ahead of the producer.
// This models a sensor producing at its rate while the writer keeps up: the
// contract guarantees ZERO overflow in this regime. The chunk is sized small
// relative to the smallest ring (frame: 32768) and the sleep is generous enough
// for the writer to flush a chunk even while the IMU streams flood it.
#define CHUNK 2000
static inline void
pace(void)
{
	os_nanosleep(2 * 1000 * 1000); // 2 ms: lets the writer drain a chunk
}

static void *
imu_producer(void *arg)
{
	uint8_t dev = (uint8_t)(intptr_t)arg;
	for (long i = 0; i < IMU_PER_THREAD; i++) {
		g2_telem_imu(dev, (uint64_t)i, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
		if ((i % CHUNK) == (CHUNK - 1)) {
			pace();
		}
	}
	return NULL;
}

static void
test_peak(const char *dir)
{
	fprintf(stderr, "== test (a): peak rate, zero overflow ==\n");
	g2_telem_init(dir);
	CHECK(g2_telem_enabled(), "init enabled / enabled flag set");

	// Producers run concurrently (proves the rings are MPSC-safe across threads).
	pthread_t th[N_IMU_THREADS];
	for (int t = 0; t < N_IMU_THREADS; t++) {
		pthread_create(&th[t], NULL, imu_producer, (void *)(intptr_t)t); // device_id = t
	}

	float pose[7] = {0.1f, 0.2f, 0.3f, 0.0f, 0.0f, 0.0f, 1.0f};
	float err3[3] = {0.01f, 0.02f, 0.03f};
	float opt[7] = {1, 2, 3, 0, 0, 0, 1};
	float pred[7] = {1, 2, 3, 0, 0, 0, 1};
	for (long i = 0; i < N_FRAME; i++) {
		g2_telem_frame((uint8_t)(i % 4), (uint64_t)i, (uint32_t)i, 12, 1000, 8, 200);
		if ((i % CHUNK) == (CHUNK - 1)) {
			pace();
		}
	}
	for (long i = 0; i < N_POSE; i++) {
		g2_telem_pose_attempt(0, 0, (uint64_t)i, 10, 8, 7, 0.5f, pose, 1);
		g2_telem_candidate(0, 0, (uint64_t)i, 3, 0, 1, 1, 1, 0x31, 10, 8, 2, 7, 0.5f, 1.0f, 1.5f, 1,
		                  0.2f, 0.1f, 0.15f, err3, err3, 0.1f, 120.0f, 4.0f, pose);
		g2_telem_search(0, 0, (uint64_t)i, 1, 6, 0x35, 1, 12, 8, 4, 6, 32, 100, 50, 2, 1, 8, 5, 4, 7,
		               0x31, 10, 8, 2, 0.5f, G2_SEARCH_BNG_REPROJ_FAIL, 0.0625f, 0.25f, 0.8f);
		if ((i % CHUNK) == (CHUNK - 1)) {
			pace();
		}
	}
	for (long i = 0; i < N_FUSION; i++) {
		g2_telem_fusion(0, g2_telem_now_ns(), opt, pred, 0.01f, 0.5f, 1);
		if ((i % CHUNK) == (CHUNK - 1)) {
			pace();
		}
	}
	g2_telem_event(0, g2_telem_now_ns(), G2_TELEM_EV_LOCK_ACQUIRED, 1.0f);

	for (int t = 0; t < N_IMU_THREADS; t++) {
		pthread_join(th[t], NULL);
	}

	// Give the writer a moment, then shut down (final drain happens in shutdown).
	g2_telem_shutdown();

	const long imu_total = (long)N_IMU_THREADS * IMU_PER_THREAD;
	long imu_bytes = file_size(dir, "imu.bin");
	long frame_bytes = file_size(dir, "frame.bin");
	long pose_bytes = file_size(dir, "pose_attempt.bin");
	long candidate_bytes = file_size(dir, "candidate.bin");
	long search_bytes = file_size(dir, "search.bin");
	long fusion_bytes = file_size(dir, "fusion.bin");
	long event_bytes = file_size(dir, "event.bin");

	char *m = slurp(dir, "manifest.json");
	CHECK(m != NULL, "manifest present");
	long imu_rs = find_row_size(m, "imu");
	long frame_rs = find_row_size(m, "frame");
	long pose_rs = find_row_size(m, "pose_attempt");
	long candidate_rs = find_row_size(m, "candidate");
	long search_rs = find_row_size(m, "search");
	long fusion_rs = find_row_size(m, "fusion");
	long event_rs = find_row_size(m, "event");

	long imu_rows = imu_bytes / imu_rs;
	long frame_rows = frame_bytes / frame_rs;
	long pose_rows = pose_bytes / pose_rs;
	long candidate_rows = candidate_bytes / candidate_rs;
	long search_rows = search_bytes / search_rs;
	long fusion_rows = fusion_bytes / fusion_rs;
	long event_rows = event_bytes / event_rs;

	fprintf(stderr, "  imu: emitted=%ld written=%ld\n", imu_total, imu_rows);
	fprintf(stderr, "  frame: emitted=%d written=%ld\n", N_FRAME, frame_rows);
	fprintf(stderr, "  pose: emitted=%d written=%ld\n", N_POSE, pose_rows);
	fprintf(stderr, "  candidate: emitted=%d written=%ld\n", N_POSE, candidate_rows);
	fprintf(stderr, "  search: emitted=%d written=%ld\n", N_POSE, search_rows);
	fprintf(stderr, "  fusion: emitted=%d written=%ld\n", N_FUSION, fusion_rows);
	fprintf(stderr, "  event: emitted=1 written=%ld\n", event_rows);

	CHECK(imu_rows == imu_total, "imu count matches (%ld != %ld)", imu_rows, imu_total);
	CHECK(frame_rows == N_FRAME, "frame count matches");
	CHECK(pose_rows == N_POSE, "pose count matches");
	CHECK(candidate_rows == N_POSE, "candidate count matches");
	CHECK(search_rows == N_POSE, "search count matches");
	CHECK(fusion_rows == N_FUSION, "fusion count matches");
	// event = 1 emitted; with zero overflow there must be NO ring_overflow rows.
	CHECK(event_rows == 1, "event count matches (zero overflow expected), got %ld", event_rows);

	// Overflow visible in manifest must be 0 for all streams.
	CHECK(strstr(m, "\"overflow_total\": 0") != NULL, "some overflow_total:0 present");
	free(m);
}


/*
 *
 * Test (c): t_mono_ns non-decreasing within a single-producer stream.
 *
 * A stream may be written by multiple producer threads (MPSC), so the merged
 * file isn't globally ordered -- but within one producer (here, one device_id,
 * which mirrors reality: one device == one emitting thread) the commit order is
 * the claim order, so t_mono_ns must be non-decreasing per device.
 *
 */

static void
test_monotonic(const char *dir)
{
	fprintf(stderr, "== test (c): t_mono_ns non-decreasing per producer ==\n");
	char *m = slurp(dir, "manifest.json");
	CHECK(m != NULL, "manifest for monotonic check");
	if (!m) {
		return;
	}
	long rs = find_row_size(m, "imu");
	long t_off = find_offset(m, "imu", "t_mono_ns");
	long dev_off = find_offset(m, "imu", "device_id");
	free(m);
	CHECK(t_off == 0, "imu t_mono_ns offset is 0 (got %ld)", t_off);

	char path[1100];
	snprintf(path, sizeof(path), "%s/imu.bin", dir);
	FILE *f = fopen(path, "rb");
	CHECK(f != NULL, "open imu.bin");
	if (!f) {
		return;
	}
	uint8_t *row = malloc((size_t)rs);
	uint64_t prev[256] = {0}; // per device_id
	long n = 0, bad = 0;
	while (fread(row, (size_t)rs, 1, f) == 1) {
		uint64_t t;
		memcpy(&t, row + t_off, sizeof(t));
		uint8_t dev = row[dev_off];
		if (t < prev[dev]) {
			bad++;
		}
		prev[dev] = t;
		n++;
	}
	fclose(f);
	free(row);
	fprintf(stderr, "  imu rows scanned=%ld, per-device out-of-order=%ld\n", n, bad);
	CHECK(bad == 0, "t_mono_ns non-decreasing per device (%ld violations)", bad);
}


/*
 *
 * Test (d): manifest offsets == real struct offsets.
 *
 */

static void
test_offsets(const char *dir)
{
	fprintf(stderr, "== test (d): manifest offsets == offsetof ==\n");
	char *m = slurp(dir, "manifest.json");
	CHECK(m != NULL, "manifest for offset check");
	if (!m) {
		return;
	}

#define CHKOFF(stream, S, F)                                                                                            \
	do {                                                                                                            \
		long got = find_offset(m, stream, #F);                                                                  \
		long want = (long)offsetof(struct S, F);                                                                \
		CHECK(got == want, "%s.%s offset %ld != offsetof %ld", stream, #F, got, want);                          \
	} while (0)

	CHKOFF("imu", g2_telem_imu, t_mono_ns);
	CHKOFF("imu", g2_telem_imu, hw_ts_ns);
	CHKOFF("imu", g2_telem_imu, device_id);
	CHKOFF("imu", g2_telem_imu, ax);
	CHKOFF("imu", g2_telem_imu, gz);
	CHKOFF("frame", g2_telem_frame, cam_id);
	CHKOFF("frame", g2_telem_frame, frame_seq);
	CHKOFF("frame", g2_telem_frame, led_intensity);
	CHKOFF("pose_attempt", g2_telem_pose_attempt, outcome);
	CHKOFF("pose_attempt", g2_telem_pose_attempt, reproj_err_px);
	CHKOFF("pose_attempt", g2_telem_pose_attempt, qw);
	CHKOFF("candidate", g2_telem_candidate, stage);
	CHKOFF("candidate", g2_telem_candidate, selected);
	CHKOFF("candidate", g2_telem_candidate, match_flags);
	CHKOFF("candidate", g2_telem_candidate, prior_cost);
	CHKOFF("candidate", g2_telem_candidate, yaw_err_rad);
	CHKOFF("candidate", g2_telem_candidate, prior_pos_err_z);
	CHKOFF("candidate", g2_telem_candidate, qw);
	CHKOFF("search", g2_telem_search, result);
	CHKOFF("search", g2_telem_search, search_flags);
	CHKOFF("search", g2_telem_search, anchors_with_3_neighbours);
	CHKOFF("search", g2_telem_search, num_pose_checks_pruned);
	CHKOFF("search", g2_telem_search, best_led_depth);
	CHKOFF("search", g2_telem_search, reproj_err_px);
	CHKOFF("search", g2_telem_search, bng_reason_flags);
	CHKOFF("search", g2_telem_search, matched_visible_ratio);
	CHKOFF("fusion", g2_telem_fusion, pos_residual_m);
	CHKOFF("fusion", g2_telem_fusion, opt_px);
	CHKOFF("fusion", g2_telem_fusion, opt_qw);
	CHKOFF("fusion", g2_telem_fusion, pred_px);
	CHKOFF("fusion", g2_telem_fusion, pred_qw);
	CHKOFF("event", g2_telem_event, event_type);
	CHKOFF("event", g2_telem_event, value);

	// row_size in manifest must equal sizeof the packed struct.
	CHECK(find_row_size(m, "imu") == (long)sizeof(struct g2_telem_imu), "imu row_size");
	CHECK(find_row_size(m, "frame") == (long)sizeof(struct g2_telem_frame), "frame row_size");
	CHECK(find_row_size(m, "pose_attempt") == (long)sizeof(struct g2_telem_pose_attempt), "pose row_size");
	CHECK(find_row_size(m, "candidate") == (long)sizeof(struct g2_telem_candidate), "candidate row_size");
	CHECK(find_row_size(m, "search") == (long)sizeof(struct g2_telem_search), "search row_size");
	CHECK(find_row_size(m, "fusion") == (long)sizeof(struct g2_telem_fusion), "fusion row_size");
	CHECK(find_row_size(m, "event") == (long)sizeof(struct g2_telem_event), "event row_size");

#undef CHKOFF
	free(m);
}


/*
 *
 * Test (b): deliberate overrun (tiny ring) -> overflow counted/logged/event row.
 *
 * We can't shrink the production rings, so we slam the smallest ring (event,
 * cap 8192) far faster than the writer drains by emitting a huge tight burst
 * (no yields, no I/O per row) from several threads at once. The writer must
 * fwrite every drained row, so a producer doing only an atomic + memcpy easily
 * outruns it and the ring fills -> overflow.
 *
 * Note: this relies on the producer outpacing the writer. Under ThreadSanitizer
 * both sides are slowed ~equally, so overflow is not guaranteed; there the two
 * timing-dependent asserts are downgraded to informational (TSan's job here is
 * to prove the lock-free path is race-free, which it does regardless).
 *
 */

#if defined(__SANITIZE_THREAD__)
#define TIMING_SENSITIVE 1
#else
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define TIMING_SENSITIVE 1
#endif
#endif
#endif
#ifndef TIMING_SENSITIVE
#define TIMING_SENSITIVE 0
#endif

#define N_OVERFLOW_THREADS 4
#define OVERFLOW_PER_THREAD 1000000

static void *
overflow_producer(void *arg)
{
	(void)arg;
	for (long i = 0; i < OVERFLOW_PER_THREAD; i++) {
		g2_telem_event(0, g2_telem_now_ns(), G2_TELEM_EV_IMU_ANOMALY, (float)i);
	}
	return NULL;
}

static void
test_overflow(const char *dir)
{
	fprintf(stderr, "== test (b): forced ring overflow ==\n");
	g2_telem_init(dir);
	CHECK(g2_telem_enabled(), "init for overflow");

	// Tight, multi-threaded burst far exceeding the event ring (8192).
	const long burst = (long)N_OVERFLOW_THREADS * OVERFLOW_PER_THREAD;
	pthread_t th[N_OVERFLOW_THREADS];
	for (int t = 0; t < N_OVERFLOW_THREADS; t++) {
		pthread_create(&th[t], NULL, overflow_producer, NULL);
	}
	for (int t = 0; t < N_OVERFLOW_THREADS; t++) {
		pthread_join(th[t], NULL);
	}

	g2_telem_shutdown();

	char *m = slurp(dir, "manifest.json");
	CHECK(m != NULL, "manifest after overflow");
	if (!m) {
		return;
	}

	// Find event stream's overflow_total in the manifest.
	const char *sp = strstr(m, "\"event\":");
	CHECK(sp != NULL, "event stream in manifest");
	long overflow = -1, written = -1;
	if (sp) {
		const char *op = strstr(sp, "\"overflow_total\":");
		const char *wp = strstr(sp, "\"rows_written\":");
		if (op) {
			overflow = strtol(op + strlen("\"overflow_total\":"), NULL, 10);
		}
		if (wp) {
			written = strtol(wp + strlen("\"rows_written\":"), NULL, 10);
		}
	}
	fprintf(stderr, "  event written=%ld overflow=%ld (burst=%ld)\n", written, overflow, burst);

	// Scan event.bin: count writer-injected ring_overflow markers vs burst rows.
	long rs = find_row_size(m, "event");
	long et_off = find_offset(m, "event", "event_type");
	free(m);

	char path[1100];
	snprintf(path, sizeof(path), "%s/event.bin", dir);
	FILE *f = fopen(path, "rb");
	CHECK(f != NULL, "open event.bin");
	if (!f) {
		return;
	}
	uint8_t *row = malloc((size_t)rs);
	long overflow_markers = 0, rows = 0;
	while (fread(row, (size_t)rs, 1, f) == 1) {
		uint16_t et;
		memcpy(&et, row + et_off, sizeof(et));
		if (et == G2_TELEM_EV_RING_OVERFLOW) {
			overflow_markers++;
		}
		rows++;
	}
	fclose(f);
	free(row);
	fprintf(stderr, "  event.bin rows=%ld, ring_overflow markers=%ld\n", rows, overflow_markers);

	// Accounting: every burst row is either written or dropped (none lost). The
	// writer's own ring_overflow markers are extra rows, so subtract them out.
	long burst_rows_written = written - overflow_markers;
	CHECK(burst_rows_written + overflow == burst, "burst rows accounted (%ld + %ld != %ld)", burst_rows_written,
	      overflow, burst);

	if (TIMING_SENSITIVE && overflow == 0) {
		fprintf(stderr, "  (timing-sensitive build: writer kept up, overflow path not exercised)\n");
	} else {
		CHECK(overflow > 0, "overflow counted (%ld)", overflow);
		CHECK(overflow_markers >= 1, "at least one ring_overflow (type=5) marker row present");
	}
}


int
main(int argc, char **argv)
{
	const char *base = (argc > 1) ? argv[1] : "/tmp/g2_telem_out";
	char dir_a[1024], dir_b[1024];
	snprintf(dir_a, sizeof(dir_a), "%s_peak", base);
	snprintf(dir_b, sizeof(dir_b), "%s_overflow", base);

	// Disabled-state sanity: emits before init must be harmless no-ops.
	CHECK(!g2_telem_enabled(), "disabled before init");
	g2_telem_imu(0, 1, 0, 0, 0, 0, 0, 0); // no crash, no file
	CHECK(g2_telem_now_ns() > 0, "clock returns monotonic ns");

	test_peak(dir_a);
	test_monotonic(dir_a);
	test_offsets(dir_a);

	test_overflow(dir_b);

	if (g_fail == 0) {
		fprintf(stderr, "\nALL TESTS PASSED\n");
		return 0;
	}
	fprintf(stderr, "\n%d CHECK(S) FAILED\n", g_fail);
	return 1;
}
