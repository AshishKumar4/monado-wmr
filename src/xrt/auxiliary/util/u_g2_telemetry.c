// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  G2 tracking telemetry implementation.
 * @author G2 telemetry
 * @ingroup aux_util
 *
 * Ring discipline (per stream): a bounded MPSC queue (Vyukov-style) with a
 * pre-allocated power-of-two slot array. Producers claim a slot index with a
 * single atomic_fetch_add on `enqueue_pos` (wait-free, lock-free, no malloc, no
 * I/O). Each slot carries an atomic `seq` stamp: a producer may write slot i
 * only when seq == i, then publishes seq = i+1 with release ordering. If the
 * slot it claimed is not yet free (consumer behind => ring full) the producer
 * does NOT block: it atomically bumps overflow_total and drops the row. The
 * single writer thread is the only consumer; it reads slot at `dequeue_pos`
 * once seq == dequeue_pos+1 (acquire), copies the row out, then frees the slot
 * by setting seq = dequeue_pos + capacity. Because producers commit
 * monotonically increasing enqueue_pos values, t_mono_ns is non-decreasing
 * within a stream as committed.
 *
 * The public API (names, signatures, arg order, return types) is owned by
 * u_g2_telemetry.h. The packed on-disk row structs, stream ids and event-type
 * enum are internal to this .c and defined just below.
 */

#include "xrt/xrt_config_os.h"

#include "os/os_time.h"
#include "os/os_threading.h"

#include "util/u_g2_telemetry.h"
#include "util/u_logging.h"
#include "util/u_misc.h"

#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#if defined(XRT_OS_LINUX) || defined(XRT_OS_OSX)
#include <sys/stat.h>
#include <sys/types.h>
#endif


/*
 *
 * Packed POD row types (one per stream). Written verbatim to the .bin files; the
 * manifest records each field's offset via offsetof so the consumer parses
 * generically. Note: a row struct's tag (e.g. `struct g2_telem_imu`) shares its
 * spelling with a public emit function (`g2_telem_imu`); C keeps tags and
 * ordinary identifiers in separate namespaces so this is well-defined.
 *
 */

#if defined(_MSC_VER)
#define G2_PACKED
#pragma pack(push, 1)
#else
#define G2_PACKED __attribute__((packed))
#endif

//! imu stream row.
struct g2_telem_imu
{
	uint64_t t_mono_ns;
	uint64_t hw_ts_ns;
	uint8_t device_id;
	float ax, ay, az;
	float gx, gy, gz;
} G2_PACKED;

//! frame stream row.
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

//! pose_attempt stream row.
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

//! candidate stream row: front-end candidate/twin/ranking diagnostics.
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
	float px, py, pz;
	float qx, qy, qz, qw;
} G2_PACKED;

//! search stream row: one correspondence-search pass, including failed passes.
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
} G2_PACKED;

//! fusion stream row. Pose components are stored as named scalar f32 fields (not
//! arrays) so the manifest emits only scalar types the Python consumer parses.
struct g2_telem_fusion
{
	uint64_t t_mono_ns;
	uint8_t device_id;
	uint8_t outcome;
	float pos_residual_m;
	float rot_residual_deg;
	float opt_px, opt_py, opt_pz, opt_qx, opt_qy, opt_qz, opt_qw;        //!< optical pose
	float pred_px, pred_py, pred_pz, pred_qx, pred_qy, pred_qz, pred_qw; //!< predicted pose
} G2_PACKED;

//! event stream row.
struct g2_telem_event
{
	uint64_t t_mono_ns;
	uint8_t device_id;
	uint16_t event_type;
	float value;
} G2_PACKED;

//! head_pose stream row: the HMD (head) pose in the world, sampled at each controller frame. This is
//! the live SLAM head pose the controller tracker uses to place its cameras in the world; recording it
//! lets the offline replay harness reproduce the true camera->world transform (instead of an IMU-only
//! reconstruction), which is required to faithfully replay the constellation front-end. t_mono_ns is the
//! frame (observation) time the pose is valid for.
struct g2_telem_head_pose
{
	uint64_t t_mono_ns;
	float px, py, pz, qx, qy, qz, qw;
} G2_PACKED;

#if defined(_MSC_VER)
#pragma pack(pop)
#endif
#undef G2_PACKED

//! event_type enum (see schema). Extend by appending only.
enum g2_telem_event_type
{
	G2_TELEM_EV_LOCK_LOST = 0,
	G2_TELEM_EV_LOCK_ACQUIRED = 1,
	G2_TELEM_EV_RECOVER_ATTEMPT = 2,
	G2_TELEM_EV_OPTICAL_JUMP_REJECTED = 3,
	G2_TELEM_EV_IMU_ANOMALY = 4,
	G2_TELEM_EV_RING_OVERFLOW = 5,      //!< value = stream id
	G2_TELEM_EV_PARTIAL_FOLD_COUNT = 9, //!< value = number of LEDs gate-folded this frame
	G2_TELEM_EV_LABEL_PROPAGATED = 10,  //!< value = number of LEDs label-propagated from the predicted pose
	G2_TELEM_EV_JOINT_PNP = 11,         //!< value = number of contributing cameras in a joint multi-cam PnP solve
	G2_TELEM_EV_ASSOC_LOCKABLE_NOT_CHOSEN = 12, //!< value = shared-blob conflicts against selected hypotheses
	G2_TELEM_EV_ASSOC_LOCK_COMMIT_FAILED = 13,  //!< value = matched blobs that survived exact commit
};

//! Stream ids, used as the value of a ring_overflow event and to index internals.
enum g2_telem_stream
{
	G2_TELEM_STREAM_IMU = 0,
	G2_TELEM_STREAM_FRAME = 1,
	G2_TELEM_STREAM_POSE_ATTEMPT = 2,
	G2_TELEM_STREAM_CANDIDATE = 3,
	G2_TELEM_STREAM_SEARCH = 4,
	G2_TELEM_STREAM_FUSION = 5,
	G2_TELEM_STREAM_EVENT = 6,
	G2_TELEM_STREAM_HEAD_POSE = 7,
	G2_TELEM_STREAM_COUNT = 8,
};


/*
 *
 * Field descriptors (offsets derived from offsetof so manifest == real layout).
 *
 */

struct g2_field
{
	const char *name;
	const char *type; //!< u8 u16 u32 u64 f32 f64
	size_t offset;
};

struct g2_stream_desc
{
	const char *name;
	const char *file;
	size_t row_size;
	uint32_t capacity; //!< entries, power of two
	const struct g2_field *fields;
	size_t n_fields;
};

#define F(STRUCT, MEMBER, TYPE)                                                                                         \
	{                                                                                                              \
		#MEMBER, TYPE, offsetof(struct STRUCT, MEMBER)                                                          \
	}

// clang-format off
static const struct g2_field imu_fields[] = {
    F(g2_telem_imu, t_mono_ns, "u64"), F(g2_telem_imu, hw_ts_ns, "u64"), F(g2_telem_imu, device_id, "u8"),
    F(g2_telem_imu, ax, "f32"), F(g2_telem_imu, ay, "f32"), F(g2_telem_imu, az, "f32"),
    F(g2_telem_imu, gx, "f32"), F(g2_telem_imu, gy, "f32"), F(g2_telem_imu, gz, "f32"),
};

static const struct g2_field frame_fields[] = {
    F(g2_telem_frame, t_mono_ns, "u64"), F(g2_telem_frame, hw_ts_ns, "u64"), F(g2_telem_frame, cam_id, "u8"),
    F(g2_telem_frame, frame_seq, "u32"), F(g2_telem_frame, n_blobs, "u16"), F(g2_telem_frame, exposure, "u16"),
    F(g2_telem_frame, gain, "u16"), F(g2_telem_frame, led_intensity, "u16"),
};

static const struct g2_field pose_attempt_fields[] = {
    F(g2_telem_pose_attempt, t_mono_ns, "u64"), F(g2_telem_pose_attempt, hw_ts_ns, "u64"),
    F(g2_telem_pose_attempt, device_id, "u8"), F(g2_telem_pose_attempt, cam_id, "u8"),
    F(g2_telem_pose_attempt, leds_visible, "u8"), F(g2_telem_pose_attempt, blobs_matched, "u8"),
    F(g2_telem_pose_attempt, inliers, "u8"), F(g2_telem_pose_attempt, outcome, "u8"),
    F(g2_telem_pose_attempt, reproj_err_px, "f32"), F(g2_telem_pose_attempt, px, "f32"),
    F(g2_telem_pose_attempt, py, "f32"), F(g2_telem_pose_attempt, pz, "f32"),
    F(g2_telem_pose_attempt, qx, "f32"), F(g2_telem_pose_attempt, qy, "f32"),
    F(g2_telem_pose_attempt, qz, "f32"), F(g2_telem_pose_attempt, qw, "f32"),
};

static const struct g2_field candidate_fields[] = {
    F(g2_telem_candidate, t_mono_ns, "u64"), F(g2_telem_candidate, hw_ts_ns, "u64"),
    F(g2_telem_candidate, device_id, "u8"), F(g2_telem_candidate, cam_id, "u8"),
    F(g2_telem_candidate, stage, "u8"), F(g2_telem_candidate, candidate, "u8"),
    F(g2_telem_candidate, selected, "u8"), F(g2_telem_candidate, had_twin, "u8"),
    F(g2_telem_candidate, outcome, "u8"), F(g2_telem_candidate, prior_tilt_trusted, "u8"),
    F(g2_telem_candidate, leds_visible, "u8"), F(g2_telem_candidate, blobs_matched, "u8"),
    F(g2_telem_candidate, unmatched_blobs, "u8"), F(g2_telem_candidate, inliers, "u8"),
    F(g2_telem_candidate, match_flags, "u32"), F(g2_telem_candidate, reproj_err_px, "f32"),
    F(g2_telem_candidate, prior_cost, "f32"), F(g2_telem_candidate, total_cost, "f32"),
    F(g2_telem_candidate, yaw_sigma_rad, "f32"), F(g2_telem_candidate, tilt_err_rad, "f32"),
    F(g2_telem_candidate, yaw_err_rad, "f32"),
    F(g2_telem_candidate, prior_pos_err_x, "f32"), F(g2_telem_candidate, prior_pos_err_y, "f32"),
    F(g2_telem_candidate, prior_pos_err_z, "f32"),
    F(g2_telem_candidate, prior_rot_err_x, "f32"), F(g2_telem_candidate, prior_rot_err_y, "f32"),
    F(g2_telem_candidate, prior_rot_err_z, "f32"),
    F(g2_telem_candidate, px, "f32"), F(g2_telem_candidate, py, "f32"), F(g2_telem_candidate, pz, "f32"),
    F(g2_telem_candidate, qx, "f32"), F(g2_telem_candidate, qy, "f32"), F(g2_telem_candidate, qz, "f32"),
    F(g2_telem_candidate, qw, "f32"),
};

static const struct g2_field search_fields[] = {
    F(g2_telem_search, t_mono_ns, "u64"), F(g2_telem_search, hw_ts_ns, "u64"),
    F(g2_telem_search, device_id, "u8"), F(g2_telem_search, cam_id, "u8"),
    F(g2_telem_search, pass, "u8"), F(g2_telem_search, result, "u8"),
    F(g2_telem_search, search_flags, "u16"), F(g2_telem_search, prior_tilt_trusted, "u8"),
    F(g2_telem_search, input_blobs, "u32"), F(g2_telem_search, searchable_anchors, "u32"),
    F(g2_telem_search, filtered_anchors, "u32"), F(g2_telem_search, anchors_with_3_neighbours, "u32"),
    F(g2_telem_search, neighbour_links, "u32"), F(g2_telem_search, num_trials, "u32"),
    F(g2_telem_search, num_pose_checks, "u32"), F(g2_telem_search, num_pose_checks_pruned, "u32"),
    F(g2_telem_search, min_led_depth, "u8"), F(g2_telem_search, max_led_depth, "u8"),
    F(g2_telem_search, max_blob_depth, "u8"), F(g2_telem_search, best_blob_depth, "u8"),
    F(g2_telem_search, best_led_depth, "u8"), F(g2_telem_search, match_flags, "u32"),
    F(g2_telem_search, leds_visible, "u8"), F(g2_telem_search, blobs_matched, "u8"),
    F(g2_telem_search, unmatched_blobs, "u8"), F(g2_telem_search, reproj_err_px, "f32"),
};

static const struct g2_field fusion_fields[] = {
    F(g2_telem_fusion, t_mono_ns, "u64"), F(g2_telem_fusion, device_id, "u8"), F(g2_telem_fusion, outcome, "u8"),
    F(g2_telem_fusion, pos_residual_m, "f32"), F(g2_telem_fusion, rot_residual_deg, "f32"),
    F(g2_telem_fusion, opt_px, "f32"), F(g2_telem_fusion, opt_py, "f32"), F(g2_telem_fusion, opt_pz, "f32"),
    F(g2_telem_fusion, opt_qx, "f32"), F(g2_telem_fusion, opt_qy, "f32"), F(g2_telem_fusion, opt_qz, "f32"),
    F(g2_telem_fusion, opt_qw, "f32"),
    F(g2_telem_fusion, pred_px, "f32"), F(g2_telem_fusion, pred_py, "f32"), F(g2_telem_fusion, pred_pz, "f32"),
    F(g2_telem_fusion, pred_qx, "f32"), F(g2_telem_fusion, pred_qy, "f32"), F(g2_telem_fusion, pred_qz, "f32"),
    F(g2_telem_fusion, pred_qw, "f32"),
};

static const struct g2_field event_fields[] = {
    F(g2_telem_event, t_mono_ns, "u64"), F(g2_telem_event, device_id, "u8"),
    F(g2_telem_event, event_type, "u16"), F(g2_telem_event, value, "f32"),
};

static const struct g2_field head_pose_fields[] = {
    F(g2_telem_head_pose, t_mono_ns, "u64"),
    F(g2_telem_head_pose, px, "f32"), F(g2_telem_head_pose, py, "f32"), F(g2_telem_head_pose, pz, "f32"),
    F(g2_telem_head_pose, qx, "f32"), F(g2_telem_head_pose, qy, "f32"), F(g2_telem_head_pose, qz, "f32"),
    F(g2_telem_head_pose, qw, "f32"),
};
// clang-format on

#define NF(arr) (sizeof(arr) / sizeof((arr)[0]))

// Reference capacities from the schema (power-of-two for masking). Many seconds
// of slack at peak sensor rate so a healthy writer never lets a ring fill.
static const struct g2_stream_desc g2_descs[G2_TELEM_STREAM_COUNT] = {
    [G2_TELEM_STREAM_IMU] = {"imu", "imu.bin", sizeof(struct g2_telem_imu), 262144, imu_fields, NF(imu_fields)},
    [G2_TELEM_STREAM_FRAME] = {"frame", "frame.bin", sizeof(struct g2_telem_frame), 32768, frame_fields,
                               NF(frame_fields)},
    [G2_TELEM_STREAM_POSE_ATTEMPT] = {"pose_attempt", "pose_attempt.bin", sizeof(struct g2_telem_pose_attempt), 65536,
                                      pose_attempt_fields, NF(pose_attempt_fields)},
    [G2_TELEM_STREAM_CANDIDATE] = {"candidate", "candidate.bin", sizeof(struct g2_telem_candidate), 131072,
                                   candidate_fields, NF(candidate_fields)},
    [G2_TELEM_STREAM_SEARCH] = {"search", "search.bin", sizeof(struct g2_telem_search), 131072, search_fields,
                                NF(search_fields)},
    [G2_TELEM_STREAM_FUSION] = {"fusion", "fusion.bin", sizeof(struct g2_telem_fusion), 65536, fusion_fields,
                                NF(fusion_fields)},
    [G2_TELEM_STREAM_EVENT] = {"event", "event.bin", sizeof(struct g2_telem_event), 8192, event_fields,
                               NF(event_fields)},
    [G2_TELEM_STREAM_HEAD_POSE] = {"head_pose", "head_pose.bin", sizeof(struct g2_telem_head_pose), 32768,
                                   head_pose_fields, NF(head_pose_fields)},
};


/*
 *
 * Bounded MPSC ring (one per stream).
 *
 */

struct g2_slot
{
	_Atomic uint64_t seq; //!< Vyukov sequence stamp
	                      // row bytes follow inline (row_size), allocated as one block per ring
};

struct g2_ring
{
	const struct g2_stream_desc *desc;
	uint32_t mask;     //!< capacity - 1
	size_t stride;     //!< bytes per slot = sizeof(seq) padded + row_size, see below
	uint8_t *slots;    //!< capacity * stride, pre-allocated
	size_t row_off;    //!< byte offset of the row inside a slot

	_Atomic uint64_t enqueue_pos; //!< next index a producer will claim
	_Atomic uint64_t dequeue_pos; //!< next index the writer will read (writer-owned)

	_Atomic uint64_t overflow_total;
	uint64_t rows_written; //!< writer-owned

	// Rate-limited overflow logging (writer-owned).
	uint64_t last_overflow_logged;
	int64_t last_overflow_log_ns;

	FILE *fp;
};

// One slot = [ _Atomic uint64_t seq | padding to row align | row bytes ].
#define G2_ROW_OFF (sizeof(_Atomic uint64_t))

static inline uint8_t *
slot_ptr(struct g2_ring *r, uint64_t pos)
{
	return r->slots + (size_t)(pos & r->mask) * r->stride;
}

static inline _Atomic uint64_t *
slot_seq(struct g2_ring *r, uint64_t pos)
{
	return (_Atomic uint64_t *)slot_ptr(r, pos);
}

static inline void *
slot_row(struct g2_ring *r, uint64_t pos)
{
	return slot_ptr(r, pos) + r->row_off;
}


/*
 *
 * Global state.
 *
 */

struct g2_telem
{
	atomic_bool enabled;     //!< fast path gate read by every emit
	atomic_bool init_done;   //!< idempotency guard: set once, first caller only
	bool started;            //!< true once a writer/rings are live (shutdown bookkeeping)

	char dir[1024];
	int64_t start_mono_ns;
	int64_t start_realtime_ns;

	struct g2_ring rings[G2_TELEM_STREAM_COUNT];
	struct os_thread_helper oth;
};

static struct g2_telem g_telem = {0};

#define WRITE_BUF_BYTES (1 << 20) //!< 1 MiB stdio buffer per .bin file


/*
 *
 * Producer side (wait-free).
 *
 */

//! Enqueue a row copy. Returns true if queued, false if dropped (overflow).
static bool
ring_emit(struct g2_ring *r, const void *row)
{
	uint64_t pos = atomic_load_explicit(&r->enqueue_pos, memory_order_relaxed);

	for (;;) {
		_Atomic uint64_t *seqp = slot_seq(r, pos);
		uint64_t seq = atomic_load_explicit(seqp, memory_order_acquire);
		int64_t diff = (int64_t)seq - (int64_t)pos;

		if (diff == 0) {
			// Slot is free at this pos; try to claim it.
			if (atomic_compare_exchange_weak_explicit(&r->enqueue_pos, &pos, pos + 1,
			                                           memory_order_relaxed, memory_order_relaxed)) {
				// Won the slot: write the row, then publish.
				memcpy(slot_row(r, pos), row, r->desc->row_size);
				atomic_store_explicit(seqp, pos + 1, memory_order_release);
				return true;
			}
			// Lost the race; pos was updated, retry.
		} else if (diff < 0) {
			// Slot still holds an unconsumed row at an earlier lap: ring full.
			atomic_fetch_add_explicit(&r->overflow_total, 1, memory_order_relaxed);
			return false;
		} else {
			// Another producer is ahead; refresh pos and retry.
			pos = atomic_load_explicit(&r->enqueue_pos, memory_order_relaxed);
		}
	}
}


/*
 *
 * Consumer side (single writer thread).
 *
 */

//! Try to dequeue one row into out. Returns true if a row was copied.
static bool
ring_dequeue(struct g2_ring *r, void *out)
{
	uint64_t pos = atomic_load_explicit(&r->dequeue_pos, memory_order_relaxed);
	_Atomic uint64_t *seqp = slot_seq(r, pos);
	uint64_t seq = atomic_load_explicit(seqp, memory_order_acquire);
	int64_t diff = (int64_t)seq - (int64_t)(pos + 1);

	if (diff != 0) {
		return false; // Not yet committed.
	}

	memcpy(out, slot_row(r, pos), r->desc->row_size);
	// Free the slot for the next lap.
	atomic_store_explicit(seqp, pos + r->mask + 1, memory_order_release);
	atomic_store_explicit(&r->dequeue_pos, pos + 1, memory_order_relaxed);
	return true;
}


/*
 *
 * Manifest.
 *
 */

static void
write_manifest(void)
{
	char path[1100];
	(void)snprintf(path, sizeof(path), "%s/manifest.json", g_telem.dir);
	FILE *f = fopen(path, "wb");
	if (f == NULL) {
		U_LOG_E("g2_telem: cannot write %s: %s", path, strerror(errno));
		return;
	}

	fprintf(f, "{\n");
	fprintf(f, "  \"version\": 1,\n");
	fprintf(f, "  \"clock\": \"CLOCK_MONOTONIC\",\n");
	fprintf(f, "  \"start\": { \"t_mono_ns\": %lld, \"t_realtime_ns\": %lld },\n",
	        (long long)g_telem.start_mono_ns, (long long)g_telem.start_realtime_ns);
	fprintf(f, "  \"types\": \"little-endian; u8 u16 u32 u64 f32 f64\",\n");
	fprintf(f, "  \"streams\": {\n");

	for (int s = 0; s < G2_TELEM_STREAM_COUNT; s++) {
		struct g2_ring *r = &g_telem.rings[s];
		const struct g2_stream_desc *d = r->desc;
		fprintf(f, "    \"%s\": {\n", d->name);
		fprintf(f, "      \"file\": \"%s\",\n", d->file);
		fprintf(f, "      \"row_size\": %zu,\n", d->row_size);
		fprintf(f, "      \"fields\": [\n");
		for (size_t i = 0; i < d->n_fields; i++) {
			fprintf(f, "        {\"name\":\"%s\",\"type\":\"%s\",\"offset\":%zu}%s\n", d->fields[i].name,
			        d->fields[i].type, d->fields[i].offset, (i + 1 < d->n_fields) ? "," : "");
		}
		fprintf(f, "      ],\n");
		fprintf(f, "      \"rows_written\": %llu,\n", (unsigned long long)r->rows_written);
		fprintf(f, "      \"overflow_total\": %llu\n",
		        (unsigned long long)atomic_load_explicit(&r->overflow_total, memory_order_relaxed));
		fprintf(f, "    }%s\n", (s + 1 < G2_TELEM_STREAM_COUNT) ? "," : "");
	}

	fprintf(f, "  }\n");
	fprintf(f, "}\n");
	fclose(f);
}


/*
 *
 * Writer thread.
 *
 */

// The writer owns event.bin, so it appends the ring_overflow marker directly
// (bypassing the ring): guaranteed to land even if the event ring itself is the
// one overflowing, and no risk of recursing into a full ring.
static void
writer_push_overflow_event(int64_t now, uint8_t stream_id)
{
	struct g2_ring *er = &g_telem.rings[G2_TELEM_STREAM_EVENT];
	if (er->fp == NULL) {
		return;
	}
	struct g2_telem_event ev = {0};
	ev.t_mono_ns = (uint64_t)now;
	ev.device_id = 0;
	ev.event_type = G2_TELEM_EV_RING_OVERFLOW;
	ev.value = (float)stream_id;
	if (fwrite(&ev, sizeof(ev), 1, er->fp) == 1) {
		er->rows_written++;
	}
}

// Drain one ring; returns number of rows written this pass.
static uint32_t
drain_ring(struct g2_ring *r, uint8_t stream_id)
{
	uint8_t row[256]; // largest row is well under this
	uint32_t n = 0;
	while (ring_dequeue(r, row)) {
		if (r->fp != NULL && fwrite(row, r->desc->row_size, 1, r->fp) == 1) {
			r->rows_written++;
		}
		n++;
		if (n >= 65536) {
			break; // bound a single pass so other rings get serviced
		}
	}

	// Overflow visibility: BOTH the log and the marker row are rate-limited to
	// <=1/s. Without this, a sustained overflow storm at a fast drain cadence
	// would append many marker rows to event.bin, inflating event counts.
	uint64_t of = atomic_load_explicit(&r->overflow_total, memory_order_relaxed);
	if (of != r->last_overflow_logged) {
		int64_t now = g2_telem_now_ns();
		if (now - r->last_overflow_log_ns > 1000000000LL) { // <= 1/s for log + marker
			U_LOG_W("g2_telem: stream '%s' overflow: %llu dropped (total)", r->desc->name,
			        (unsigned long long)of);
			// One marker row so the trace is self-describing (incl. when the
			// event ring itself overflowed -- writer appends straight to event.bin).
			writer_push_overflow_event(now, stream_id);
			r->last_overflow_log_ns = now;
			r->last_overflow_logged = of;
		}
	}

	return n;
}

static void *
writer_thread(void *ptr)
{
	(void)ptr;
	os_thread_helper_name(&g_telem.oth, "g2_telem");

	while (os_thread_helper_is_running(&g_telem.oth)) {
		uint32_t total = 0;
		for (int s = 0; s < G2_TELEM_STREAM_COUNT; s++) {
			total += drain_ring(&g_telem.rings[s], (uint8_t)s);
		}
		if (total == 0) {
			os_nanosleep(500 * 1000); // 0.5 ms idle nap; keeps rings drained at rate
		}
	}

	// Final drain after stop is requested.
	for (int s = 0; s < G2_TELEM_STREAM_COUNT; s++) {
		while (drain_ring(&g_telem.rings[s], (uint8_t)s) > 0) {
		}
		if (g_telem.rings[s].fp != NULL) {
			fflush(g_telem.rings[s].fp);
		}
	}

	return NULL;
}


/*
 *
 * Lifecycle.
 *
 */

uint64_t
g2_telem_now_ns(void)
{
	return (uint64_t)os_monotonic_get_ns();
}

bool
g2_telem_enabled(void)
{
	return atomic_load_explicit(&g_telem.enabled, memory_order_relaxed);
}

static bool
ring_alloc(struct g2_ring *r, const struct g2_stream_desc *d)
{
	r->desc = d;
	r->mask = d->capacity - 1;
	r->row_off = G2_ROW_OFF;
	// Round stride up to 8 so each slot's seq stays 8-byte aligned.
	size_t raw = r->row_off + d->row_size;
	r->stride = (raw + 7u) & ~(size_t)7u;
	r->slots = (uint8_t *)calloc(d->capacity, r->stride);
	if (r->slots == NULL) {
		return false;
	}
	// Seed each slot's seq with its index (free).
	for (uint32_t i = 0; i < d->capacity; i++) {
		atomic_store_explicit(slot_seq(r, i), (uint64_t)i, memory_order_relaxed);
	}
	atomic_store_explicit(&r->enqueue_pos, 0, memory_order_relaxed);
	atomic_store_explicit(&r->dequeue_pos, 0, memory_order_relaxed);
	atomic_store_explicit(&r->overflow_total, 0, memory_order_relaxed);
	r->rows_written = 0;
	r->last_overflow_logged = 0;
	r->last_overflow_log_ns = 0;
	r->fp = NULL;
	return true;
}

// Close a ring's file at shutdown. The slot buffer is deliberately NOT freed:
// a producer that already passed the g2_telem_enabled() gate may still be mid
// ring_emit() (the memcpy into r->slots) when shutdown runs on another thread.
// Freeing the buffer there would be a use-after-free. The buffers are small,
// fixed, and live for the process lifetime, so we leak-at-exit on purpose --
// the OS reclaims them at process teardown.
static void
ring_close_file(struct g2_ring *r)
{
	if (r->fp != NULL) {
		fflush(r->fp);
		fclose(r->fp);
		r->fp = NULL;
	}
}

// Free a ring's slot buffer + close its file. Only safe BEFORE the writer thread
// has started and producers are wired up (i.e. the init error path), never at
// shutdown -- use ring_close_file() there.
static void
ring_free(struct g2_ring *r)
{
	ring_close_file(r);
	free(r->slots);
	r->slots = NULL;
}

// Create @p path and any missing parents (mkdir -p semantics).
static bool
make_dir(const char *path)
{
#if defined(XRT_OS_LINUX) || defined(XRT_OS_OSX)
	char tmp[1024];
	(void)snprintf(tmp, sizeof(tmp), "%s", path);
	size_t len = strlen(tmp);
	while (len > 1 && tmp[len - 1] == '/') {
		tmp[--len] = '\0'; // strip trailing slashes
	}
	for (char *p = tmp + 1; *p != '\0'; p++) {
		if (*p != '/') {
			continue;
		}
		*p = '\0';
		if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
			U_LOG_E("g2_telem: mkdir(%s) failed: %s", tmp, strerror(errno));
			return false;
		}
		*p = '/';
	}
	if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
		U_LOG_E("g2_telem: mkdir(%s) failed: %s", tmp, strerror(errno));
		return false;
	}
	return true;
#else
	(void)path;
	return false;
#endif
}

void
g2_telem_init(const char *out_dir)
{
	// Idempotent: many independent code paths (WMR HMD/controller creation, the
	// SLAM tracker) all call init; only the FIRST winner actually initialises.
	// The atomic exchange makes "exactly one init runs" hold even if two threads
	// race here. Later callers (and a NULL/empty dir) just return.
	bool expected = false;
	if (!atomic_compare_exchange_strong_explicit(&g_telem.init_done, &expected, true, memory_order_acq_rel,
	                                             memory_order_acquire)) {
		return; // someone already won init; no-op
	}

	if (out_dir == NULL || out_dir[0] == '\0') {
		atomic_store_explicit(&g_telem.enabled, false, memory_order_relaxed);
		return; // disabled, all emits no-op (init_done stays set: stays disabled)
	}

	(void)snprintf(g_telem.dir, sizeof(g_telem.dir), "%s", out_dir);

	if (!make_dir(g_telem.dir)) {
		return;
	}

	g_telem.start_mono_ns = os_monotonic_get_ns();
	g_telem.start_realtime_ns = os_realtime_get_ns();

	// Allocate rings + open files.
	for (int s = 0; s < G2_TELEM_STREAM_COUNT; s++) {
		const struct g2_stream_desc *d = &g2_descs[s];
		struct g2_ring *r = &g_telem.rings[s];
		if (!ring_alloc(r, d)) {
			U_LOG_E("g2_telem: ring alloc failed for '%s'", d->name);
			goto err;
		}
		char path[1100];
		(void)snprintf(path, sizeof(path), "%s/%s", g_telem.dir, d->file);
		r->fp = fopen(path, "wb");
		if (r->fp == NULL) {
			U_LOG_E("g2_telem: cannot open %s: %s", path, strerror(errno));
			goto err;
		}
		setvbuf(r->fp, NULL, _IOFBF, WRITE_BUF_BYTES);
	}

	write_manifest();

	if (os_thread_helper_init(&g_telem.oth) != 0) {
		U_LOG_E("g2_telem: thread helper init failed");
		goto err;
	}
	if (os_thread_helper_start(&g_telem.oth, writer_thread, NULL) != 0) {
		U_LOG_E("g2_telem: writer thread start failed");
		os_thread_helper_destroy(&g_telem.oth);
		goto err;
	}

	g_telem.started = true;
	atomic_store_explicit(&g_telem.enabled, true, memory_order_release);
	U_LOG_I("g2_telem: enabled, output dir = %s", g_telem.dir);
	return;

err:
	for (int s = 0; s < G2_TELEM_STREAM_COUNT; s++) {
		ring_free(&g_telem.rings[s]);
	}
	atomic_store_explicit(&g_telem.enabled, false, memory_order_relaxed);
}

void
g2_telem_shutdown(void)
{
	if (!g_telem.started) {
		return; // never started (or already shut down); idempotent
	}
	g_telem.started = false; // gate this function against a second call

	// Order matters:
	//  1. Flip the enabled flag OFF so any producer that reaches the gate from
	//     now on no-ops and never enters ring_emit().
	//  2. Stop + join the writer thread (its final drain flushes the rings).
	//     os_thread_helper_destroy() itself stops-and-waits, so a single call
	//     suffices.
	//  3. Close the files. We do NOT free the ring slot buffers: a producer that
	//     passed the gate just before step 1 may still be mid-emit (memcpy into
	//     a slot); freeing under it would be a use-after-free. The buffers are
	//     leaked at exit on purpose (process-lifetime sink).
	atomic_store_explicit(&g_telem.enabled, false, memory_order_release);
	os_thread_helper_destroy(&g_telem.oth);

	for (int s = 0; s < G2_TELEM_STREAM_COUNT; s++) {
		struct g2_ring *r = &g_telem.rings[s];
		uint64_t of = atomic_load_explicit(&r->overflow_total, memory_order_relaxed);
		U_LOG_I("g2_telem: stream '%s' rows_written=%llu overflow_total=%llu", r->desc->name,
		        (unsigned long long)r->rows_written, (unsigned long long)of);
	}

	// Rewrite manifest with final totals (files still open => flush first).
	for (int s = 0; s < G2_TELEM_STREAM_COUNT; s++) {
		if (g_telem.rings[s].fp != NULL) {
			fflush(g_telem.rings[s].fp);
		}
	}
	write_manifest();

	for (int s = 0; s < G2_TELEM_STREAM_COUNT; s++) {
		ring_close_file(&g_telem.rings[s]); // close files; intentionally leak slots
	}

	// Clear the idempotency guard last, so a fresh init() can start a new session
	// after this clean shutdown (e.g. device re-create). Released after the writer
	// is joined and files are closed, so no producer can be mid-emit at this point.
	atomic_store_explicit(&g_telem.init_done, false, memory_order_release);
}


/*
 *
 * Emits.
 *
 */

void
g2_telem_imu(uint8_t device_id, uint64_t hw_ts_ns, float ax, float ay, float az, float gx, float gy, float gz)
{
	if (!g2_telem_enabled()) {
		return;
	}
	struct g2_telem_imu row = {0};
	row.t_mono_ns = (uint64_t)os_monotonic_get_ns();
	row.hw_ts_ns = hw_ts_ns;
	row.device_id = device_id;
	row.ax = ax;
	row.ay = ay;
	row.az = az;
	row.gx = gx;
	row.gy = gy;
	row.gz = gz;
	(void)ring_emit(&g_telem.rings[G2_TELEM_STREAM_IMU], &row);
}

void
g2_telem_frame(uint8_t cam_id,
               uint64_t hw_ts_ns,
               uint32_t frame_seq,
               uint16_t n_blobs,
               uint16_t exposure,
               uint16_t gain,
               uint16_t led_intensity)
{
	if (!g2_telem_enabled()) {
		return;
	}
	struct g2_telem_frame row = {0};
	row.t_mono_ns = (uint64_t)os_monotonic_get_ns();
	row.hw_ts_ns = hw_ts_ns;
	row.cam_id = cam_id;
	row.frame_seq = frame_seq;
	row.n_blobs = n_blobs;
	row.exposure = exposure;
	row.gain = gain;
	row.led_intensity = led_intensity;
	(void)ring_emit(&g_telem.rings[G2_TELEM_STREAM_FRAME], &row);
}

void
g2_telem_pose_attempt(uint8_t device_id,
                      uint8_t cam_id,
                      uint64_t hw_ts_ns,
                      uint8_t leds_visible,
                      uint8_t blobs_matched,
                      uint8_t inliers,
                      float reproj_err_px,
                      const float pose[7],
                      uint8_t outcome)
{
	if (!g2_telem_enabled()) {
		return;
	}
	struct g2_telem_pose_attempt row = {0};
	row.t_mono_ns = (uint64_t)os_monotonic_get_ns();
	row.hw_ts_ns = hw_ts_ns;
	row.device_id = device_id;
	row.cam_id = cam_id;
	row.leds_visible = leds_visible;
	row.blobs_matched = blobs_matched;
	row.inliers = inliers;
	row.outcome = outcome;
	row.reproj_err_px = reproj_err_px;
	if (pose != NULL) {
		row.px = pose[0];
		row.py = pose[1];
		row.pz = pose[2];
		row.qx = pose[3];
		row.qy = pose[4];
		row.qz = pose[5];
		row.qw = pose[6];
	}
	(void)ring_emit(&g_telem.rings[G2_TELEM_STREAM_POSE_ATTEMPT], &row);
}

void
g2_telem_candidate(uint8_t device_id,
                   uint8_t cam_id,
                   uint64_t ts_ns,
                   uint8_t stage,
                   uint8_t candidate,
                   uint8_t selected,
                   uint8_t had_twin,
                   uint8_t outcome,
                   uint32_t match_flags,
                   uint8_t leds_visible,
                   uint8_t blobs_matched,
                   uint8_t unmatched_blobs,
                   uint8_t inliers,
                   float reproj_err_px,
                   float prior_cost,
                   float total_cost,
                   uint8_t prior_tilt_trusted,
                   float yaw_sigma_rad,
                   float tilt_err_rad,
                   float yaw_err_rad,
                   const float prior_pos_err[3],
                   const float prior_rot_err[3],
                   const float pose[7])
{
	if (!g2_telem_enabled()) {
		return;
	}
	struct g2_telem_candidate row = {0};
	row.t_mono_ns = ts_ns;
	row.hw_ts_ns = ts_ns;
	row.device_id = device_id;
	row.cam_id = cam_id;
	row.stage = stage;
	row.candidate = candidate;
	row.selected = selected;
	row.had_twin = had_twin;
	row.outcome = outcome;
	row.prior_tilt_trusted = prior_tilt_trusted;
	row.leds_visible = leds_visible;
	row.blobs_matched = blobs_matched;
	row.unmatched_blobs = unmatched_blobs;
	row.inliers = inliers;
	row.match_flags = match_flags;
	row.reproj_err_px = reproj_err_px;
	row.prior_cost = prior_cost;
	row.total_cost = total_cost;
	row.yaw_sigma_rad = yaw_sigma_rad;
	row.tilt_err_rad = tilt_err_rad;
	row.yaw_err_rad = yaw_err_rad;
	if (prior_pos_err != NULL) {
		row.prior_pos_err_x = prior_pos_err[0];
		row.prior_pos_err_y = prior_pos_err[1];
		row.prior_pos_err_z = prior_pos_err[2];
	}
	if (prior_rot_err != NULL) {
		row.prior_rot_err_x = prior_rot_err[0];
		row.prior_rot_err_y = prior_rot_err[1];
		row.prior_rot_err_z = prior_rot_err[2];
	}
	if (pose != NULL) {
		row.px = pose[0];
		row.py = pose[1];
		row.pz = pose[2];
		row.qx = pose[3];
		row.qy = pose[4];
		row.qz = pose[5];
		row.qw = pose[6];
	}
	(void)ring_emit(&g_telem.rings[G2_TELEM_STREAM_CANDIDATE], &row);
}

void
g2_telem_search(uint8_t device_id,
                uint8_t cam_id,
                uint64_t ts_ns,
                uint8_t pass,
                uint8_t result,
                uint16_t search_flags,
                uint8_t prior_tilt_trusted,
                uint32_t input_blobs,
                uint32_t searchable_anchors,
                uint32_t filtered_anchors,
                uint32_t anchors_with_3_neighbours,
                uint32_t neighbour_links,
                uint32_t num_trials,
                uint32_t num_pose_checks,
                uint32_t num_pose_checks_pruned,
                uint8_t min_led_depth,
                uint8_t max_led_depth,
                uint8_t max_blob_depth,
                uint8_t best_blob_depth,
                uint8_t best_led_depth,
                uint32_t match_flags,
                uint8_t leds_visible,
                uint8_t blobs_matched,
                uint8_t unmatched_blobs,
                float reproj_err_px)
{
	if (!g2_telem_enabled()) {
		return;
	}
	struct g2_telem_search row = {0};
	row.t_mono_ns = ts_ns;
	row.hw_ts_ns = ts_ns;
	row.device_id = device_id;
	row.cam_id = cam_id;
	row.pass = pass;
	row.result = result;
	row.search_flags = search_flags;
	row.prior_tilt_trusted = prior_tilt_trusted;
	row.input_blobs = input_blobs;
	row.searchable_anchors = searchable_anchors;
	row.filtered_anchors = filtered_anchors;
	row.anchors_with_3_neighbours = anchors_with_3_neighbours;
	row.neighbour_links = neighbour_links;
	row.num_trials = num_trials;
	row.num_pose_checks = num_pose_checks;
	row.num_pose_checks_pruned = num_pose_checks_pruned;
	row.min_led_depth = min_led_depth;
	row.max_led_depth = max_led_depth;
	row.max_blob_depth = max_blob_depth;
	row.best_blob_depth = best_blob_depth;
	row.best_led_depth = best_led_depth;
	row.match_flags = match_flags;
	row.leds_visible = leds_visible;
	row.blobs_matched = blobs_matched;
	row.unmatched_blobs = unmatched_blobs;
	row.reproj_err_px = reproj_err_px;
	(void)ring_emit(&g_telem.rings[G2_TELEM_STREAM_SEARCH], &row);
}

void
g2_telem_fusion(uint8_t device_id,
                uint64_t ts_ns,
                const float optical_pose[7],
                const float predicted_pose[7],
                float pos_residual_m,
                float rot_residual_deg,
                uint8_t outcome)
{
	if (!g2_telem_enabled()) {
		return;
	}
	// t_mono_ns carries the observation (frame/sample) time here, not the emit
	// time -- see the per-stream note in u_g2_telemetry.h / TELEMETRY-SCHEMA.md.
	struct g2_telem_fusion row = {0};
	row.t_mono_ns = ts_ns;
	row.device_id = device_id;
	row.outcome = outcome;
	row.pos_residual_m = pos_residual_m;
	row.rot_residual_deg = rot_residual_deg;
	if (optical_pose != NULL) {
		row.opt_px = optical_pose[0];
		row.opt_py = optical_pose[1];
		row.opt_pz = optical_pose[2];
		row.opt_qx = optical_pose[3];
		row.opt_qy = optical_pose[4];
		row.opt_qz = optical_pose[5];
		row.opt_qw = optical_pose[6];
	}
	if (predicted_pose != NULL) {
		row.pred_px = predicted_pose[0];
		row.pred_py = predicted_pose[1];
		row.pred_pz = predicted_pose[2];
		row.pred_qx = predicted_pose[3];
		row.pred_qy = predicted_pose[4];
		row.pred_qz = predicted_pose[5];
		row.pred_qw = predicted_pose[6];
	}
	(void)ring_emit(&g_telem.rings[G2_TELEM_STREAM_FUSION], &row);
}

void
g2_telem_event(uint8_t device_id, uint64_t ts_ns, uint16_t event_type, float value)
{
	if (!g2_telem_enabled()) {
		return;
	}
	struct g2_telem_event row = {0};
	row.t_mono_ns = ts_ns;
	row.device_id = device_id;
	row.event_type = event_type;
	row.value = value;
	(void)ring_emit(&g_telem.rings[G2_TELEM_STREAM_EVENT], &row);
}

void
g2_telem_head_pose(uint64_t ts_ns, const float pose[7])
{
	if (!g2_telem_enabled() || pose == NULL) {
		return;
	}
	// t_mono_ns carries the frame (observation) time the head pose is valid for.
	struct g2_telem_head_pose row = {0};
	row.t_mono_ns = ts_ns;
	row.px = pose[0];
	row.py = pose[1];
	row.pz = pose[2];
	row.qx = pose[3];
	row.qy = pose[4];
	row.qz = pose[5];
	row.qw = pose[6];
	(void)ring_emit(&g_telem.rings[G2_TELEM_STREAM_HEAD_POSE], &row);
}
