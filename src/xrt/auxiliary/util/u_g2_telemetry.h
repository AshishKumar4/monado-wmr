// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief G2 tracking telemetry — lock-free, async, single-clock data capture.
 *
 * Purpose: replace ad-hoc fprintf(stderr) debug spam with a high-rate, non-blocking
 * telemetry pipeline suitable for serious offline analytics (IMU vs visual rates,
 * noise, SLAM<->IMU drift, fusion correctness).
 *
 * Design contract (do not violate in the implementation):
 *  - COMMON CLOCK: every t_mono_ns in every stream is CLOCK_MONOTONIC nanoseconds,
 *    obtained from g2_telem_now_ns() which MUST wrap os_monotonic_get_ns() — the same
 *    clock Monado stamps camera frames and IMU samples with, and the same clock the
 *    euroc recorder writes. This guarantees all streams (and the recorded EuRoC
 *    dataset) align on one timebase. t_mono_ns is the EMIT time for imu/frame/
 *    pose_attempt rows, and the OBSERVATION (frame/sample) time for fusion/event rows
 *    (both are monotonic; only fusion/event differ in which monotonic instant they
 *    name). Sensor rows additionally carry hw_ts_ns, the sensor's own DEVICE-clock
 *    timestamp (per-sensor epoch/rate, NOT cross-stream comparable); only t_mono_ns
 *    is comparable across streams. hw_ts_ns is for per-sensor diagnostics, not for
 *    a t_mono_ns - hw_ts_ns latency that compares across the HMD vs controllers.
 *  - NON-BLOCKING: every g2_telem_* emit is wait-free for the producer (multi-producer
 *    /single-consumer ring per stream — e.g. the three IMU producers share the imu
 *    stream — drained by one background writer thread). An emit NEVER locks, mallocs,
 *    or does I/O on the calling (tracking/IMU) thread.
 *  - NEVER SILENTLY LOSE DATA: each ring is sized large enough to never roll over at
 *    expected rates (see u_g2_telemetry.c capacities). If a ring is ever full, the emit
 *    increments that stream's overflow counter and drops the row; the writer thread logs
 *    the overflow (rate-limited) and g2_telem_shutdown() logs per-stream overflow totals.
 *    Overflow must be impossible in normal operation and loudly visible if it happens.
 *
 * Enable by setting G2_TELEMETRY=<output-dir> in the environment. When unset, every
 * function is a cheap no-op (g2_telem_enabled() == false).
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! The one true clock for all telemetry: CLOCK_MONOTONIC ns (== os_monotonic_get_ns()). */
uint64_t
g2_telem_now_ns(void);

/*!
 * Initialise telemetry. @p out_dir is the directory to write streams + manifest into
 * (typically from getenv("G2_TELEMETRY")). If @p out_dir is NULL/empty, telemetry stays
 * disabled and all emits are no-ops. Safe to call once at tracker startup.
 */
void
g2_telem_init(const char *out_dir);

/*! Flush all rings, join the writer thread, log per-stream overflow totals, close files. */
void
g2_telem_shutdown(void);

/*! True if G2_TELEMETRY was set and init succeeded. Emits are no-ops otherwise. */
bool
g2_telem_enabled(void);

/*! Event type ids for event.bin. Append only: captured telemetry is decoded by these stable numeric ids. */
enum g2_telem_event_type
{
	G2_TELEM_EV_LOCK_LOST = 0,
	G2_TELEM_EV_LOCK_ACQUIRED = 1,
	G2_TELEM_EV_RECOVER_ATTEMPT = 2,
	G2_TELEM_EV_OPTICAL_JUMP_REJECTED = 3,
	G2_TELEM_EV_IMU_ANOMALY = 4,
	G2_TELEM_EV_RING_OVERFLOW = 5,
	/* 6 retired; never reuse — captured event.bin decodes by these ids. */
	G2_TELEM_EV_ESKF_FOLD_COUNT = 7,
	G2_TELEM_EV_ESKF_LEDS_SEEN = 8,
	G2_TELEM_EV_PARTIAL_FOLD_COUNT = 9,
	G2_TELEM_EV_LABEL_PROPAGATED = 10,
	G2_TELEM_EV_JOINT_PNP = 11,
	G2_TELEM_EV_ASSOC_LOCKABLE_NOT_CHOSEN = 12,
	G2_TELEM_EV_ASSOC_LOCK_COMMIT_FAILED = 13,
	G2_TELEM_EV_ASSOC_POSITION_ONLY_SELECTED = 14,
	G2_TELEM_EV_ASSOC_L1_DEPTH_CHECK = 15,
	G2_TELEM_EV_ASSOC_JOINT_CONTENTION = 16,
	G2_TELEM_EV_TRACKER_SEQ_DELTA = 17,
	G2_TELEM_EV_TRACKER_BLOB_MS = 18,
	G2_TELEM_EV_TRACKER_FAST_MS = 19,
	G2_TELEM_EV_FRAME_DUMP_DROPPED = 20,
	G2_TELEM_EV_CAMERA_SOURCE_DELTA = 21,
	G2_TELEM_EV_ASSOC_RAW_EPIPOLAR_POSITION = 22,
	/*! Cold-search work units spent this frame (1 unit = 1 P3P trial, pose check = 5).
	 * Deterministic: <= ASSOC_FRAME_WORK_BUDGET by construction; pairs with TRACKER_FAST_MS
	 * so units->ms drift on other machines is observable offline. */
	G2_TELEM_EV_TRACKER_WORK_UNITS = 23,
	/*! World re-anchor guard absorbed a SLAM discontinuity (B2): the orientation / position
	 * excess beyond the IMU envelope composed into the presentation glide delta this pull.
	 * Emitted as a pair (device 0, same ts): _ANG_DEG = absorbed angle (deg), _POS_M =
	 * absorbed translation (m). The per-pull delta itself is recorded in the dev0 getpose rows. */
	G2_TELEM_EV_WORLD_REANCHOR_ANG_DEG = 24,
	G2_TELEM_EV_WORLD_REANCHOR_POS_M = 25,
	/*! A SLAM camera frame was dropped by the timestamp guard instead of being handed to the
	 * external SLAM system (which aborts on non-monotonic timelines — 2026-07-06 crash class).
	 * t_mono_ns = the offending frame timestamp; value = its delta vs the last pushed frame of
	 * the same camera in ms (<= 0: regression; large > 0: implausible forward jump). One event
	 * per camera view, so a dropped 4-cam group emits 4 events. */
	G2_TELEM_EV_SLAM_FRAME_TS_DROPPED = 26,
	/*! A structurally-complete camera USB transfer was dropped at the parse boundary
	 * (wmr_camera img_xfer_cb) before its content could select or reach either tracking
	 * pipeline. Emitted with the transfer's parsed device-clock start_ts in ns — garbage
	 * values are recorded verbatim for forensics. value: 1 = footer "Dlo+" magic invalid
	 * (footer region carried image data), 2 = device timestamp regressed vs the last
	 * accepted transfer, 3 = implausible forward device-timestamp jump. */
	G2_TELEM_EV_CAMERA_XFER_DROPPED = 27,
};

/* ---- Stream emit functions (POD rows; lock-free; safe from any single producer) ---- */

/*! Raw IMU sample. @p device_id: 0=HMD, 1=left controller, 2=right controller. */
void
g2_telem_imu(uint8_t device_id, uint64_t hw_ts_ns, float ax, float ay, float az, float gx, float gy, float gz);

/*! Per-camera-frame blob/exposure stats (one row per processed frame per camera).
 *  @p dropped_capacity counts qualifying blobs that lost the priority selection at the
 *  per-frame blob cap (0 on every frame that does not hit the cap). */
void
g2_telem_frame(uint8_t cam_id,
               uint64_t hw_ts_ns,
               uint32_t frame_seq,
               uint16_t n_blobs,
               uint16_t exposure,
               uint16_t gain,
               uint16_t led_intensity,
               uint16_t dropped_capacity);

/*! Per-blob retention/centroid row (one per detected blob per camera frame), emitted after the
 *  tracker's static-clutter map has written the blob's retention class. The authoritative record
 *  of what the detector kept and how retention classified it — detection P/R/F1 and
 *  clutter-suppression metrics are scored offline from this stream against projected replay
 *  poses / hand GT. @p retention_class is enum blob_retention_class. */
void
g2_telem_blob(uint8_t cam_id,
              uint64_t hw_ts_ns,
              float x,
              float y,
              uint8_t brightness,
              uint8_t retention_class,
              float static_dwell_s,
              float pos_var_px2);

/*! Constellation/PnP optical pose attempt. @p outcome: 0=rejected,1=accepted,2=recovered.
 *  @p pose is [px,py,pz, qx,qy,qz,qw] in the camera-relative frame. */
void
g2_telem_pose_attempt(uint8_t device_id,
                      uint8_t cam_id,
                      uint64_t hw_ts_ns,
                      uint8_t leds_visible,
                      uint8_t blobs_matched,
                      uint8_t inliers,
                      float reproj_err_px,
                      const float pose[7],
                      uint8_t outcome);

/*! Candidate/twin decision telemetry. @p stage: 1=global/last pose, 2=recovery, 3=prior-refine,
 * 4=long-search. @p candidate: 0=primary/no-twin, 1=twin. @p selected says this candidate was the one carried
 * forward by the front-end. @p outcome uses pose_attempt's 0/1/2 rejected/accepted/recovered convention.
 * @p prior_pos_err and @p prior_rot_err are the per-axis errors from pose_metrics, and @p pose is
 * [px,py,pz,qx,qy,qz,qw] in the camera-relative frame. */
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
                   float blob_var_mean_px2,
                   float blob_brightness_mean,
                   float blob_area_mean,
                   const float pose[7]);

/*! Correspondence long-search diagnostics. This is emitted once per search pass,
 * including failed passes, so visible-controller dropouts are not silent. @p result:
 * 0=success, 1=no_searchable_anchors, 2=no_anchor_with_3_neighbours, 3=no_p3p_trials,
 * 4=no_pose_checks, 5=all_pose_checks_pruned, 6=best_not_good, 7=no_good_candidate.
 * 8-14 are prior-ROI skip reasons emitted before full-frame fallback. */
enum g2_telem_search_bng_reason
{
	G2_SEARCH_BNG_MATCHED_LT3 = 1u << 0,
	G2_SEARCH_BNG_PRIOR_POSITION_FAIL = 1u << 1,
	G2_SEARCH_BNG_PRIOR_ORIENT_FAIL = 1u << 2,
	G2_SEARCH_BNG_LED_IDS_FAIL = 1u << 3,
	G2_SEARCH_BNG_REPROJ_FAIL = 1u << 4,
	G2_SEARCH_BNG_CLEAN_CLUSTER_FAIL = 1u << 5,
	G2_SEARCH_BNG_VISIBLE_COVER_FAIL = 1u << 6,
	G2_SEARCH_BNG_MINIMAL_PRIOR_FAIL = 1u << 7,
	G2_SEARCH_BNG_PRIORLESS_LARGE_FAIL = 1u << 8,
};

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
                float reproj_err_px,
                uint32_t bng_reason_flags,
                float reproj_err_per_match,
                float unmatched_per_match,
                float matched_visible_ratio,
                uint32_t work_spent,
                uint8_t budget_exhausted);

/*! Fusion step: optical observation vs IMU-predicted state -> residual (the SLAM<->IMU
 *  drift). @p outcome: 0=rejected,1=accepted,2=reset. Poses are [px,py,pz,qx,qy,qz,qw]. */
void
g2_telem_fusion(uint8_t device_id,
                uint64_t ts_ns,
                const float optical_pose[7],
                const float predicted_pose[7],
                float pos_residual_m,
                float rot_residual_deg,
                uint8_t outcome);

/*! Generic event marker for "interesting points" (lock loss, recovery, large jump…).
 *  @p event_type values are the stable ids in enum g2_telem_event_type. @p value is event-specific. */
void
g2_telem_event(uint8_t device_id, uint64_t ts_ns, uint16_t event_type, float value);

/*! HMD (head) world pose at a controller frame, [px,py,pz, qx,qy,qz,qw]. @p ts_ns is the frame time the
 *  pose is valid for. Recorded so the offline replay harness can reproduce the true camera->world
 *  transform the live SLAM head pose provides (instead of an IMU-only reconstruction). */
void
g2_telem_head_pose(uint64_t ts_ns, const float pose[7]);

/*! Flags for a mask stream row. A rect can be built from the live prediction, the last
 *  optically-seen pose, or both; ENABLED means the rect was actually pushed to SLAM
 *  (AREA_CAPPED marks a rect the per-camera area cap disabled instead). */
enum g2_telem_mask_flags
{
	G2_TELEM_MASK_ENABLED = 1u << 0,
	G2_TELEM_MASK_FROM_PREDICTION = 1u << 1,
	G2_TELEM_MASK_FROM_LAST_SEEN = 1u << 2,
	G2_TELEM_MASK_AREA_CAPPED = 1u << 3,
};

/*! SLAM controller-mask rect pushed for one device on one SLAM camera at one controller
 *  frame (B5 mask-repair observability: rect age / steal / leak are scored offline from
 *  this stream). @p ts_ns is the frame (exposure) time the rect was computed for,
 *  @p cam_id the SLAM camera index the rect applies to. @p rect is [x0,y0,x1,y1] pixels
 *  (valid when ENABLED). @p sigma_px is the projected PRIOR_GATE_SIGMA position
 *  uncertainty the prediction rect was inflated by (<0 when the fusion exposes no
 *  uncertainty). @p optical_age_ms is the age of the device's last accepted optical pose
 *  (<0 before the first accept). */
void
g2_telem_mask(uint8_t cam_id,
              uint64_t ts_ns,
              uint8_t device_id,
              uint8_t flags,
              const float rect[4],
              float sigma_px,
              float optical_age_ms);

/*! The PRESENTED pose a SteamVR GetPose pull returned for a device (0=HMD, 1=left, 2=right):
 *  pose [px,py,pz,qx,qy,qz,qw], linear + angular velocity, and the xrt relation flags. The
 *  signal the user physically feels (vrserver predicts photon time from this pose+velocity).
 *  @p delta (optional, [dpx,dpy,dpz,dqx,dqy,dqz,dqw]) is the world re-anchor glide correction
 *  composed into the presented pose at this pull (identity/zeros while no glide is active), so
 *  the raw tracker pose is recoverable offline as delta^-1 composed with the presented pose. */
void
g2_telem_getpose(uint8_t device_id,
                 uint64_t ts_ns,
                 const float pose[7],
                 const float lin_vel[3],
                 const float ang_vel[3],
                 uint32_t relation_flags,
                 const float delta[7]);

#ifdef __cplusplus
}
#endif
