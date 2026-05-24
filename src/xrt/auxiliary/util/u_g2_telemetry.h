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
 *  - NON-BLOCKING: every g2_telem_* emit is wait-free for the producer (single-producer
 *    /single-consumer ring per stream + one background writer thread). An emit NEVER
 *    locks, mallocs, or does I/O on the calling (tracking/IMU) thread.
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

/* ---- Stream emit functions (POD rows; lock-free; safe from any single producer) ---- */

/*! Raw IMU sample. @p device_id: 0=HMD, 1=left controller, 2=right controller. */
void
g2_telem_imu(uint8_t device_id, uint64_t hw_ts_ns, float ax, float ay, float az, float gx, float gy, float gz);

/*! Per-camera-frame blob/exposure stats (one row per processed frame per camera). */
void
g2_telem_frame(uint8_t cam_id,
               uint64_t hw_ts_ns,
               uint32_t frame_seq,
               uint16_t n_blobs,
               uint16_t exposure,
               uint16_t gain,
               uint16_t led_intensity);

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
 *  @p event_type values are documented in docs/TELEMETRY-SCHEMA.md. @p value is event-specific. */
void
g2_telem_event(uint8_t device_id, uint64_t ts_ns, uint16_t event_type, float value);

#ifdef __cplusplus
}
#endif
