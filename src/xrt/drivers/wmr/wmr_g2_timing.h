/*!
 * @file
 * @brief WMR G2 sensor-pair timing.
 * @ingroup drv_wmr
 */
#pragma once

/*! Constant controller-IMU vs headset-camera clock skew (ns), added to every camera-frame-derived
 * timestamp before it meets the fusion (optical folds, LED observations, gate and prior queries).
 * The OOSM absorbs variable processing lag; this is the residual fixed skew. Measured on the
 * 2026-06-11 live capture: rotation innovation is gyro-axis-aligned with a best-fit dt of
 * +4.7-4.9 ms on BOTH controllers, and applying +4.8 ms cuts median fast-rotation innovation
 * 23-37% with still-rate bins unchanged. Override per session with G2_CTRL_TD_NS. Shared with the
 * offline replay harness, which plays the controller-transport role and must stamp identically. */
#define WMR_CTRL_OPTICAL_TD_DEFAULT_NS (4800000)
