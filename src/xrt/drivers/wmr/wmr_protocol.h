// Copyright 2018, Philipp Zabel.
// Copyright 2020-2021, N Madsen.
// Copyright 2020-2021, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  WMR and MS HoloLens protocol constants, structures and helpers header
 * @author Philipp Zabel <philipp.zabel@gmail.com>
 * @author nima01 <nima_zero_one@protonmail.com>
 * @ingroup drv_wmr
 */

#pragma once

#include "math/m_vec2.h"


#ifdef __cplusplus
extern "C" {
#endif


/*!
 * WMR and MS HoloLens Sensors protocol constants and structures
 *
 * @addtogroup drv_wmr
 * @{
 */

#define WMR_FEATURE_BUFFER_SIZE 497
#define WMR_MS_HOLOLENS_NS_PER_TICK 100

// Messages types specific to WMR Hololens Sensors devices
#define WMR_MS_HOLOLENS_MSG_SENSORS 0x01
#define WMR_MS_HOLOLENS_MSG_CONTROL 0x02 // Firmware read control responses
#define WMR_MS_HOLOLENS_MSG_DEBUG 0x03
#define WMR_MS_HOLOLENS_MSG_BT_IFACE 0x05         /* Bluetooth interface */
#define WMR_MS_HOLOLENS_MSG_LEFT_CONTROLLER 0x06  /* Left controller */
#define WMR_MS_HOLOLENS_MSG_RIGHT_CONTROLLER 0x0E /* Right controller */
#define WMR_MS_HOLOLENS_MSG_BT_CONTROL 0x16       /* BT control message on Reverb G2 & Odyssey+ */
#define WMR_MS_HOLOLENS_MSG_CONTROLLER_STATUS 0x17

// Messages types specific to WMR Hololens Sensors' companion devices
#define WMR_CONTROL_MSG_IPD_VALUE 0x01
#define WMR_CONTROL_MSG_UNKNOWN_02 0x02    // Seen in relation to proximity events on Reverb G1
#define WMR_CONTROL_MSG_DEVICE_STATUS 0x05 // Seen in relation screen state changes on Reverb G1

// Message sub-types for WMR_MS_HOLOLENS_MSG_BT_IFACE WMR Hololens Sensors message
#define WMR_BT_IFACE_MSG_DEBUG 0x19

// Controller status codes for WMR_MS_HOLOLENS_MSG_CONTROLLER_STATUS status message
#define WMR_CONTROLLER_STATUS_UNPAIRED 0x0
#define WMR_CONTROLLER_STATUS_OFFLINE 0x1
#define WMR_CONTROLLER_STATUS_ONLINE 0x2

/* Messages we can send the G2 via WMR_MS_HOLOLENS_MSG_BT_CONTROL */
enum wmr_bt_control_msg
{
	WMR_BT_CONTROL_MSG_ONLINE_STATUS = 0x04,
	WMR_BT_CONTROL_MSG_PAIR = 0x05,
	WMR_BT_CONTROL_MSG_UNPAIR = 0x06,
	WMR_BT_CONTROL_MSG_PAIRING_STATUS = 0x08,
	WMR_BT_CONTROL_MSG_CMD_STATUS = 0x09,
};

#define STR_TO_U32(s) ((uint32_t)(((s)[0]) | ((s)[1] << 8) | ((s)[2] << 16) | ((s)[3] << 24)))
#define WMR_MAGIC STR_TO_U32("Dlo+")

#define WMR_MIN_EXPOSURE 60
#define WMR_MAX_OBSERVED_EXPOSURE 6000
#define WMR_MAX_EXPOSURE 9000
#define WMR_MIN_GAIN 16
#define WMR_MAX_GAIN 255

static const unsigned char hololens_sensors_imu_on[64] = {0x02, 0x07};


struct hololens_sensors_packet
{
	uint8_t id;
	uint16_t temperature[4];
	uint64_t gyro_timestamp[4];
	int16_t gyro[3][4 * 8];
	uint64_t accel_timestamp[4];
	int32_t accel[3][4];
	uint64_t video_timestamp[4];
};

struct wmr_config_header
{
	uint32_t json_start;
	uint32_t json_size;
	char manufacturer[0x40];
	char device[0x40];
	char serial[0x40];
	char uid[0x26];
	char unk[0xd5];
	char name[0x40];
	char revision[0x20];
	char revision_date[0x20];
};

/*!
 * @}
 */


/*!
 * WMR and MS HoloLens Sensors protocol helpers
 *
 * @addtogroup drv_wmr
 * @{
 */

void
vec3_from_hololens_accel(int32_t sample[3][4], int i, float scale, struct xrt_vec3 *out_vec);

void
vec3_from_hololens_gyro(int16_t sample[3][32], int i, struct xrt_vec3 *out_vec);


static inline uint8_t
read8(const unsigned char **buffer)
{
	uint8_t ret = **buffer;
	*buffer += 1;
	return ret;
}

static inline int16_t
read16(const unsigned char **buffer)
{
	uint16_t ret = ((uint16_t) * (*buffer + 0) << 0) | //
	               ((uint16_t) * (*buffer + 1) << 8);
	*buffer += 2;
	return (int16_t)ret;
}

static inline int32_t
read24(const unsigned char **buffer)
{
	// Note: Preserve sign by shifting up to write MSB
	uint32_t ret = ((uint32_t) * (*buffer + 0) << 8) |  //
	               ((uint32_t) * (*buffer + 1) << 16) | //
	               ((uint32_t) * (*buffer + 2) << 24);
	*buffer += 3;

	// restore 24 bit scale again
	return (int32_t)ret >> 8;
}

static inline int32_t
read32(const unsigned char **buffer)
{
	uint32_t ret = ((uint32_t) * (*buffer + 0) << 0) |  //
	               ((uint32_t) * (*buffer + 1) << 8) |  //
	               ((uint32_t) * (*buffer + 2) << 16) | //
	               ((uint32_t) * (*buffer + 3) << 24);
	*buffer += 4;
	return (int32_t)ret;
}

static inline uint64_t
read64(const unsigned char **buffer)
{
	uint64_t ret = ((uint64_t) * (*buffer + 0) << 0) |  //
	               ((uint64_t) * (*buffer + 1) << 8) |  //
	               ((uint64_t) * (*buffer + 2) << 16) | //
	               ((uint64_t) * (*buffer + 3) << 24) | //
	               ((uint64_t) * (*buffer + 4) << 32) | //
	               ((uint64_t) * (*buffer + 5) << 40) | //
	               ((uint64_t) * (*buffer + 6) << 48) | //
	               ((uint64_t) * (*buffer + 7) << 56);
	*buffer += 8;
	return ret;
}

/*
 * Camera bulk-transfer footer.
 */

#define WMR_CAMERA_XFER_FOOTER_SIZE 26

//! frametype values of the camera transfer footer.
#define WMR_FRAMETYPE_SLAM 0x0       //!< Long-exposure head-tracking (SLAM) frame
#define WMR_FRAMETYPE_CONTROLLER 0x2 //!< Short-exposure controller constellation frame

/*!
 * The 26-byte footer that trails the pixel chunks of a camera bulk transfer:
 *   __le64 start_ts;  - exposure start, 100 ns device-clock ticks (same clock as the IMU feed)
 *   __le64 end_ts;    - always ~111000 ticks (11.1 ms, 90 Hz frame slot) after start_ts
 *   __le16 ctr1;      - counter that increments by 88 (sometimes 96) and wraps at 16384
 *   __le16 unknown0;  - has only ever been 0
 *   __be32 magic;     - "Dlo+"
 *   __le16 frametype; - WMR_FRAMETYPE_SLAM or WMR_FRAMETYPE_CONTROLLER
 */
struct wmr_camera_xfer_footer
{
	uint64_t start_ts_ticks;
	uint64_t end_ts_ticks;
	uint16_t ctr1;
	uint16_t unknown0;
	uint16_t frametype;
};

/*!
 * Parse a camera transfer footer, validating its magic. Returns false when the magic does not
 * match: under stream disturbance the device emits structurally intact transfers (chunk magics
 * and footer position check out) whose footer region carries image pixels instead of a footer,
 * so every field — the timestamps AND the pipeline-selecting frametype — is garbage
 * (results/forensics-20260709/REPORT.md ITEM B). @p out is filled either way so a caller can
 * record the offending raw values.
 */
static inline bool
wmr_camera_xfer_footer_parse(const unsigned char *buf, struct wmr_camera_xfer_footer *out)
{
	out->start_ts_ticks = read64(&buf);
	out->end_ts_ticks = read64(&buf);
	out->ctr1 = (uint16_t)read16(&buf);
	out->unknown0 = (uint16_t)read16(&buf);
	uint32_t magic = (uint32_t)read32(&buf);
	out->frametype = (uint16_t)read16(&buf);
	return magic == WMR_MAGIC;
}

/*!
 * @}
 */


#ifdef __cplusplus
}
#endif
