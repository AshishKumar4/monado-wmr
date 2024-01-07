// Copyright 2020-2021, N Madsen.
// Copyright 2020-2021, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  WMR Motion Controller protocol constants, structures and helpers
 * @author Nis Madsen <nima_zero_one@protonmail.com>
 * @ingroup drv_wmr
 */

#pragma once

#include "xrt/xrt_byte_order.h"

#include "wmr_protocol.h"


#ifdef __cplusplus
extern "C" {
#endif

#ifdef XRT_DOXYGEN
#define WMR_PACKED
#else
#define WMR_PACKED __attribute__((packed))
#endif

/*!
 * WMR Motion Controller protocol constant and structures
 *
 * @addtogroup drv_wmr
 * @{
 */

// Todo: Is this enough?
#define WMR_MOTION_CONTROLLER_MSG_BUFFER_SIZE 256
#define WMR_MOTION_CONTROLLER_NS_PER_TICK 100


// Messages types for WMR motion controllers
#define WMR_MOTION_CONTROLLER_STATUS_MSG 0x01
#define WMR_MOTION_CONTROLLER_FW_TASK 0x02     /* FW replies, but also sent for FW writes */
#define WMR_MOTION_CONTROLLER_LED_CONTROL 0x03 /* Sent to control LED brightness / timing */
#define WMR_MOTION_CONTROLLER_HAPTIC_BUZZ 0x04 /* Turn on / off haptic motors */
#define WMR_MOTION_CONTROLLER_KEEPALIVE 0x05   /* Regularly sent Keepalives */
#define WMR_MOTION_CONTROLLER_CONTROL 0x06     /* Various control: FW read, telemetry enable, clock reset */

struct wmr_controller_fw_cmd
{
	union {
		struct
		{
			uint8_t prefix;
			uint8_t cmd_id;
			uint8_t block_id;

			__le32 addr;
		} WMR_PACKED cmd;
		uint8_t buf[64];
	};
};

#define WMR_CONTROLLER_FW_CMD_INIT(p, c, b, a)                                                                         \
	((struct wmr_controller_fw_cmd){                                                                               \
	    .cmd = {.prefix = (p), .cmd_id = (c), .block_id = (b), .addr = __cpu_to_le32((a))}})

struct wmr_controller_fw_cmd_response
{
	union {
		struct
		{
			uint8_t prefix;
			uint8_t zero;
			uint8_t cmd_id_echo;
			uint8_t zero1;
			uint8_t block_id_echo;

			__le32 blk_remain; /* Remaining bytes available in the block */
			uint8_t len;       /* Bytes in this response data */

			uint8_t data[68];
		} WMR_PACKED response;
		uint8_t buf[78];
	};
};

/*!
 * @}
 */

#undef WMR_PACKED

#ifdef __cplusplus
}
#endif
