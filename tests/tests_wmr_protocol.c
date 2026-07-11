// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Tests for the wmr_protocol.h byte readers, compiled in C mode.
 *
 * Regression coverage for the promoted-int shift UB family (audit 2026-07-10 R4#3):
 * read16/read24/read32 shifted device bytes as promoted `int`, which is undefined
 * behaviour in C11 whenever the top byte is >= 0x80 — i.e. on every negative IMU
 * sample. The readers must sign-extend exactly as the packed little-endian
 * two's-complement encoding demands, and must advance the cursor.
 */
#include "wmr/wmr_protocol.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

static int failures = 0;

#define CHECK_EQ(got, want, what)                                                                                      \
	do {                                                                                                           \
		int64_t g = (int64_t)(got);                                                                            \
		int64_t w = (int64_t)(want);                                                                           \
		if (g != w) {                                                                                          \
			fprintf(stderr, "FAIL %s: got %" PRId64 " want %" PRId64 "\n", what, g, w);                    \
			failures++;                                                                                    \
		}                                                                                                      \
	} while (0)

#define CHECK_EQ_U64(got, want, what)                                                                                  \
	do {                                                                                                           \
		uint64_t g = (uint64_t)(got);                                                                          \
		uint64_t w = (uint64_t)(want);                                                                         \
		if (g != w) {                                                                                          \
			fprintf(stderr, "FAIL %s: got %" PRIu64 " want %" PRIu64 "\n", what, g, w);                    \
			failures++;                                                                                    \
		}                                                                                                      \
	} while (0)

int
main(void)
{
	{
		const unsigned char buf[] = {0x7f, 0xff};
		const unsigned char *p = buf;
		CHECK_EQ(read8(&p), 0x7f, "read8 positive");
		CHECK_EQ(read8(&p), 0xff, "read8 high byte is unsigned");
		CHECK_EQ(p - buf, 2, "read8 advances");
	}

	{
		const unsigned char buf[] = {
		    0x34, 0x12, // 0x1234
		    0xff, 0xff, // -1
		    0x00, 0x80, // INT16_MIN
		    0xd6, 0xff, // -42
		};
		const unsigned char *p = buf;
		CHECK_EQ(read16(&p), 0x1234, "read16 positive");
		CHECK_EQ(read16(&p), -1, "read16 all-ones");
		CHECK_EQ(read16(&p), INT16_MIN, "read16 most-negative");
		CHECK_EQ(read16(&p), -42, "read16 negative");
		CHECK_EQ(p - buf, 8, "read16 advances");
	}

	{
		const unsigned char buf[] = {
		    0x56, 0x34, 0x12, // 0x123456
		    0xff, 0xff, 0xff, // -1
		    0x00, 0x00, 0x80, // -2^23
		    0xd6, 0xff, 0xff, // -42 (the negative-IMU-sample shape)
		    0xff, 0xff, 0x7f, // 2^23 - 1
		};
		const unsigned char *p = buf;
		CHECK_EQ(read24(&p), 0x123456, "read24 positive");
		CHECK_EQ(read24(&p), -1, "read24 all-ones");
		CHECK_EQ(read24(&p), -8388608, "read24 most-negative");
		CHECK_EQ(read24(&p), -42, "read24 negative");
		CHECK_EQ(read24(&p), 8388607, "read24 most-positive");
		CHECK_EQ(p - buf, 15, "read24 advances");
	}

	{
		const unsigned char buf[] = {
		    0x78, 0x56, 0x34, 0x12, // 0x12345678
		    0xff, 0xff, 0xff, 0xff, // -1
		    0x00, 0x00, 0x00, 0x80, // INT32_MIN
		    0xd6, 0xff, 0xff, 0xff, // -42
		};
		const unsigned char *p = buf;
		CHECK_EQ(read32(&p), 0x12345678, "read32 positive");
		CHECK_EQ(read32(&p), -1, "read32 all-ones");
		CHECK_EQ(read32(&p), INT32_MIN, "read32 most-negative");
		CHECK_EQ(read32(&p), -42, "read32 negative");
		CHECK_EQ(p - buf, 16, "read32 advances");
	}

	{
		const unsigned char buf[] = {
		    0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01, // 0x0123456789abcdef
		    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // UINT64_MAX
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, // 2^63
		};
		const unsigned char *p = buf;
		CHECK_EQ_U64(read64(&p), UINT64_C(0x0123456789abcdef), "read64 positive");
		CHECK_EQ_U64(read64(&p), UINT64_MAX, "read64 all-ones");
		CHECK_EQ_U64(read64(&p), UINT64_C(0x8000000000000000), "read64 msb");
		CHECK_EQ(p - buf, 24, "read64 advances");
	}

	if (failures != 0) {
		fprintf(stderr, "%d failure(s)\n", failures);
		return 1;
	}
	printf("tests_wmr_protocol: all checks passed\n");
	return 0;
}
