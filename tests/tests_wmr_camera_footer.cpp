// Copyright 2026, G2-on-Linux project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Tests for wmr_camera_xfer_footer_parse — camera transfer footer validation.
 *
 * The garbage fixtures reproduce the only two recorded garbage-footer transfers
 * (results/forensics-20260709/REPORT.md ITEM B): structurally intact transfers — every pixel
 * chunk magic and the footer position check out — whose 26-byte footer region carried image
 * pixels instead of a footer. The parse boundary previously accepted them because the footer's
 * own "Dlo+" magic was skipped unchecked, so garbage timestamps entered tracking state and a
 * garbage frametype even selected the pipeline.
 */
#include "catch_amalgamated.hpp"

#include "wmr/wmr_protocol.h"

#include <array>
#include <cstdint>

namespace {

using footer_bytes = std::array<unsigned char, WMR_CAMERA_XFER_FOOTER_SIZE>;

void
write_le(footer_bytes &b, size_t offset, uint64_t value, size_t len)
{
	for (size_t i = 0; i < len; i++) {
		b[offset + i] = (unsigned char)((value >> (8 * i)) & 0xff);
	}
}

footer_bytes
make_valid_footer(uint64_t start_ticks, uint64_t end_ticks, uint16_t ctr1, uint16_t frametype)
{
	footer_bytes b{};
	write_le(b, 0, start_ticks, 8);
	write_le(b, 8, end_ticks, 8);
	write_le(b, 16, ctr1, 2);
	write_le(b, 18, 0, 2); // unknown0
	b[20] = 'D';
	b[21] = 'l';
	b[22] = 'o';
	b[23] = '+';
	write_le(b, 24, frametype, 2);
	return b;
}

} // namespace

TEST_CASE("wmr_camera_xfer_footer_parse accepts a valid footer and round-trips its fields")
{
	// Plausible mid-session values: device uptime ticks, one 11.1 ms (90 Hz) frame slot.
	const uint64_t start_ticks = UINT64_C(813400225689);
	const uint64_t end_ticks = start_ticks + 111000;

	for (uint16_t frametype : {uint16_t(0), uint16_t(2)}) {
		footer_bytes b = make_valid_footer(start_ticks, end_ticks, 88, frametype);
		struct wmr_camera_xfer_footer footer = {};
		REQUIRE(wmr_camera_xfer_footer_parse(b.data(), &footer));
		CHECK(footer.start_ts_ticks == start_ticks);
		CHECK(footer.end_ts_ticks == end_ticks);
		CHECK(footer.ctr1 == 88);
		CHECK(footer.unknown0 == 0);
		CHECK(footer.frametype == frametype);
	}
}

TEST_CASE("garbage footer of 20260611-180703 rows 0-3 is rejected (dark-pedestal pixels)")
{
	// The session's first transfer: footer region full of dark-image pixels at the background
	// pedestal (~4). The wire bytes at the start_ts position were recovered exactly from the
	// recorded frame.bin hw stamp 0x91919191912d2d90 = ticks * 100 mod 2^64 (forensics ITEM B).
	footer_bytes b{};
	const unsigned char pedestal[] = {0x04, 0x03, 0x03, 0x04, 0x04, 0x04, 0x04, 0x04};
	for (size_t i = 0; i < b.size(); i++) {
		b[i] = pedestal[i % sizeof(pedestal)];
	}

	struct wmr_camera_xfer_footer footer = {};
	REQUIRE(!wmr_camera_xfer_footer_parse(b.data(), &footer));

	// The parsed-anyway start_ts reproduces the recorded telemetry stamp byte-exactly, so the
	// G2_TELEM_EV_CAMERA_XFER_DROPPED event preserves the raw evidence verbatim.
	uint64_t recorded_hw_ns = footer.start_ts_ticks * (uint64_t)WMR_MS_HOLOLENS_NS_PER_TICK;
	CHECK(recorded_hw_ns == UINT64_C(0x91919191912d2d90));
}

TEST_CASE("garbage footer of the s3-leg2 mid-stream transfer is rejected (bright pixels)")
{
	// One full controller group ~1 s into the crash-grade launch-churn leg: footer region of
	// repeated bright pixels (0xd0/0xd1 forced by the recorded hw stamp's top bytes).
	footer_bytes b{};
	const unsigned char bright[] = {0xd0, 0xd1, 0xd1, 0xd0};
	for (size_t i = 0; i < b.size(); i++) {
		b[i] = bright[i % sizeof(bright)];
	}

	struct wmr_camera_xfer_footer footer = {};
	REQUIRE(!wmr_camera_xfer_footer_parse(b.data(), &footer));
}

TEST_CASE("a single corrupted magic byte is rejected")
{
	footer_bytes b = make_valid_footer(UINT64_C(813400225689), UINT64_C(813400336689), 88, 2);
	b[22] = 0x04; // "Dl\x04+"
	struct wmr_camera_xfer_footer footer = {};
	REQUIRE(!wmr_camera_xfer_footer_parse(b.data(), &footer));
}
