// Copyright 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Decoupled, adversarial unit tests for the constellation blob-detection
 *        layer (blobwatch). All tests synthesise frames with KNOWN ground truth
 *        and exercise blobwatch through its PUBLIC interface (blobwatch_process),
 *        so a wrong centroid / wrong uncertainty / wrong rejection fails rather
 *        than agreeing with itself. They cover:
 *          - saturated-blob sub-pixel centroid recovery vs the greysum plateau bias,
 *          - per-blob measurement-variance (R) behaviour (spread / saturation / edge /
 *            relative dimness inflation),
 *          - reflection/streak rejection while a clean LED is retained.
 */
#include "catch_amalgamated.hpp"

#include "internal/blobwatch.h"
#include "xrt/xrt_frame.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t W = 128, H = 128;
constexpr uint8_t PIX_THR = 8;     // matches the real G2 blob_min_threshold
constexpr uint8_t DETECT_THR = 24; // matches the real G2 blob_detect_threshold
constexpr uint8_t SAT = 255;

// A self-owned 8-bit greyscale frame (stride == width, single plane) for blobwatch_process.
struct TestFrame
{
	std::vector<uint8_t> buf;
	struct xrt_frame f = {};

	TestFrame(uint32_t w = W, uint32_t h = H) : buf((size_t)w * h, 0)
	{
		f.width = w;
		f.height = h;
		f.stride = w;
		f.size = buf.size();
		f.data = buf.data();
		f.format = XRT_FORMAT_L8;
		f.timestamp = 1;
		f.source_sequence = 1;
	}

	void
	clear()
	{
		std::fill(buf.begin(), buf.end(), (uint8_t)0);
	}

	uint8_t &
	at(int x, int y)
	{
		return buf[(size_t)y * f.width + x];
	}

	// Round Gaussian spot centred at (cx,cy) with peak `peak` and 1-sigma `sigma` (px).
	// Optionally clip at the saturation level to model a clipped plateau.
	void
	add_gaussian(double cx, double cy, double sigma, double peak, bool clip)
	{
		const int r = (int)std::ceil(4 * sigma) + 1;
		for (int y = (int)cy - r; y <= (int)cy + r; y++) {
			for (int x = (int)cx - r; x <= (int)cx + r; x++) {
				if (x < 0 || y < 0 || x >= (int)f.width || y >= (int)f.height)
					continue;
				const double dx = x - cx, dy = y - cy;
				double v = peak * std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
				if (clip && v > SAT)
					v = SAT;
				int iv = (int)std::lround(v);
				if (iv > at(x, y))
					at(x, y) = (uint8_t)(iv > 255 ? 255 : iv);
			}
		}
	}

	// Horizontal bright streak (window-edge / reflection shape): long, thin, flat.
	void
	add_streak(int x0, int x1, int y, int thick, uint8_t val)
	{
		for (int yy = y; yy < y + thick; yy++)
			for (int xx = x0; xx <= x1; xx++)
				if (xx >= 0 && yy >= 0 && xx < (int)f.width && yy < (int)f.height)
					at(xx, yy) = val;
	}
};

// Run blobwatch on a single frame and return its blobs via the public interface. The exposure/gain
// arguments only feed telemetry/frame-dump (not detection), so fixed nominal values are used here.
std::vector<blob>
detect(TestFrame &tf)
{
	blobwatch *bw = blobwatch_new(PIX_THR, DETECT_THR, 0);
	blobservation *ob = nullptr;
	blobwatch_process(bw, &tf.f, /*exposure=*/100, /*gain=*/0, &ob);
	std::vector<blob> out;
	if (ob != nullptr)
		out.assign(ob->blobs, ob->blobs + ob->num_blobs);
	blobwatch_free(bw);
	return out;
}

// Nearest blob to (x,y); asserts there is exactly one within `tol` so a duplicate/missing blob fails.
// Returns by value so it stays valid even when the caller passes a temporary blob vector.
blob
nearest(const std::vector<blob> &bs, double x, double y, double tol = 6.0)
{
	int found = -1, count = 0;
	for (size_t i = 0; i < bs.size(); i++) {
		double d = std::hypot(bs[i].x - x, bs[i].y - y);
		if (d < tol) {
			count++;
			found = (int)i;
		}
	}
	REQUIRE(count == 1);
	return bs[found];
}

// Plain intensity-weighted greysum centre over a window (the un-corrected estimate, for bias comparison).
void
greysum_center(TestFrame &tf, int x0, int y0, int x1, int y1, double &gx, double &gy)
{
	double s = 0, sx = 0, sy = 0;
	for (int y = y0; y <= y1; y++)
		for (int x = x0; x <= x1; x++) {
			double v = tf.at(x, y);
			s += v;
			sx += v * x;
			sy += v * y;
		}
	gx = sx / s;
	gy = sy / s;
}

} // namespace

TEST_CASE("blobwatch: unsaturated round LED is retained near its true centre")
{
	TestFrame tf;
	tf.add_gaussian(64.3, 48.7, 1.6, 200, /*clip=*/false);
	auto bs = detect(tf);
	REQUIRE(bs.size() == 1);
	const blob b = nearest(bs, 64.3, 48.7);
	REQUIRE(std::hypot(b.x - 64.3, b.y - 48.7) < 0.6);
	REQUIRE(b.pos_var_px2 > 0.0f); // always a finite, positive R
}

TEST_CASE("blobwatch: saturated centroid beats the greysum plateau bias (asymmetric skirt)")
{
	// A saturated spot with an asymmetric skirt: a real LED whose one side has extra glow. The clipped
	// plateau is symmetric about the TRUE centre, but the greysum is pulled toward the brighter skirt.
	const double tx = 64.0, ty = 64.0;
	TestFrame tf;
	tf.add_gaussian(tx, ty, 2.2, 600, /*clip=*/true); // big peak -> wide clipped plateau
	// Asymmetric extra glow on the right side only (below saturation), biasing a naive greysum right.
	for (int y = (int)ty - 2; y <= (int)ty + 2; y++)
		for (int x = (int)tx + 3; x <= (int)tx + 6; x++)
			tf.at(x, y) = (uint8_t)std::max<int>(tf.at(x, y), 120);

	auto bs = detect(tf);
	const blob b = nearest(bs, tx, ty, 10.0);

	double gx, gy;
	greysum_center(tf, (int)tx - 10, (int)ty - 10, (int)tx + 10, (int)ty + 10, gx, gy);

	const double err_fix = std::abs(b.x - tx);
	const double err_greysum = std::abs(gx - tx);
	INFO("edge-fit x=" << b.x << " greysum x=" << gx << " truth=" << tx);
	REQUIRE(err_fix < 0.75);             // contour centre recovers the truth
	REQUIRE(err_fix < err_greysum - 0.3); // and is materially better than the biased greysum
}

TEST_CASE("blobwatch: saturated blob carries larger R than a tight clean blob")
{
	TestFrame clean;
	clean.add_gaussian(40.0, 40.0, 1.2, 180, false);
	const blob bc = nearest(detect(clean), 40.0, 40.0);

	TestFrame sat;
	sat.add_gaussian(40.0, 40.0, 2.6, 700, true); // wide saturated plateau
	const blob bs = nearest(detect(sat), 40.0, 40.0, 12.0);

	INFO("clean R=" << bc.pos_var_px2 << " saturated R=" << bs.pos_var_px2);
	REQUIRE(bs.pos_var_px2 > bc.pos_var_px2);
}

TEST_CASE("blobwatch: a frame-edge blob is down-weighted (larger R) vs the same blob centred")
{
	TestFrame centred;
	centred.add_gaussian(64.0, 64.0, 1.6, 200, false);
	const blob bc = nearest(detect(centred), 64.0, 64.0);

	TestFrame edge;
	edge.add_gaussian(1.2, 64.0, 1.6, 200, false); // truncated by the left border
	auto bs = detect(edge);
	REQUIRE(bs.size() >= 1);
	// the only blob is the edge one
	const blob &be = bs[0];
	INFO("centred R=" << bc.pos_var_px2 << " edge R=" << be.pos_var_px2);
	REQUIRE(be.pos_var_px2 > bc.pos_var_px2);
}

TEST_CASE("blobwatch: a much dimmer candidate gets a larger R than the bright one (intensity rank)")
{
	TestFrame tf;
	tf.add_gaussian(30.0, 30.0, 1.6, 230, false); // bright = strong LED candidate
	tf.add_gaussian(90.0, 90.0, 1.6, 40, false);  // faint = weaker candidate (possible reflection)
	auto bs = detect(tf);
	const blob bright = nearest(bs, 30.0, 30.0);
	const blob faint = nearest(bs, 90.0, 90.0);
	INFO("bright R=" << bright.pos_var_px2 << " faint R=" << faint.pos_var_px2);
	REQUIRE(faint.pos_var_px2 > bright.pos_var_px2);
}

TEST_CASE("blobwatch: a flat bright reflection/streak is rejected while a clean LED is retained")
{
	TestFrame tf;
	tf.add_gaussian(30.0, 30.0, 1.6, 200, false); // real LED
	tf.add_streak(60, 110, 70, 2, 200);           // long thin flat bright patch (window edge / reflection)
	auto bs = detect(tf);
	// The clean LED survives...
	(void)nearest(bs, 30.0, 30.0);
	// ...and no blob is reported on the streak (rejected by aspect / fill / peak-to-mean).
	for (const blob &b : bs) {
		REQUIRE(!(b.x > 55 && b.x < 115 && b.y > 65 && b.y < 75));
	}
}
