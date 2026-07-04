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
#include "internal/static_map.h"
#include "math/m_api.h"
#include "xrt/xrt_frame.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t W = 128, H = 128;
constexpr uint8_t PIX_THR = 8; // matches the real G2 blob_min_threshold
constexpr uint8_t DIM_REF = 24; // dim-blob reference these retention tests assert against
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

	void
	fill(uint8_t value)
	{
		std::fill(buf.begin(), buf.end(), value);
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

// Run blobwatch on a single frame and return its blobs via the public interface. The exposure argument
// only feeds telemetry/frame-dump; gain re-denominates the dim-blob R inflation via the K gain law
// (0 = unknown = the gain-16 calibration point), so these tests default to the calibration point.
std::vector<blob>
detect_gain(TestFrame &tf, uint16_t gain)
{
	blobwatch *bw = blobwatch_new(PIX_THR, 0);
	blobservation *ob = nullptr;
	blobwatch_process(bw, &tf.f, /*exposure=*/100, gain, &ob);
	std::vector<blob> out;
	if (ob != nullptr)
		out.assign(ob->blobs, ob->blobs + ob->num_blobs);
	blobwatch_free(bw);
	return out;
}

std::vector<blob>
detect(TestFrame &tf)
{
	return detect_gain(tf, /*gain=*/0);
}

std::vector<blob>
detect_roi(TestFrame &tf, int x, int y, int w, int h)
{
	blobwatch *bw = blobwatch_new(PIX_THR, 0);
	blobservation *ob = nullptr;
	blobwatch_process_roi(bw, &tf.f, /*exposure=*/100, /*gain=*/0, x, y, w, h, &ob);
	std::vector<blob> out;
	if (ob != nullptr)
		out.assign(ob->blobs, ob->blobs + ob->num_blobs);
	blobwatch_free(bw);
	return out;
}

std::vector<blob>
detect_roi_lowthresh(TestFrame &tf, int x, int y, int w, int h)
{
	blobwatch *bw = blobwatch_new(PIX_THR, 0);
	blobservation *ob = nullptr;
	blobwatch_process_roi_lowthresh(bw, &tf.f, /*exposure=*/100, /*gain=*/0, x, y, w, h,
	                                /*roi_pixel_threshold=*/4, /*roi_adapt_margin=*/3, &ob);
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

TEST_CASE("blobwatch: non-zero ROI reports the same centroid as full-frame detection")
{
	TestFrame tf;
	tf.add_gaussian(82.2, 71.4, 1.5, 190, /*clip=*/false);

	const blob full = nearest(detect(tf), 82.2, 71.4);
	const blob roi = nearest(detect_roi(tf, 74, 63, 18, 18), 82.2, 71.4);

	INFO("full=(" << full.x << "," << full.y << ") roi=(" << roi.x << "," << roi.y << ")");
	REQUIRE(std::hypot(full.x - roi.x, full.y - roi.y) < 0.2);
}

TEST_CASE("blobwatch: ROI scans and finalizes the bottom row")
{
	TestFrame tf;
	for (int y = 71; y <= 72; y++) {
		tf.at(81, y) = 50;
		tf.at(82, y) = 100;
		tf.at(83, y) = 50;
	}

	auto bs = detect_roi(tf, 78, 64, 10, 9);
	const blob b = nearest(bs, 82.0, 71.5, 1.0);
	INFO("bottom-row ROI blob=(" << b.x << "," << b.y << ") count=" << bs.size());
	REQUIRE(std::abs(b.y - 71.5) < 0.35);
}

TEST_CASE("blobwatch: compact dim LED is retained inside an ROI")
{
	TestFrame tf;
	tf.at(63, 64) = 12;
	tf.at(64, 64) = 18;
	tf.at(65, 64) = 12;
	tf.at(64, 63) = 12;
	tf.at(64, 65) = 12;

	auto bs = detect_roi(tf, 56, 56, 17, 17);
	const blob b = nearest(bs, 64.0, 64.0, 1.0);
	REQUIRE(b.brightness < DIM_REF);
}

TEST_CASE("blobwatch: ROI recovery retains compact LEDs clipped by the adaptive margin")
{
	TestFrame tf;
	tf.fill(20);
	tf.at(64, 64) = 26;
	tf.at(63, 64) = 24;
	tf.at(65, 64) = 24;
	tf.at(64, 63) = 24;
	tf.at(64, 65) = 24;

	auto bs = detect_roi(tf, 56, 56, 17, 17);
	const blob b = nearest(bs, 64.0, 64.0, 1.0);
	REQUIRE(b.brightness < DIM_REF + 4);
}

TEST_CASE("blobwatch: ROI recovery rejects flat low-contrast patches")
{
	TestFrame tf;
	tf.fill(20);
	for (int y = 62; y <= 66; y++) {
		for (int x = 62; x <= 66; x++) {
			tf.at(x, y) = 25;
		}
	}

	auto bs = detect_roi(tf, 56, 56, 17, 17);
	REQUIRE(bs.empty());
}

TEST_CASE("blobwatch: low-threshold ROI retains shaped dim blobs above the ROI pixel floor")
{
	TestFrame tf;
	tf.add_gaussian(64.0, 64.0, 2.2, 10, /*clip=*/false);

	auto bs = detect_roi_lowthresh(tf, 52, 52, 25, 25);
	const blob b = nearest(bs, 64.0, 64.0, 2.0);
	REQUIRE(b.brightness < 12);
}

TEST_CASE("blobwatch: isolated one-pixel dim speck is rejected")
{
	TestFrame tf;
	tf.at(64, 64) = 18;
	auto bs = detect(tf);
	REQUIRE(bs.empty());
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
	sat.add_gaussian(40.0, 40.0, 2.1, 650, true);
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

TEST_CASE("blobwatch: a much dimmer candidate gets a larger R than the bright one (fixed dim anchor)")
{
	TestFrame tf;
	tf.add_gaussian(30.0, 30.0, 1.6, 230, false); // bright = strong LED candidate
	tf.add_gaussian(90.0, 90.0, 1.6, 20, false);  // faint = weaker candidate (possible reflection)
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

TEST_CASE("blobwatch: dim-blob R is a pure function of the blob, not the frame's brightest blob")
{
	// The same dim blob must carry the SAME measurement noise whether or not a brighter
	// (window/lamp) blob shares the frame — R decoupled from room content. This was the sole
	// live effect of the retired per-frame retention threshold and the verified mechanism of
	// the 16<->24 margin knife-edge.
	TestFrame alone;
	alone.add_gaussian(90.0, 90.0, 1.6, 20, false);
	alone.add_gaussian(30.0, 30.0, 1.6, 20, false);
	const blob b_alone = nearest(detect(alone), 90.0, 90.0);

	TestFrame with_clutter;
	with_clutter.add_gaussian(90.0, 90.0, 1.6, 20, false);
	with_clutter.add_gaussian(30.0, 30.0, 1.6, 230, false); // bright "window" blob
	const blob b_clutter = nearest(detect(with_clutter), 90.0, 90.0);

	INFO("alone R=" << b_alone.pos_var_px2 << " with-bright-clutter R=" << b_clutter.pos_var_px2);
	REQUIRE(b_alone.pos_var_px2 == b_clutter.pos_var_px2);
}

TEST_CASE("blobwatch: dim-noise K follows the commanded-gain law; gain 0/16 stay bit-identical")
{
	// The K gain law's regression tooth (B3): "unknown" (0) and the gain-16 calibration point must
	// yield EXACTLY the calibrated K16 = 40 so pre-gain-plumbing captures replay bit-identically,
	// while the production gain-32 point re-denominates K binary-exactly to the felt-room point
	// estimate K(2) = 65 (f_pre = 0.546875, both representable).
	REQUIRE(blobwatch_dim_noise_k(0) == 40.0f);
	REQUIRE(blobwatch_dim_noise_k(16) == 40.0f);
	REQUIRE(blobwatch_dim_noise_k(32) == 65.0f);
	REQUIRE(blobwatch_gain_multiplier(0) == 1.0f);
	REQUIRE(blobwatch_gain_multiplier(16) == 1.0f);
	REQUIRE(blobwatch_gain_multiplier(32) == 2.0f);
	REQUIRE(blobwatch_gain_multiplier(48) == 3.0f);

	// And through the public detection interface: R at gain 0 == gain 16 EXACTLY; at gain 32 the
	// same image-DN blob is physically dimmer, so its R rises by exactly the g(b) ratio of the two
	// K values (blob chosen in the uncapped band under both).
	TestFrame tf;
	tf.add_gaussian(90.0, 90.0, 1.6, 60, false);
	const blob b0 = nearest(detect_gain(tf, 0), 90.0, 90.0);
	const blob b16 = nearest(detect_gain(tf, 16), 90.0, 90.0);
	const blob b32 = nearest(detect_gain(tf, 32), 90.0, 90.0);
	REQUIRE(b0.pos_var_px2 == b16.pos_var_px2);
	const float bright = (float)b0.brightness;
	const float g16 = 1.0f + (40.0f / bright) * (40.0f / bright);
	const float g32 = 1.0f + (65.0f / bright) * (65.0f / bright);
	INFO("brightness=" << bright << " R0=" << b0.pos_var_px2 << " R32=" << b32.pos_var_px2);
	REQUIRE(b32.pos_var_px2 == Catch::Approx(b0.pos_var_px2 / g16 * g32).epsilon(1e-5));
}

TEST_CASE("blobwatch: cap-pressure retention keeps high-contrast blobs over scanline order")
{
	// 130 qualifying blobs: 30 dim ones FIRST in scanline order (top rows), 100 bright below.
	// The old code kept the first 100 in scanline order (all 30 dim + only 70 bright); priority
	// retention must keep all 100 bright ones and account for the dropped dim ones.
	TestFrame tf(320, 320);
	int placed_dim = 0, placed_bright = 0;
	for (int gy = 0; gy < 20 && placed_dim + placed_bright < 130; gy++) {
		for (int gx = 0; gx < 20 && placed_dim + placed_bright < 130; gx++) {
			const double x = 12.0 + gx * 15.0, y = 12.0 + gy * 15.0;
			if (placed_dim < 30) {
				tf.add_gaussian(x, y, 1.2, 30, false);
				placed_dim++;
			} else {
				tf.add_gaussian(x, y, 1.2, 200, false);
				placed_bright++;
			}
		}
	}
	REQUIRE(placed_dim == 30);
	REQUIRE(placed_bright == 100);

	blobwatch *bw = blobwatch_new(PIX_THR, 0);
	blobservation *ob = nullptr;
	blobwatch_process(bw, &tf.f, /*exposure=*/100, /*gain=*/0, &ob);
	REQUIRE(ob != nullptr);
	REQUIRE(ob->num_blobs == MAX_BLOBS_PER_FRAME);
	REQUIRE(ob->dropped_capacity == 30);
	int bright_kept = 0;
	for (int i = 0; i < ob->num_blobs; i++) {
		if (ob->blobs[i].brightness > 100) {
			bright_kept++;
		}
	}
	REQUIRE(bright_kept == 100);
	blobwatch_free(bw);
}

namespace {

// Distortion-free RADTAN8 camera (all coefficients zero) for static-map geometry tests.
camera_model
test_camera()
{
	camera_model cm = {};
	cm.width = 640;
	cm.height = 480;
	cm.calib.fx = 300.0f;
	cm.calib.fy = 300.0f;
	cm.calib.cx = 320.0f;
	cm.calib.cy = 240.0f;
	cm.calib.model = T_DISTORTION_OPENCV_RADTAN_8;
	return cm;
}

// One-blob observation at the projection of `world_point` through P_cam_world.
blobservation
observe_world_point(const camera_model &cm, const xrt_pose &P_cam_world, const xrt_vec3 &world_point)
{
	xrt_vec3 cam_pt;
	math_pose_transform_point(&P_cam_world, &world_point, &cam_pt);
	float u = 0, v = 0;
	REQUIRE(t_camera_models_project(&cm.calib, cam_pt.x, cam_pt.y, cam_pt.z, &u, &v));
	blobservation ob = {};
	ob.num_blobs = 1;
	ob.blobs[0].x = u;
	ob.blobs[0].y = v;
	ob.blobs[0].brightness = 30;
	return ob;
}

constexpr uint64_t FRAME_DT_NS = 50000000; // 20 Hz

} // namespace

TEST_CASE("static_map: a world-still spot under camera motion accumulates dwell and goes STATIC_CLUTTER")
{
	const camera_model cm = test_camera();
	static_map sm = {};
	const xrt_vec3 world_point = {0.3f, -0.2f, 2.5f};

	uint64_t ts = 1000000000;
	int frames_to_static = -1;
	for (int f = 0; f < 30; f++, ts += FRAME_DT_NS) {
		// Camera translates AND the spot stays put in the world: image position moves,
		// the world anchor does not.
		xrt_pose P_world_cam = XRT_POSE_IDENTITY;
		P_world_cam.position.x = 0.02f * (float)f; // 2 cm/frame lateral head motion
		xrt_pose P_cam_world;
		math_pose_invert(&P_world_cam, &P_cam_world);

		blobservation ob = observe_world_point(cm, P_cam_world, world_point);
		static_map_update(&sm, &ob, &cm, &P_world_cam, &P_cam_world, ts, nullptr, 0);

		if (ob.blobs[0].retention_class == BLOB_RETENTION_STATIC_CLUTTER && frames_to_static < 0) {
			frames_to_static = f;
		}
		if ((double)(ts - 1000000000) < STATIC_MAP_STATIC_DWELL_S * 1e9) {
			REQUIRE(ob.blobs[0].retention_class == BLOB_RETENTION_FRESH);
		}
	}
	// Dwell crosses 1 s at frame 20 (20 x 50 ms).
	REQUIRE(frames_to_static == 20);
	REQUIRE(sm.num_entries == 1);
}

TEST_CASE("static_map: the device exemption overrides static dwell (resting hands are never demoted)")
{
	const camera_model cm = test_camera();
	static_map sm = {};
	const xrt_vec3 world_point = {0.0f, 0.0f, 2.0f};
	const xrt_pose identity = XRT_POSE_IDENTITY;

	uint64_t ts = 1000000000;
	blobservation ob = {};
	for (int f = 0; f < 30; f++, ts += FRAME_DT_NS) {
		ob = observe_world_point(cm, identity, world_point);
		// A projected device LED sits within the exemption radius of the blob.
		const xrt_vec2 exempt = {ob.blobs[0].x + 10.0f, ob.blobs[0].y};
		static_map_update(&sm, &ob, &cm, &identity, &identity, ts, &exempt, 1);
		REQUIRE(ob.blobs[0].retention_class == BLOB_RETENTION_DEVICE_NEAR);
	}
	// Dwell still accumulated underneath: the entry is ready the moment the device moves away.
	REQUIRE(ob.blobs[0].static_dwell_s > STATIC_MAP_STATIC_DWELL_S);
}

TEST_CASE("static_map: an over-gap disappearance expires the entry and dwell restarts")
{
	const camera_model cm = test_camera();
	static_map sm = {};
	const xrt_vec3 world_point = {0.0f, 0.0f, 2.0f};
	const xrt_pose identity = XRT_POSE_IDENTITY;

	uint64_t ts = 1000000000;
	for (int f = 0; f < 25; f++, ts += FRAME_DT_NS) {
		blobservation ob = observe_world_point(cm, identity, world_point);
		static_map_update(&sm, &ob, &cm, &identity, &identity, ts, nullptr, 0);
		if (f == 24) {
			REQUIRE(ob.blobs[0].retention_class == BLOB_RETENTION_STATIC_CLUTTER);
		}
	}
	// Spot vanishes for longer than the gap tolerance (e.g. occluded), then returns.
	ts += STATIC_MAP_GAP_NS + 2 * FRAME_DT_NS;
	blobservation ob = observe_world_point(cm, identity, world_point);
	static_map_update(&sm, &ob, &cm, &identity, &identity, ts, nullptr, 0);
	REQUIRE(ob.blobs[0].retention_class == BLOB_RETENTION_FRESH);
	REQUIRE(ob.blobs[0].static_dwell_s == 0.0f);
}

TEST_CASE("static_map: a sub-gap flicker keeps the dwell accumulating")
{
	const camera_model cm = test_camera();
	static_map sm = {};
	const xrt_vec3 world_point = {0.1f, 0.1f, 3.0f};
	const xrt_pose identity = XRT_POSE_IDENTITY;

	uint64_t ts = 1000000000;
	blobservation ob = {};
	for (int f = 0; f < 30; f++, ts += FRAME_DT_NS) {
		if (f % 3 == 1 && f < 25) {
			continue; // detection flicker: the spot misses every third frame
		}
		ob = observe_world_point(cm, identity, world_point);
		static_map_update(&sm, &ob, &cm, &identity, &identity, ts, nullptr, 0);
	}
	REQUIRE(ob.blobs[0].retention_class == BLOB_RETENTION_STATIC_CLUTTER);
}
