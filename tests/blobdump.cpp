// Decoupled detection-eval harness: run the REAL production blobwatch_process on a list of PGM frames
// and emit the detected blob centroids as JSON. This scores the C detector itself (not a Python proxy,
// not the matcher) against the agent-annotated LED GT in tools/led_dataset, so detection recall /
// precision can be measured authoritatively and apples-to-apples before/after a detector change.
//
//   blobdump <pix_thr> <req_thr> <out.json> <frame0.pgm> [frame1.pgm ...]
//
// Each frame is processed by a FRESH blobwatch (no cross-frame history), so the per-frame blob set is
// exactly what the first-frame detector would produce — the temporal blob-tracker in blobwatch_process
// only annotates ids/velocities and never adds or removes blobs, so a fresh instance gives the same
// detected set while keeping each frame independent (the GT frames are not a contiguous sequence).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

extern "C" {
#include "internal/blobwatch.h"
#include "xrt/xrt_frame.h"
}

int
main(int argc, char **argv)
{
	if (argc < 5) {
		fprintf(stderr, "usage: %s <pix_thr> <req_thr> <out.json> <frame.pgm> [frame.pgm ...]\n", argv[0]);
		return 2;
	}
	const uint8_t pix_thr = (uint8_t)atoi(argv[1]);
	const uint8_t req_thr = (uint8_t)atoi(argv[2]);
	FILE *out = fopen(argv[3], "wb");
	if (!out) {
		fprintf(stderr, "cannot open %s\n", argv[3]);
		return 2;
	}

	fprintf(out, "{\n");
	bool first = true;
	for (int a = 4; a < argc; a++) {
		const std::string path = argv[a];
		cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
		if (img.empty() || !img.isContinuous()) {
			fprintf(stderr, "skip (unreadable/non-contiguous): %s\n", path.c_str());
			continue;
		}

		struct xrt_frame f = {};
		f.width = (uint32_t)img.cols;
		f.height = (uint32_t)img.rows;
		f.stride = (size_t)img.step;
		f.size = (size_t)img.total();
		f.data = img.data;
		f.format = XRT_FORMAT_L8;
		f.timestamp = 1;
		f.source_sequence = 1;

		blobwatch *bw = blobwatch_new(pix_thr, req_thr, 0);
		blobservation *ob = NULL;
		blobwatch_process(bw, &f, 0, 0, &ob);

		// derive the GT tag from the filename stem (the eval keys on it)
		std::string stem = path;
		size_t slash = stem.find_last_of('/');
		if (slash != std::string::npos)
			stem = stem.substr(slash + 1);
		size_t dot = stem.find_last_of('.');
		if (dot != std::string::npos)
			stem = stem.substr(0, dot);

		if (!first)
			fprintf(out, ",\n");
		first = false;
		fprintf(out, "  \"%s\": [", stem.c_str());
		if (ob != NULL) {
			for (int i = 0; i < ob->num_blobs; i++) {
				const struct blob *b = &ob->blobs[i];
				fprintf(out, "%s[%.3f,%.3f,%u,%.4f]", i ? "," : "", b->x, b->y, (unsigned)b->brightness,
				        b->pos_var_px2);
			}
		}
		fprintf(out, "]");

		if (ob != NULL)
			blobwatch_release_observation(bw, ob);
		blobwatch_free(bw);
	}
	fprintf(out, "\n}\n");
	fclose(out);
	return 0;
}
