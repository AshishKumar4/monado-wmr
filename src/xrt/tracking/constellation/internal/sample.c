/* Copyright 2023, Jan Schmidt
 * SPDX-License-Identifier: BSL-1.0
 */
/*!
 * @file
 * @brief  Constellation tracking details for 1 exposure sample
 * @author Jan Schmidt <jan@centricular.com>
 * @ingroup drv_wmr
 */
#include "sample.h"

struct constellation_tracking_sample *
constellation_tracking_sample_new(void)
{
	struct constellation_tracking_sample *sample = calloc(1, sizeof(struct constellation_tracking_sample));
	return sample;
}

void
constellation_tracking_sample_free(struct constellation_tracking_sample *sample)
{
	int i;

	assert(sample->n_views <= CONSTELLATION_MAX_CAMERAS);
	for (i = 0; i < sample->n_views; i++) {
		struct tracking_sample_frame *view = sample->views + i;
		xrt_frame_reference(&view->vframe, NULL);
		if (view->bwobs != NULL) {
			assert(view->bw != NULL);
			blobwatch_release_observation(view->bw, view->bwobs);
		}
	}

	free(sample);
}
