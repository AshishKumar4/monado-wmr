#pragma once

#include "xrt/xrt_defines.h"

#include "pose_metrics.h"

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ASSOCIATION_MAX_BLOBS_PER_HYPOTHESIS 32
#define ASSOCIATION_MAX_HYPOTHESES_PER_DEVICE 32
#define ASSOCIATION_INVALID_INDEX INT16_MIN

enum association_hypothesis_source
{
	ASSOC_SOURCE_ABSENT = 0,
	ASSOC_SOURCE_PRIOR_POSE = 1,
	ASSOC_SOURCE_LAST_SEEN = 2,
	ASSOC_SOURCE_LABELLED_PNP = 3,
	ASSOC_SOURCE_PRIOR_LABELLED_PNP = 4,
	ASSOC_SOURCE_JOINT_PNP = 5,
	ASSOC_SOURCE_COLD_SEARCH = 6,
};

enum association_hypothesis_flags
{
	ASSOC_HYP_NONE = 0x0,
	ASSOC_HYP_HAS_POSE = 0x1,
	ASSOC_HYP_HAS_TWIN = 0x2,
	ASSOC_HYP_IS_TWIN = 0x4,
	ASSOC_HYP_PARTIAL_ONLY = 0x8,
	ASSOC_HYP_JOINT = 0x10,
};

enum association_state
{
	ASSOC_STATE_INERTIAL_SHORT_GAP = 0,
	ASSOC_STATE_VISUAL_LOCKED = 1,
	ASSOC_STATE_VISUAL_AMBIGUOUS = 2,
	ASSOC_STATE_BODY_LOCKED = 3,
	ASSOC_STATE_CONFUSED = 4,
};

struct association_blob_ref
{
	int16_t view_id;
	int16_t blob_id;
};

struct association_cost_terms
{
	float reprojection_nll;
	float missed_led_nll;
	float clutter_nll;
	float matched_evidence_nll;
	float position_prior_nll;
	float orientation_prior_nll;
	float head_anchor_nll;
	float body_state_nll;
	float orientation_consensus_nll;
	float temporal_nll;
	float total_nll;
};

struct association_pose_hypothesis
{
	uint8_t device_slot;
	uint8_t source;
	uint16_t flags;

	int16_t primary_view_id;
	uint8_t matched_count;
	uint8_t visible_count;
	uint8_t unmatched_count;
	struct association_blob_ref matched_blobs[ASSOCIATION_MAX_BLOBS_PER_HYPOTHESIS];
	int16_t matched_led_ids[ASSOCIATION_MAX_BLOBS_PER_HYPOTHESIS];

	float tilt_error_rad;
	bool tilt_valid;

	struct association_cost_terms cost;
	struct pose_metrics score;
	struct xrt_pose pose_cam;
	struct xrt_pose pose_world;
	struct xrt_pose pose_imu;
};

static inline struct association_blob_ref
association_blob_ref_invalid(void)
{
	struct association_blob_ref ref = {ASSOCIATION_INVALID_INDEX, ASSOCIATION_INVALID_INDEX};
	return ref;
}

static inline bool
association_blob_ref_is_valid(const struct association_blob_ref *ref)
{
	return ref != NULL && ref->view_id != ASSOCIATION_INVALID_INDEX && ref->blob_id != ASSOCIATION_INVALID_INDEX;
}

static inline bool
association_blob_ref_equal(const struct association_blob_ref *a, const struct association_blob_ref *b)
{
	return association_blob_ref_is_valid(a) && association_blob_ref_is_valid(b) && a->view_id == b->view_id &&
	       a->blob_id == b->blob_id;
}

static inline void
association_hypothesis_init(struct association_pose_hypothesis *hyp, uint8_t device_slot, uint8_t source)
{
	if (hyp == NULL) {
		return;
	}
	memset(hyp, 0, sizeof(*hyp));
	hyp->device_slot = device_slot;
	hyp->source = source;
	hyp->primary_view_id = ASSOCIATION_INVALID_INDEX;
}

static inline bool
association_hypothesis_add_blob(struct association_pose_hypothesis *hyp,
                                int16_t view_id,
                                int16_t blob_id,
                                int16_t led_id)
{
	if (hyp == NULL || hyp->matched_count >= ASSOCIATION_MAX_BLOBS_PER_HYPOTHESIS) {
		return false;
	}
	struct association_blob_ref ref = {view_id, blob_id};
	if (!association_blob_ref_is_valid(&ref) || led_id == ASSOCIATION_INVALID_INDEX) {
		return false;
	}
	for (uint8_t i = 0; i < hyp->matched_count; i++) {
		if (association_blob_ref_equal(&hyp->matched_blobs[i], &ref)) {
			return true;
		}
		if (hyp->matched_blobs[i].view_id == view_id && hyp->matched_led_ids[i] == led_id) {
			return true;
		}
	}
	hyp->matched_blobs[hyp->matched_count] = ref;
	hyp->matched_led_ids[hyp->matched_count] = led_id;
	hyp->matched_count++;
	return true;
}

static inline bool
association_hypotheses_share_blob(const struct association_pose_hypothesis *a,
                                  const struct association_pose_hypothesis *b)
{
	if (a == NULL || b == NULL) {
		return false;
	}
	for (uint8_t ai = 0; ai < a->matched_count; ai++) {
		for (uint8_t bi = 0; bi < b->matched_count; bi++) {
			if (association_blob_ref_equal(&a->matched_blobs[ai], &b->matched_blobs[bi])) {
				return true;
			}
		}
	}
	return false;
}

static inline uint8_t
association_hypotheses_shared_blob_count(const struct association_pose_hypothesis *a,
                                         const struct association_pose_hypothesis *b)
{
	uint8_t count = 0;
	if (a == NULL || b == NULL) {
		return count;
	}
	for (uint8_t ai = 0; ai < a->matched_count; ai++) {
		for (uint8_t bi = 0; bi < b->matched_count; bi++) {
			if (association_blob_ref_equal(&a->matched_blobs[ai], &b->matched_blobs[bi])) {
				count++;
			}
		}
	}
	return count;
}

static inline bool
association_joint_pair_is_compatible(const struct association_pose_hypothesis *a,
                                     const struct association_pose_hypothesis *b)
{
	if (a == NULL || b == NULL) {
		return true;
	}
	if (a->device_slot == b->device_slot) {
		return false;
	}
	return !association_hypotheses_share_blob(a, b);
}

#ifdef __cplusplus
}
#endif
