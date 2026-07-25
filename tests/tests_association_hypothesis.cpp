#include "internal/association_hypothesis.h"

#include "catch_amalgamated.hpp"

TEST_CASE("association hypotheses reject shared final blobs")
{
	association_pose_hypothesis left{};
	association_pose_hypothesis right{};
	association_hypothesis_init(&left, 0, ASSOC_SOURCE_PRIOR_LABELLED_PNP);
	association_hypothesis_init(&right, 1, ASSOC_SOURCE_PRIOR_LABELLED_PNP);

	REQUIRE(association_hypothesis_add_blob(&left, 0, 10, 3));
	REQUIRE(association_hypothesis_add_blob(&left, 0, 11, 4));
	REQUIRE(association_hypothesis_add_blob(&right, 0, 20, 3));
	REQUIRE(association_hypothesis_add_blob(&right, 0, 21, 4));
	CHECK(association_joint_pair_is_compatible(&left, &right));

	REQUIRE(association_hypothesis_add_blob(&right, 0, 10, 5));
	CHECK_FALSE(association_joint_pair_is_compatible(&left, &right));
	CHECK(association_hypotheses_shared_blob_count(&left, &right) == 1);
}

TEST_CASE("rival claims on one cluster are decided by position, not by a shared blob")
{
	// The measured separation the identity-ambiguity deferral rests on. A shared blob answers
	// neither direction: two adjacent controllers contend for one boundary blob 10.2 cm apart
	// (xv1 t=52.122 s, 1 of 16) while two poses on ONE ring split its blobs between them
	// (xv1 t=60.996 s, 1 of 5, 2.1 cm apart).
	const float margin_m = 0.06f;
	association_pose_hypothesis mine{};
	association_hypothesis_init(&mine, 0, ASSOC_SOURCE_COLD_SEARCH);
	mine.flags |= ASSOC_HYP_HAS_POSE;
	mine.pose_world.position = xrt_vec3{-0.462f, 0.201f, -0.796f};
	for (int16_t blob = 0; blob < 5; blob++) {
		REQUIRE(association_hypothesis_add_blob(&mine, 0, blob, blob));
	}

	// One shared blob, 2.1 cm away: the same ring, split between two devices.
	association_pose_hypothesis rival{};
	association_hypothesis_init(&rival, 1, ASSOC_SOURCE_PRIOR_LABELLED_PNP);
	rival.flags |= ASSOC_HYP_HAS_POSE;
	rival.pose_world.position = xrt_vec3{-0.446f, 0.202f, -0.783f};
	REQUIRE(association_hypothesis_add_blob(&rival, 0, 4, 0));
	CHECK(association_hypotheses_shared_blob_count(&rival, &mine) == 1);
	CHECK(association_hypotheses_same_cluster(&rival, &mine, margin_m));

	// One shared blob, 10.2 cm away: two controllers contending for a boundary blob.
	association_pose_hypothesis neighbour{};
	association_hypothesis_init(&neighbour, 1, ASSOC_SOURCE_COLD_SEARCH);
	neighbour.flags |= ASSOC_HYP_HAS_POSE;
	neighbour.pose_world.position = xrt_vec3{-0.361f, 0.199f, -0.766f};
	REQUIRE(association_hypothesis_add_blob(&neighbour, 0, 4, 1));
	CHECK(association_hypotheses_shared_blob_count(&neighbour, &mine) == 1);
	CHECK_FALSE(association_hypotheses_same_cluster(&neighbour, &mine, margin_m));

	// A poseless or absent claim is never a rival explanation of a cluster.
	association_pose_hypothesis poseless = rival;
	poseless.flags &= (uint16_t)~ASSOC_HYP_HAS_POSE;
	CHECK_FALSE(association_hypotheses_same_cluster(&poseless, &mine, margin_m));
	CHECK_FALSE(association_hypotheses_same_cluster(nullptr, &mine, margin_m));
	CHECK_FALSE(association_hypotheses_same_cluster(&rival, nullptr, margin_m));
}

TEST_CASE("association blob identity includes camera view")
{
	association_pose_hypothesis left{};
	association_pose_hypothesis right{};
	association_hypothesis_init(&left, 0, ASSOC_SOURCE_LABELLED_PNP);
	association_hypothesis_init(&right, 1, ASSOC_SOURCE_LABELLED_PNP);

	REQUIRE(association_hypothesis_add_blob(&left, 0, 7, 1));
	REQUIRE(association_hypothesis_add_blob(&right, 1, 7, 1));
	CHECK(association_joint_pair_is_compatible(&left, &right));
}

TEST_CASE("association hypothesis keeps blob and led assignments unique")
{
	association_pose_hypothesis hyp{};
	association_hypothesis_init(&hyp, 0, ASSOC_SOURCE_COLD_SEARCH);

	REQUIRE(association_hypothesis_add_blob(&hyp, 2, 5, 8));
	REQUIRE(association_hypothesis_add_blob(&hyp, 2, 5, 8));
	CHECK(hyp.matched_count == 1);

	REQUIRE(association_hypothesis_add_blob(&hyp, 2, 6, 8));
	CHECK(hyp.matched_count == 1);

	REQUIRE(association_hypothesis_add_blob(&hyp, 2, 6, 9));
	CHECK(hyp.matched_count == 2);

	REQUIRE(association_hypothesis_add_blob(&hyp, 3, 1, 8));
	CHECK(hyp.matched_count == 3);
}

TEST_CASE("same-device hypotheses are not a final joint pair")
{
	association_pose_hypothesis primary{};
	association_pose_hypothesis twin{};
	association_hypothesis_init(&primary, 0, ASSOC_SOURCE_LABELLED_PNP);
	association_hypothesis_init(&twin, 0, ASSOC_SOURCE_LABELLED_PNP);

	CHECK_FALSE(association_joint_pair_is_compatible(&primary, &twin));
}
