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
