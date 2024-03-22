# Copyright 2022, Collabora, Ltd.
# SPDX-License-Identifier: BSL-1.0

if(XRT_MODULE_MERCURY_HANDTRACKING)
	add_subdirectory(hand)
endif()

if(XRT_MODULE_CONSTELLATION_TRACKING)
	add_subdirectory(constellation)
endif()
