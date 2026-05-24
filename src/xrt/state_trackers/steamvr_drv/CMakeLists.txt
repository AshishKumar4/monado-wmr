# Copyright 2020-2021, Collabora, Ltd.
# SPDX-License-Identifier: BSL-1.0

add_subdirectory(steamvr_bindings)

add_library(st_ovrd STATIC ovrd_driver.cpp ovrd_interface.h)

target_include_directories(st_ovrd INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
	st_ovrd
	PRIVATE
		xrt-interfaces
		xrt-external-openvr
		aux_math
		aux_generated_bindings_helpers
		ovrd_generated_bindings
	)

# X RandR is used in Init() to wait for the HMD's native display mode to
# enumerate before SteamVR's vrcompositor scans for a direct-mode display.
# XRT_HAVE_XCB is CMake-only, so expose it to the source as a compile define.
if(XRT_HAVE_XCB)
	target_compile_definitions(st_ovrd PRIVATE XRT_HAVE_XCB)
	target_include_directories(st_ovrd SYSTEM PRIVATE ${XCB_INCLUDE_DIRS})
	target_link_libraries(st_ovrd PRIVATE ${XCB_LIBRARIES})
endif()
