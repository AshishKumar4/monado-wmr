# Copyright 2018-2024, Collabora, Ltd.
# Copyright 2024-2026, NVIDIA CORPORATION.
#
# SPDX-License-Identifier: BSL-1.0

set(tests
    tests_cxx_wrappers
    tests_deque
    tests_generic_callbacks
    tests_glue_device
    tests_history_buf
    tests_id_ringbuffer
    tests_json
    tests_lowpass_float
    tests_lowpass_integer
    tests_pacing
    tests_quatexpmap
    tests_quat_change_of_basis
    tests_quat_swing_twist
    tests_rational
    tests_relation_chain
    tests_vector
    tests_worker
    tests_pose
    tests_vec3_angle
	)
if(XRT_HAVE_OPENCV)
	list(APPEND tests tests_kalman_fusion)
endif()
if(XRT_HAVE_D3D11)
	list(APPEND tests tests_aux_d3d_d3d11 tests_comp_client_d3d11)
endif()
if(XRT_HAVE_D3D12)
	list(APPEND tests tests_aux_d3d_d3d12 tests_comp_client_d3d12)
endif()
if(XRT_HAVE_VULKAN)
	list(APPEND tests tests_comp_client_vulkan tests_uv_to_tangent)
endif()
if(XRT_FEATURE_OPENXR)
	list(APPEND tests tests_input_transform)
endif()
if(XRT_HAVE_OPENGL
   AND XRT_HAVE_OPENGL_GLX
   AND XRT_HAVE_SDL2
   AND (WIN32 OR VK_USE_PLATFORM_XLIB_XRANDR_EXT)
	)
	set(_have_opengl_test ON)
	list(APPEND tests tests_comp_client_opengl)
endif()
if(XRT_BUILD_DRIVER_HANDTRACKING)
	list(APPEND tests tests_levenbergmarquardt)
endif()

foreach(testname ${tests})
	add_executable(${testname} ${testname}.cpp)
	target_link_libraries(${testname} PRIVATE xrt-external-catch2)
	target_link_libraries(${testname} PRIVATE aux_util)
	add_test(NAME ${testname} COMMAND ${testname} --success --allow-running-no-tests)
	set_target_properties(${testname} PROPERTIES FOLDER monado_tests)
endforeach()

# For tests that require more than just aux_util, link those other libs down here.

target_link_libraries(tests_cxx_wrappers PRIVATE xrt-interfaces)
target_link_libraries(tests_glue_device PRIVATE xrt-interfaces aux_glue)
target_link_libraries(tests_history_buf PRIVATE aux_math)
target_link_libraries(tests_lowpass_float PRIVATE aux_math)
target_link_libraries(tests_lowpass_integer PRIVATE aux_math)
target_link_libraries(tests_quatexpmap PRIVATE aux_math)
target_link_libraries(tests_rational PRIVATE aux_math)
target_link_libraries(tests_relation_chain PRIVATE aux_math)
target_link_libraries(tests_pose PRIVATE aux_math)
target_link_libraries(tests_quat_change_of_basis PRIVATE aux_math)
target_link_libraries(tests_quat_swing_twist PRIVATE aux_math)
target_link_libraries(tests_vec3_angle PRIVATE aux_math)

if(XRT_HAVE_OPENCV)
	target_link_libraries(tests_kalman_fusion PRIVATE aux_math aux_tracking Threads::Threads ${OpenCV_LIBRARIES})
	target_include_directories(tests_kalman_fusion PRIVATE ${OpenCV_INCLUDE_DIRS})
	# Path to the committed real-session .replay fixtures; the real-data corpus replay test loads
	# these at runtime and no-ops gracefully if the directory is absent.
	target_compile_definitions(tests_kalman_fusion PRIVATE G2_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
endif()

# Offline controller-VIO replay harness: drives the real constellation tracker + ESKF over recorded raw
# frames + controller IMU (for front-end iteration + IMU<->visual drift study, headless). A tool, not a
# ctest. Needs the constellation tracker (built with the WMR/rift_s constellation drivers) + OpenCV.
if(XRT_HAVE_OPENCV AND TARGET constellation_tracker)
	add_executable(offline_vio_replay offline_vio_replay.cpp)
	target_link_libraries(
		offline_vio_replay
		PRIVATE constellation_tracker aux_tracking aux_util aux_math aux_os xrt-interfaces
				Threads::Threads ${OpenCV_LIBRARIES})
	target_include_directories(offline_vio_replay PRIVATE ${OpenCV_INCLUDE_DIRS})
	set_target_properties(offline_vio_replay PROPERTIES FOLDER monado_tests)

	# Inertial coast-drift experiment: real ESKF on recorded IMU + optical, with withheld-optical
	# windows, to measure out-of-view dead-reckon drift (a tool, not a ctest).
	add_executable(eskf_coast_experiment eskf_coast_experiment.cpp)
	target_link_libraries(
		eskf_coast_experiment PRIVATE aux_math aux_tracking xrt-interfaces Threads::Threads
									  ${OpenCV_LIBRARIES})
	set_target_properties(eskf_coast_experiment PROPERTIES FOLDER monado_tests)

	# Decoupled unit tests for the constellation PnP mirror-twin enumeration (synthetic geometry,
	# known truth). A real ctest.
	add_executable(tests_constellation_pnp tests_constellation_pnp.cpp)
	target_link_libraries(
		tests_constellation_pnp PRIVATE xrt-external-catch2 constellation_tracker aux_math aux_util
										xrt-interfaces ${OpenCV_LIBRARIES})
	target_include_directories(
		tests_constellation_pnp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/xrt/tracking/constellation
										${OpenCV_INCLUDE_DIRS})
	add_test(NAME tests_constellation_pnp COMMAND tests_constellation_pnp --success --allow-running-no-tests)
	set_target_properties(tests_constellation_pnp PROPERTIES FOLDER monado_tests)
endif()

if(XRT_BUILD_DRIVER_HANDTRACKING)
	target_link_libraries(
		tests_levenbergmarquardt
		PRIVATE
			aux_math
			t_ht_mercury_includes
			t_ht_mercury_kine_lm_includes
			t_ht_mercury
			t_ht_mercury_kine_lm
		)
endif()

if(XRT_HAVE_D3D11)
	target_link_libraries(tests_aux_d3d_d3d11 PRIVATE aux_d3d)
	target_link_libraries(tests_comp_client_d3d11 PRIVATE comp_client comp_mock)
endif()

if(XRT_HAVE_D3D12)
	target_link_libraries(tests_aux_d3d_d3d12 PRIVATE aux_d3d)
	target_link_libraries(tests_comp_client_d3d12 PRIVATE comp_client comp_mock)
endif()

if(XRT_HAVE_VULKAN)
	target_link_libraries(
		tests_comp_client_vulkan PRIVATE comp_client comp_mock comp_util aux_vk
		)
	target_link_libraries(tests_uv_to_tangent PRIVATE comp_render)
endif()

if(XRT_FEATURE_OPENXR)
	target_link_libraries(
		tests_input_transform PRIVATE st_oxr xrt-interfaces xrt-external-openxr
		)
endif()
if(_have_opengl_test)
	target_link_libraries(
		tests_comp_client_opengl PRIVATE comp_client comp_mock aux_ogl SDL2::SDL2
		)
	target_include_directories(tests_comp_client_opengl PRIVATE SDL2::SDL2)
endif()

if(XRT_HAVE_VULKAN AND XRT_HAVE_D3D11)
	target_link_libraries(tests_aux_d3d_d3d11 PRIVATE comp_util aux_vk)
endif()

if(XRT_HAVE_VULKAN AND XRT_HAVE_D3D12)
	target_link_libraries(tests_aux_d3d_d3d12 PRIVATE comp_util aux_vk)
endif()
