// Copyright 2026, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Regression test for WMR controller inertial-sensor calibration selection.
 *
 * G2 controllers list two co-located IMUs of each type in their factory calibration:
 * the identified ICM-20602 (first) and a second "Undefined" entry with slightly
 * different extrinsics. The parser must keep the first of each type (matching the
 * Windows driver's first-match-by-type), not let the later duplicate overwrite it.
 */

#include "catch_amalgamated.hpp"

#include <string>
#include <vector>

#include "wmr/wmr_config.h"

static std::string
imu_sensor_json(const char *type, const char *id, float rot_marker, float trans_marker)
{
	std::string s = "{";
	s += "\"SensorType\":\"";
	s += type;
	s += "\",\"Id\":\"";
	s += id;
	s += "\",\"Rt\":{\"Rotation\":[";
	for (int i = 0; i < 9; i++) {
		s += std::to_string(rot_marker);
		if (i < 8) {
			s += ",";
		}
	}
	s += "],\"Translation\":[";
	for (int i = 0; i < 3; i++) {
		s += std::to_string(trans_marker);
		if (i < 2) {
			s += ",";
		}
	}
	s += "]},\"MixingMatrixTemperatureModel\":[";
	for (int i = 0; i < 36; i++) {
		s += "0.0";
		if (i < 35) {
			s += ",";
		}
	}
	s += "],\"BiasTemperatureModel\":[";
	for (int i = 0; i < 12; i++) {
		s += "0.0";
		if (i < 11) {
			s += ",";
		}
	}
	s += "],\"BiasUncertainty\":[0.001,0.001,0.001],";
	s += "\"Noise\":[0.001,0.001,0.001,0.0,0.0,0.0]}";
	return s;
}

TEST_CASE("wmr controller config keeps the first (ICM-20602) IMU over the Undefined duplicate")
{
	// ICM-20602 listed first (rotation marker 1.0 gyro / 2.0 accel); the co-located
	// "Undefined" duplicate second (markers 9.0 / 8.0).
	std::string json = "{\"CalibrationInformation\":{\"InertialSensors\":[";
	json += imu_sensor_json("CALIBRATION_InertialSensorType_Gyro", "CALIBRATION_InertialSensorId_ICM20602", 1.0f, 0.1f);
	json += ",";
	json += imu_sensor_json("CALIBRATION_InertialSensorType_Accelerometer", "CALIBRATION_InertialSensorId_ICM20602",
	                        2.0f, 0.2f);
	json += ",";
	json += imu_sensor_json("CALIBRATION_InertialSensorType_Gyro", "CALIBRATION_InertialSensorId_Undefined", 9.0f,
	                        0.9f);
	json += ",";
	json += imu_sensor_json("CALIBRATION_InertialSensorType_Accelerometer", "CALIBRATION_InertialSensorId_Undefined",
	                        8.0f, 0.8f);
	json += "],\"ControllerLeds\":[],\"Metadata\":{\"SerialId\":\"TESTUNIT\"}}}";

	std::vector<char> buf(json.begin(), json.end());
	buf.push_back('\0');

	struct wmr_controller_config cfg;
	REQUIRE(wmr_controller_config_parse(&cfg, buf.data(), U_LOGGING_WARN));

	// First-of-type (ICM-20602) wins for both gyro and accel.
	CHECK(cfg.sensors.gyro.present);
	CHECK(cfg.sensors.accel.present);
	CHECK(cfg.sensors.gyro.rotation.v[0] == 1.0f);
	CHECK(cfg.sensors.accel.rotation.v[0] == 2.0f);

	// Regression guard: the later "Undefined" duplicate must NOT have overwritten them.
	CHECK(cfg.sensors.gyro.rotation.v[0] != 9.0f);
	CHECK(cfg.sensors.accel.rotation.v[0] != 8.0f);
}
