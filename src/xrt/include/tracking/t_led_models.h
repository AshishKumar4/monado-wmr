/*
 * LED constellation models
 * Copyright 2014-2015 Philipp Zabel
 * Copyright 2019-2023 Jan Schmidt
 * SPDX-License-Identifier:	LGPL-2.0+ or BSL-1.0
 */
#pragma once

#include <stdint.h>

#include "math/m_api.h"
#include "math/m_vec3.h"

/* This is the angle that Rift CV1 LEDs are visible at. Let's see if
 * it works for other controller types... */
#define LED_ANGLE 82

struct t_constellation_led
{
	// LED
	uint8_t id;
	// position relative to the device model in micrometers
	struct xrt_vec3 pos;
	// Normal
	struct xrt_vec3 dir;
	// LED radius in mm
	float radius_mm;
};

struct t_constellation_led_model
{
	// Device ID
	uint8_t id;

	struct t_constellation_led *leds;
	uint8_t num_leds;
};

void
t_constellation_led_model_init(uint8_t device_id, struct t_constellation_led_model *led_model, uint8_t num_leds);
void
t_constellation_led_model_dump(struct t_constellation_led_model *led_model, const char *desc);
void
t_constellation_led_model_clear(struct t_constellation_led_model *led_model);

struct t_constellation_search_led_candidate
{
	struct t_constellation_led *led;

	/* Transform to rotate the anchor LED to face forward @ 0,0,0 */
	struct xrt_pose pose;

	/* List of possible neighbours for this LED, sorted by distance */
	uint8_t num_neighbours;
	struct t_constellation_led **neighbours;
};

struct t_constellation_search_led_candidate *
t_constellation_search_led_candidate_new(struct t_constellation_led *anchor_led,
                                         struct t_constellation_led_model *led_model);
void
t_constellation_search_led_candidate_free(struct t_constellation_search_led_candidate *candidate);

struct t_constellation_search_model
{
	uint32_t id; /* Device ID */

	uint8_t num_points;
	struct t_constellation_search_led_candidate **points;

	struct t_constellation_led_model *led_model;
};

struct t_constellation_search_model *
t_constellation_search_model_new(struct t_constellation_led_model *led_model);
void
t_constellation_search_model_free(struct t_constellation_search_model *model);
