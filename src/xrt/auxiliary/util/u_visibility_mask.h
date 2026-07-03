// Copyright 2023, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Visibility mask utilitary header
 * @author Simon Zeni <simon.zeni@collabora.com>
 * @ingroup aux_util
 */

#include "xrt/xrt_defines.h"
#include "xrt/xrt_visibility_mask.h"


#ifdef __cplusplus
extern "C" {
#endif


/*!
 * Default visibility mask, only returns a very simple mask with four small
 * triangles in each corner, scaled to the given FoV so it matches the OpenXR
 * conventions. The caller must take care of de-allocating the mask once done
 * with it.
 *
 * @ingroup aux_util
 */
void
u_visibility_mask_get_default(enum xrt_visibility_mask_type type,
                              const struct xrt_fov *fov,
                              struct xrt_visibility_mask **out_mask);

/*!
 * Build a visibility mask from an ordered closed boundary of the visible
 * region, given in render texture UV space ([0,1]^2, 0,0 = upper left, the
 * same space @ref xrt_device::compute_distortion samples from). Points are
 * clamped to the unit square and the region must be star-shaped with respect
 * to its centroid. The hidden mesh fills the area between the boundary and
 * the unit square, the visible mesh fans the boundary from its centroid, and
 * together they exactly partition the unit square. Mask vertices are
 * converted to view tangent space using @p fov, following the
 * XR_KHR_visibility_mask convention (x right, y up). The caller must take
 * care of de-allocating the mask once done with it.
 *
 * @ingroup aux_util
 */
void
u_visibility_mask_from_uv_boundary(enum xrt_visibility_mask_type type,
                                   const struct xrt_vec2 *boundary_uv,
                                   uint32_t point_count,
                                   const struct xrt_fov *fov,
                                   struct xrt_visibility_mask **out_mask);


#ifdef __cplusplus
}
#endif
