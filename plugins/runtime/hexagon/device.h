// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HEXAGON_DEVICE_H_
#define IREE_HAL_DRIVERS_HEXAGON_DEVICE_H_

#include "hexagon/api.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "rpc_types.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_device_t
//===----------------------------------------------------------------------===//

// Get the name of a domain from the ID.
//
// The string returned in *name is a pointer to a string literal, i.e., has
// global/infinite lifetime, no alloc, no free.
//
// Return 1 for success. Return 0 for failure / not found.
int iree_hal_hexagon_get_domain_name(iree_hal_hexagon_domain_id_t domain_id,
                                     const char **name);

// Get the name of a domain from the ID or "unknown" for unknown ID.
//
// The returned string is a pointer to a string literal, i.e., has
// global/infinite lifetime, no alloc, no free.
const char *iree_hal_hexagon_get_domain_name_or_unknown(
    iree_hal_hexagon_domain_id_t domain_id);

// Get the ID of a domain from a name
//
// Return 1 for success. Return 0 for failure / not found.
int iree_hal_hexagon_get_domain_id(iree_string_view_t name,
                                   iree_hal_hexagon_domain_id_t *domain_id);

// Fill a device info structure based on the domain.
//
// The name and path strings referenced by the device info structure have
// global/infinite lifetime, so those are not allocated and do not need to be
// freed.
void iree_hal_hexagon_fill_device_info(iree_hal_hexagon_domain_id_t domain_id,
                                       iree_hal_device_info_t *device_info);

// NOTE: nothing in the skeleton implementation. Device creation and adoption is
// part of the public API header. This header can contain internal types and
// functions.

iree_hal_hexagon_domain_id_t
iree_hal_hexagon_device_get_domain_id(iree_hal_hexagon_device_t *device);

rpc_session_handle_t iree_hal_hexagon_device_get_rpc_session_handle(
    iree_hal_hexagon_device_t *device);

#endif // IREE_HAL_DRIVERS_HEXAGON_DEVICE_H_
