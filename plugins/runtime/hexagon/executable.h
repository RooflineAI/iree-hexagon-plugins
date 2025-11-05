// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_HEXAGON_EXECUTABLE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_executable_t
//===----------------------------------------------------------------------===//

// Creates a {Qualcomm Hexagon} executable from a binary in memory. Each
// executable may contain multiple entry points and be composed of several
// modules presented to the HAL as a single instance. See
// iree_hal_executable_params_t for more information about the lifetime of the
// resources referenced within.
iree_status_t iree_hal_hexagon_executable_create(
    const iree_hal_executable_params_t *executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t **out_executable);

#endif // IREE_HAL_DRIVERS_HEXAGON_EXECUTABLE_H_
