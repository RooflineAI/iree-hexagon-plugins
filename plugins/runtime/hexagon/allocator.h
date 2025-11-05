// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_HEXAGON_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_allocator_t
//===----------------------------------------------------------------------===//

// Creates a {Qualcomm Hexagon} buffer allocator used for persistent
// allocations.
iree_status_t
iree_hal_hexagon_allocator_create(iree_allocator_t host_allocator,
                                  iree_hal_allocator_t **out_allocator);

#endif // IREE_HAL_DRIVERS_HEXAGON_ALLOCATOR_H_
