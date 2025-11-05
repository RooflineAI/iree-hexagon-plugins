// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_HEXAGON_SEMAPHORE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_semaphore_t
//===----------------------------------------------------------------------===//

// Creates a {Qualcomm Hexagon} semaphore used for ordering queue operations and
// synchronizing between host/device and device/device.
iree_status_t iree_hal_hexagon_semaphore_create(
    iree_hal_queue_affinity_t queue_affinity, uint64_t initial_value,
    iree_hal_semaphore_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_semaphore_t **out_semaphore);

#endif // IREE_HAL_DRIVERS_HEXAGON_SEMAPHORE_H_
