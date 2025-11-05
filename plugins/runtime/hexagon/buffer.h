// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_BUFFER_H_
#define IREE_HAL_DRIVERS_HEXAGON_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_buffer_t
//===----------------------------------------------------------------------===//

// Wraps a {Qualcomm Hexagon} allocation in an iree_hal_buffer_t.
iree_status_t iree_hal_hexagon_buffer_wrap(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t **out_buffer);

#endif // IREE_HAL_DRIVERS_HEXAGON_BUFFER_H_
