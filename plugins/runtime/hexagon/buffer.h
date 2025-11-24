// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_BUFFER_H_
#define IREE_HAL_DRIVERS_HEXAGON_BUFFER_H_

#include <iree/base/config.h>
#include <iree/hal/buffer.h>

#include "hexagon/api.h"
#include "hexagon/mem_alloc.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_buffer_t
//===----------------------------------------------------------------------===//

// Wraps a {Qualcomm Hexagon} allocation in an iree_hal_buffer_t.
iree_status_t iree_hal_hexagon_buffer_wrap(
    iree_hal_hexagon_mem_alloc_t *alloc, iree_hal_buffer_placement_t placement,
    iree_hal_memory_type_t memory_type, iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_hexagon_device_t *device,
    iree_hal_buffer_t **out_buffer);

/// Return the impl_ptr from inside the buffer for accounting purposes.
/// The impl_ptr is used for accounting purposes. It just needs to be stable
/// (not change for the same buffer) and be different for each buffer (see
/// comment in iree_hal_hexagon_allocator_allocate_buffer() in allocator.c).
void *iree_hal_hexagon_buffer_impl_ptr(iree_hal_buffer_t *base_buffer);

#endif // IREE_HAL_DRIVERS_HEXAGON_BUFFER_H_
