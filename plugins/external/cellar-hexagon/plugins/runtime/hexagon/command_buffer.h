// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_HEXAGON_COMMAND_BUFFER_H_

#include "hexagon/serialize/rpc_types.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_command_buffer_t
//===----------------------------------------------------------------------===//

// Creates {Qualcomm Hexagon} command buffer.
iree_status_t iree_hal_hexagon_command_buffer_create(
    iree_hal_allocator_t *device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator, rpc_session_handle_t rpc_session_handle,
    iree_arena_block_pool_t *block_pool,
    uint32_t profiling_extra_records_per_dispatch,
    iree_hal_command_buffer_t **out_command_buffer);

// Returns true if |command_buffer| is a {Qualcomm Hexagon} command buffer.
bool iree_hal_hexagon_command_buffer_isa(
    iree_hal_command_buffer_t *base_command_buffer);

// Get RPC command buffer handle if possible.
iree_status_t iree_hal_hexagon_command_buffer_get_rpc_handle(
    iree_hal_command_buffer_t *base_command_buffer,
    rpc_command_buffer_handle_t *out_rpc_handle);

#endif // IREE_HAL_DRIVERS_HEXAGON_COMMAND_BUFFER_H_
