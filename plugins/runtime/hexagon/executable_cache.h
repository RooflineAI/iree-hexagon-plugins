// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_EXECUTABLE_CACHE_H_
#define IREE_HAL_DRIVERS_HEXAGON_EXECUTABLE_CACHE_H_

#include "hexagon/rpc_types.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_executable_cache_t
//===----------------------------------------------------------------------===//

// Creates a no-op executable cache that does not cache at all.
// This is useful to isolate pipeline caching behavior and verify compilation
// behavior.
// TODO(hexagon): retain any shared resources (like device handles and symbols)
// that are needed to create executables.
iree_status_t iree_hal_hexagon_executable_cache_create(
    iree_string_view_t identifier, iree_allocator_t host_allocator,
    const iree_hal_hexagon_rpc_session_t *rpc_session,
    iree_hal_executable_cache_t **out_executable_cache);

#endif // IREE_HAL_DRIVERS_HEXAGON_EXECUTABLE_CACHE_H_
