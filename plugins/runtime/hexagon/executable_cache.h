// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HEXAGON_EXECUTABLE_CACHE_H_
#define IREE_HAL_DRIVERS_HEXAGON_EXECUTABLE_CACHE_H_

#include "hexagon/serialize/rpc_types.h"
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
    rpc_session_handle_t rpc_session_handle,
    iree_hal_executable_cache_t **out_executable_cache);

#endif // IREE_HAL_DRIVERS_HEXAGON_EXECUTABLE_CACHE_H_
