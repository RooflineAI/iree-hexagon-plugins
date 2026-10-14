// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "malloc_free.h"

#include "hexagon/dsp/rt/vtcm_pool.h"

#include <stddef.h>

// Profiler is inserted by the compiler as marker ops around the memref
// operations that lower to these helpers, so these
// helpers no longer emit their own profiler zones.

void *hexagon_runtime_malloc(size_t size) {
  return hexagon_dsp_vtcm_pool_allocate(size);
}

void hexagon_runtime_free(void *ptr) {
  if (!ptr) {
    return;
  }

  hexagon_dsp_vtcm_pool_free(ptr);
}
