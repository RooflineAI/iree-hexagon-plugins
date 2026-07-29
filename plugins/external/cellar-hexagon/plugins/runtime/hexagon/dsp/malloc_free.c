// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "malloc_free.h"

#include "hexagon/arm_dsp/profiler.h"
#include "hexagon/dsp/profiler.h"
#include "hexagon/dsp/vtcm_pool.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include "HAP_farf.h"

void *hexagon_runtime_malloc(int64_t size) {
  if (size < 0 || (uint64_t)size > UINT_MAX) {
    FARF(ERROR, "HEXAGON-RUNTIME-ERROR: malloc failed for invalid size=%lld",
         (long long)size);
    return NULL;
  }

  hexagon_rt_prof_record_t *prof_rec = hexagon_runtime_profiler_zone_begin(
      MEMORY_MANAGEMENT, "kernel_allocation");
  void *res = hexagon_dsp_vtcm_pool_allocate(size);
  hexagon_runtime_profiler_zone_end(prof_rec);

  return res;
}

void hexagon_runtime_free(void *ptr) {
  if (!ptr) {
    return;
  }

  hexagon_rt_prof_record_t *prof_rec =
      hexagon_runtime_profiler_zone_begin(MEMORY_MANAGEMENT, "kernel_free");
  hexagon_dsp_vtcm_pool_free(ptr);
  hexagon_runtime_profiler_zone_end(prof_rec);
}
