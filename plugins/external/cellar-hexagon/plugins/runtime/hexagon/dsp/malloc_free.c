// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "malloc_free.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include "HAP_farf.h"
#include "HAP_mem.h"

void *hexagon_runtime_malloc(int64_t size) {
  // Avoid relying on target-specific malloc(0) behavior.
  if (size == 0) {
    size = 1;
  }
  if (size < 0 || (uint64_t)size > UINT_MAX) {
    FARF(ERROR, "HEXAGON-RUNTIME-ERROR: malloc failed for invalid size=%lld",
         (long long)size);
    return NULL;
  }

  void *ptr = NULL;
  int err = HAP_malloc((unsigned int)size, &ptr);
  if (err != 0 || !ptr) {
    FARF(ERROR,
         "HEXAGON-RUNTIME-ERROR: HAP_malloc failed for size=%lld (err=0x%x, "
         "ptr=%p)",
         (long long)size, err, ptr);
    return NULL;
  }
  return ptr;
}

void hexagon_runtime_free(void *ptr) {
  if (!ptr) {
    return;
  }
  HAP_free(ptr);
}
