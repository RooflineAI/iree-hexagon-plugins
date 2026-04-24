// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_MALLOC_FREE_H
#define HEXAGON_DSP_MALLOC_FREE_H

#include <stdint.h>

// Runtime implementation for codegen-issued `malloc`/`free` imports.
//
// These are intentionally kept separate from the DSP-side HAL import adapter
// layer, mirroring how `memrefCopy` is provided.
void *hexagon_runtime_malloc(int64_t size);
void hexagon_runtime_free(void *ptr);

#endif // HEXAGON_DSP_MALLOC_FREE_H
