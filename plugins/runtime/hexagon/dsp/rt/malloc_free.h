// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_RT_MALLOC_FREE_H
#define HEXAGON_DSP_RT_MALLOC_FREE_H

#include <stddef.h>

// Runtime implementation for codegen-issued `malloc`/`free` imports.
//
// These are intentionally kept separate from the DSP-side HAL import adapter
// layer, mirroring how `memrefCopy` is provided. The compiler passes the size
// as an `index`, which has the pointer width, i.e. a `size_t`.
void *hexagon_runtime_malloc(size_t size);
void hexagon_runtime_free(void *ptr);

#endif // HEXAGON_DSP_RT_MALLOC_FREE_H
