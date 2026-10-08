// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_RT_MEMREF_COPY_H
#define HEXAGON_DSP_RT_MEMREF_COPY_H

#include <stdint.h>

// Runtime implementation for MLIR's `memrefCopy` helper.
//
// Signature matches the helper declaration emitted by
// `LLVM::lookupOrCreateMemRefCopyFn`:
//   void memrefCopy(index elemSize, void* srcUnranked, void* dstUnranked)
//
// The `srcUnranked` and `dstUnranked` pointers are expected to point to
// MLIR-compatible unranked memref descriptors. Every `index` in the signature
// and in the descriptors has the pointer width, matching the compiler's index
// bitwidth, so it is an `intptr_t` here (unlike MLIR's CRunnerUtils, which
// assume a 64-bit index).
void hexagon_runtime_memref_copy(intptr_t elemSize, void *srcUnranked,
                                 void *dstUnranked);

#endif // HEXAGON_DSP_RT_MEMREF_COPY_H
