// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_MEMREF_COPY_H
#define HEXAGON_DSP_MEMREF_COPY_H

#include <stdint.h>

// Runtime implementation for MLIR's `memrefCopy` helper.
//
// Signature matches the helper declaration emitted by
// `LLVM::lookupOrCreateMemRefCopyFn`:
//   void memrefCopy(int64_t elemSize, void* srcUnranked, void* dstUnranked)
//
// The `srcUnranked` and `dstUnranked` pointers are expected to point to
// MLIR-compatible unranked memref descriptors.
void hexagon_runtime_memref_copy(int64_t elemSize, void *srcUnranked,
                                 void *dstUnranked);

#endif // HEXAGON_DSP_MEMREF_COPY_H
