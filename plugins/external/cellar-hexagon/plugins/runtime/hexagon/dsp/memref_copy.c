// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "memref_copy.h"
#include "malloc_free.h"

#include <stdint.h>
#include <string.h>

// This file is currently intended as a temporary fix and not a permanent
// solution. I would like to revisit this at some point and think about
// alternative solutions. This is currently not thought out enough.
// Additionally, hexagon-mlir's implementation looks dubious or unclear to me
// right now, so this should be analyzed further.

// Why this implementation exists:
//
// 1) MLIR/Hexagon lowering intentionally emits calls to `memrefCopy` for copy
//    patterns that are not lowered to contiguous memcpy/DMA. In particular,
//    strided/non-contiguous copies from lowered `memref.copy` /
//    `hexagonmem.copy` can reach LLVM as:
//      llvm.call @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
//    Note that contiguous cases are lowered to intrinsic llvm calls instead
//
// 2) In hexagon-mlir's launcher flow, that symbol is typically satisfied by
//    directly including a test utility implementation from:
//      test/utils/dsp/include/CRunnerUtils.cpp
//    The IREE Hexagon runtime path does not use that launcher wrapper and
//    therefore cannot rely on test utilities to provide this symbol.
//
// 3) The DSP runtime therefore owns a stable implementation so model execution
//    does not depend on external wrapper glue.
//
// The logic below mirrors the standard MLIR CRunnerUtils memref copy algorithm:
// copy element-by-element using rank/shape/stride metadata for unranked
// memrefs.

typedef struct {
  int64_t rank;
  void *descriptor;
} hexagon_unranked_memref_t;

// Strided memref header. The trailing descriptor memory contains:
//   sizes[rank] followed by strides[rank]
// The `sizes` member below is just the first element of that trailing storage.
typedef struct {
  char *basePtr;
  char *data;
  int64_t offset;
  int64_t sizes[1];
} hexagon_strided_memref_header_t;

typedef struct {
  int64_t rank;
  char *data;
  int64_t offset;
  const int64_t *sizes;
  const int64_t *strides;
} hexagon_dynamic_memref_t;

static int hexagon_unpack_dynamic_memref(const void *unrankedMemrefPtr,
                                         hexagon_dynamic_memref_t *out) {
  if (!unrankedMemrefPtr || !out) {
    return 0;
  }

  const hexagon_unranked_memref_t *unranked =
      (const hexagon_unranked_memref_t *)unrankedMemrefPtr;
  if (unranked->rank < 0 || !unranked->descriptor) {
    return 0;
  }

  const hexagon_strided_memref_header_t *desc =
      (const hexagon_strided_memref_header_t *)unranked->descriptor;
  out->rank = unranked->rank;
  out->data = desc->data;
  out->offset = desc->offset;
  out->sizes = out->rank == 0 ? NULL : desc->sizes;
  out->strides = out->rank == 0 ? NULL : (desc->sizes + out->rank);
  return 1;
}

void hexagon_runtime_memref_copy(int64_t elemSize, void *srcUnranked,
                                 void *dstUnranked) {
  hexagon_dynamic_memref_t src;
  hexagon_dynamic_memref_t dst;
  if (!hexagon_unpack_dynamic_memref(srcUnranked, &src) ||
      !hexagon_unpack_dynamic_memref(dstUnranked, &dst)) {
    return;
  }

  if (src.rank != dst.rank) {
    return;
  }

  // Handle empty shapes -> nothing to copy.
  for (int64_t dim = 0; dim < src.rank; ++dim) {
    if (src.sizes[dim] == 0) {
      return;
    }
  }

  char *srcPtr = src.data + src.offset * elemSize;
  char *dstPtr = dst.data + dst.offset * elemSize;

  if (src.rank == 0) {
    memcpy(dstPtr, srcPtr, (size_t)elemSize);
    return;
  }

  int64_t rank = src.rank;
  size_t bytes = (size_t)rank * sizeof(int64_t);
  int64_t *indices = (int64_t *)hexagon_runtime_malloc((int64_t)bytes);
  int64_t *srcStrides = (int64_t *)hexagon_runtime_malloc((int64_t)bytes);
  int64_t *dstStrides = (int64_t *)hexagon_runtime_malloc((int64_t)bytes);
  if (!indices || !srcStrides || !dstStrides) {
    hexagon_runtime_free(indices);
    hexagon_runtime_free(srcStrides);
    hexagon_runtime_free(dstStrides);
    return;
  }

  // Initialize multidimensional index and byte-scaled strides.
  for (int64_t dim = 0; dim < rank; ++dim) {
    indices[dim] = 0;
    srcStrides[dim] = src.strides[dim] * elemSize;
    dstStrides[dim] = dst.strides[dim] * elemSize;
  }

  int64_t readIndex = 0;
  int64_t writeIndex = 0;
  for (;;) {
    memcpy(dstPtr + writeIndex, srcPtr + readIndex, (size_t)elemSize);

    // Increment inner-most dimension and carry as needed.
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      int64_t newIndex = ++indices[axis];
      readIndex += srcStrides[axis];
      writeIndex += dstStrides[axis];

      if (src.sizes[axis] != newIndex) {
        break;
      }
      if (axis == 0) {
        hexagon_runtime_free(indices);
        hexagon_runtime_free(srcStrides);
        hexagon_runtime_free(dstStrides);
        return;
      }

      // Carry to outer dimension: reset current axis and undo linear advance.
      indices[axis] = 0;
      readIndex -= src.sizes[axis] * srcStrides[axis];
      writeIndex -= dst.sizes[axis] * dstStrides[axis];
    }
  }
}
