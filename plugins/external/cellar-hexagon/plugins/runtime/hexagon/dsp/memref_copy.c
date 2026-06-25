// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "memref_copy.h"

#include "hexagon/arm_dsp/profiling.h"
#include "hexagon/dsp/profiling.h"

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

// Recursive N-dimensional copy kernel.
//
// At the innermost dimension (current_dim == rank - 1), when both source and
// destination have unit stride (contiguous elements), the entire inner row is
// copied with a single bulk memcpy.
//
// Outer dimensions loop and recurse regardless of stride
//
// Recursion depth equals the tensor rank.
static void copy_nd(const char *srcBase, char *dstBase, int64_t rank,
                    int64_t dim, const int64_t *sizes,
                    const int64_t *srcStrides, const int64_t *dstStrides,
                    int64_t elemSize) {
  if (dim == rank - 1) {
    // Innermost dimension.
    int64_t n = sizes[dim];
    if (srcStrides[dim] == 1 && dstStrides[dim] == 1) {
      memcpy(dstBase, srcBase, (size_t)(n * elemSize));
    } else {
      for (int64_t i = 0; i < n; ++i) {
        memcpy(dstBase + i * dstStrides[dim] * elemSize,
               srcBase + i * srcStrides[dim] * elemSize, (size_t)elemSize);
      }
    }
    return;
  }
  for (int64_t i = 0; i < sizes[dim]; ++i) {
    copy_nd(srcBase + i * srcStrides[dim] * elemSize,
            dstBase + i * dstStrides[dim] * elemSize, rank, dim + 1, sizes,
            srcStrides, dstStrides, elemSize);
  }
}

void hexagon_runtime_memref_copy(int64_t elemSize, void *srcUnranked,
                                 void *dstUnranked) {
  hexagon_runtime_profiling_zone_begin(MEMORY_MANAGEMENT, "memref_copy");
  hexagon_dynamic_memref_t src;
  hexagon_dynamic_memref_t dst;
  if (!hexagon_unpack_dynamic_memref(srcUnranked, &src) ||
      !hexagon_unpack_dynamic_memref(dstUnranked, &dst) ||
      src.rank != dst.rank) {
    hexagon_runtime_profiling_zone_end();
    return;
  }

  char *srcPtr = src.data + src.offset * elemSize;
  char *dstPtr = dst.data + dst.offset * elemSize;

  if (src.rank == 0) {
    memcpy(dstPtr, srcPtr, (size_t)elemSize);
    hexagon_runtime_profiling_zone_end();
    return;
  }

  for (int64_t dim = 0; dim < src.rank; ++dim) {
    if (src.sizes[dim] == 0) {
      hexagon_runtime_profiling_zone_end();
      return;
    }
  }

  copy_nd(srcPtr, dstPtr, src.rank, 0, src.sizes, src.strides, dst.strides,
          elemSize);

  hexagon_runtime_profiling_zone_end();
}
