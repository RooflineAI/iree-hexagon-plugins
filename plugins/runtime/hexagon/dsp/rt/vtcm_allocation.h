// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Granularities the DSP runtime allocates VTCM from a dispatch's partition in
// (vtcm_partition.h). Header-only and shared with the compiler, whose
// HexagonAnnotateDispatchResourcesPass bounds the VTCM each dispatch allocates
// and has to round the same way.

#ifndef HEXAGON_DSP_RT_VTCM_ALLOCATION_H
#define HEXAGON_DSP_RT_VTCM_ALLOCATION_H

// Smallest block a partition hands out, and the alignment and granularity of
// allocations aligned to less than HEXAGON_DSP_VTCM_LARGE_ALIGNMENT.
#define HEXAGON_DSP_VTCM_MIN_ALLOCATION_BYTES 128

// Granularity of allocations taken from the front of the partition
#define HEXAGON_DSP_VTCM_LARGE_ALIGNMENT 2048

#endif // HEXAGON_DSP_RT_VTCM_ALLOCATION_H
