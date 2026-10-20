// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sizes the DSP runtime allocates VTCM in. Header-only and shared with the
// compiler, whose HexagonAnnotateDispatchResourcesPass bounds the VTCM each
// dispatch allocates and has to round the same way.

#ifndef HEXAGON_DSP_RT_VTCM_ALLOCATION_H
#define HEXAGON_DSP_RT_VTCM_ALLOCATION_H

// Smallest block the VTCM pool hands out; smaller requests are rounded up to
// it (vtcm_pool.cpp).
#define HEXAGON_DSP_VTCM_MIN_ALLOCATION_BYTES 128

// hexagon-mlir's VTCMAllocation rounds the size of a buffer with this
// alignment to a multiple of it, and the size of any less aligned buffer to a
// multiple of HEXAGON_DSP_VTCM_MIN_ALLOCATION_BYTES
// (qcom_hexagon_backend/bin/runtime/src/HexagonBuffer.cpp). That code is
// third-party, so the value is mirrored here.
#define HEXAGON_DSP_VTCM_LARGE_ALIGNMENT 2048

#endif // HEXAGON_DSP_RT_VTCM_ALLOCATION_H
