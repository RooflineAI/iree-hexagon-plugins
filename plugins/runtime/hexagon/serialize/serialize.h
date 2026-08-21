// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_SERIALIZE_H_
#define IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_SERIALIZE_H_

/// Check that the serialization buffer with current pointer P and end pointer E
/// still has space for N bytes.
#define SERIALIZE_CHECK_SIZE(P, E, N)                                          \
  if (P + N > E) {                                                             \
    return iree_make_status(IREE_STATUS_INTERNAL,                              \
                            "size computation and actual serialization of "    \
                            "command buffer did not match");                   \
  }

/// Put a data structure of type T into the serialization buffer with current
/// pointer P and end pointer E.
/// Make a pointer V to the data structure for assigning values.
/// note: Because the ARM/DSP data structures are packed ones, casting uint8_t*
/// to struct pointers is okay.
#define SERIALIZE_TO(P, E, T, V)                                               \
  SERIALIZE_CHECK_SIZE(P, E, sizeof(T))                                        \
  T *V = (T *)P;                                                               \
  P += sizeof(T);

/// Copy pre-serialized data (pointer D, size S) into the serialization buffer
/// with current pointer P and end pointer E.
/// Note: S must be a side-effect-free expression because it is evaluated
//        multiple times.
#define SERIALIZE_COPY_TO(P, E, D, S)                                          \
  SERIALIZE_CHECK_SIZE(P, E, S)                                                \
  memcpy(P, D, S);                                                             \
  P += S;

#endif // IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_SERIALIZE_H_
