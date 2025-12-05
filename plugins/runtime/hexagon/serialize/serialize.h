// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_H_
#define IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_H_

/// Put a data structure of type T into the serialization buffer with current
/// pointer P and end pointer E.
/// Make a pointer V to the data structure for assigning values.
/// note: Because the ARM/DSP data structures are packed ones, casting uint8_t*
/// to struct pointers is okay.
#define SERIALIZE_TO(P, E, T, V)                                               \
  if (P + sizeof(T) > E) {                                                     \
    return iree_make_status(IREE_STATUS_INTERNAL,                              \
                            "size computation and actual serialization of "    \
                            "command buffer did not match");                   \
  }                                                                            \
  T *V = (T *)P;                                                               \
  P += sizeof(T);

#endif // IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_H_
