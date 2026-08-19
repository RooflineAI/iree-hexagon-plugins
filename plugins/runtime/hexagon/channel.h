// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_CHANNEL_H_
#define IREE_HAL_DRIVERS_HEXAGON_CHANNEL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_channel_t
//===----------------------------------------------------------------------===//

// Creates a {Qualcomm Hexagon} HAL collective channel using the given |params|.
iree_status_t iree_hal_hexagon_channel_create(iree_hal_channel_params_t params,
                                              iree_allocator_t host_allocator,
                                              iree_hal_channel_t **out_channel);

#endif // IREE_HAL_DRIVERS_HEXAGON_CHANNEL_H_
