// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_UTILS_H_
#define IREE_HAL_DRIVERS_HEXAGON_UTILS_H_

#include "iree/base/api.h"

/// make an IREE status code from the errno variable
iree_status_code_t iree_hal_hexagon_status_code_from_errno(void);

/// parse a boolean from a string
iree_status_t iree_hal_hexagon_parse_bool_from_string(iree_string_view_t str,
                                                      int *parsed);

#endif // IREE_HAL_DRIVERS_HEXAGON_UTILS_H_
