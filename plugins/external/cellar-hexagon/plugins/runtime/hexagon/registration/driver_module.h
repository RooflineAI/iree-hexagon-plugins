// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVER_HEXAGON_REGISTRATION_DRIVER_MODULE_H_
#define IREE_HAL_DRIVER_HEXAGON_REGISTRATION_DRIVER_MODULE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

IREE_API_EXPORT iree_status_t
iree_hal_hexagon_driver_module_register(iree_hal_driver_registry_t *registry);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVER_HEXAGON_REGISTRATION_DRIVER_MODULE_H_
