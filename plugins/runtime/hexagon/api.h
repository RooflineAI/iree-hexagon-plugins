// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_API_H_
#define IREE_HAL_DRIVERS_HEXAGON_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_device_t
//===----------------------------------------------------------------------===//

// DSP domain ID used by Hexagon SDK
typedef int iree_hal_hexagon_domain_id_t;

// Parameters configuring an iree_hal_hexagon_device_t.
// Must be initialized with iree_hal_hexagon_device_options_initialize prior to
// use.
typedef struct iree_hal_hexagon_device_options_t {
  /// verbose (Debug) output enabled, 0 by default, 1 if enabled
  int verbose;
  /// notify about DSP status changes, 0 by default, 1 if enabled
  int dsp_status_notify;
  // note: This sdata structure is copied into the device data in
  // iree_hal_hexagon_device_create()
  // TODO(hexagon): options for initializing a device such as hardware
  // identifiers, implementation mode switches, and debugging control.
} iree_hal_hexagon_device_options_t;

// Initializes |out_options| to default values.
IREE_API_EXPORT void iree_hal_hexagon_device_options_initialize(
    iree_hal_hexagon_device_options_t *out_options);

// Creates a {Qualcomm Hexagon} HAL device with the given |options|.
//
// The provided |identifier| will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by `IREE::HAL::TargetDevice`.
//
// |out_device| must be released by the caller (see iree_hal_device_release).
IREE_API_EXPORT iree_status_t iree_hal_hexagon_device_create(
    iree_string_view_t identifier, iree_hal_hexagon_domain_id_t domain_id,
    const iree_hal_hexagon_device_options_t *options,
    iree_allocator_t host_allocator, iree_hal_device_t **out_device);

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_driver_t
//===----------------------------------------------------------------------===//

// Parameters for configuring an iree_hal_hexagon_driver_t.
// Must be initialized with iree_hal_hexagon_driver_options_initialize prior to
// use.
typedef struct iree_hal_hexagon_driver_options_t {
  // TODO(hexagon): options for initializing the driver such as library search
  // paths, version min/max, etc.

  // Default device options when none are provided during device creation.
  iree_hal_hexagon_device_options_t default_device_options;
} iree_hal_hexagon_driver_options_t;

// Initializes the given |out_options| with default driver creation options.
IREE_API_EXPORT void iree_hal_hexagon_driver_options_initialize(
    iree_hal_hexagon_driver_options_t *out_options);

// Creates a {Qualcomm Hexagon} HAL driver with the given |options|, from which
// {Qualcomm Hexagon} devices can be enumerated and created with specific
// parameters.
//
// The provided |identifier| will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by IREE::HAL::TargetDevice.
//
// |out_driver| must be released by the caller (see iree_hal_driver_release).
IREE_API_EXPORT iree_status_t iree_hal_hexagon_driver_create(
    iree_string_view_t identifier,
    const iree_hal_hexagon_driver_options_t *options,
    iree_allocator_t host_allocator, iree_hal_driver_t **out_driver);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_HEXAGON_API_H_
