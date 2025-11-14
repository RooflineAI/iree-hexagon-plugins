// Copyright 2025 RooflineAI GmbH

#include "hexagon/driver.h"

#include "device.h"
#include "hexagon/api.h"
#include "hexagon/utils.h"

// Hexagon SDK includes
#include <AEEStdErr.h>
#include <remote.h>

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_driver_options_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_hexagon_driver_options_initialize(
    iree_hal_hexagon_driver_options_t *out_options) {
  memset(out_options, 0, sizeof(*out_options));

  // TODO(hexagon): set defaults based on compiler configuration. Flags should
  // not be used as multiple devices may be configured within the process or the
  // hosting application may be authored in python/etc that does not use a flags
  // mechanism accessible here.

  iree_hal_hexagon_device_options_initialize(
      &out_options->default_device_options);
}

static iree_status_t iree_hal_hexagon_driver_options_verify(
    const iree_hal_hexagon_driver_options_t *options) {
  // TODO(hexagon): verify that the parameters are within expected ranges and
  // any requested features are supported.

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_driver_t
//===----------------------------------------------------------------------===//

// TODO(hexagon): if it's possible to have more than one device use real IDs.
// This is a placeholder for this skeleton that just indicates the first and
// only device.
#define IREE_HAL_HEXAGON_DEVICE_ID_DEFAULT                                     \
  ((iree_hal_hexagon_domain_id_t)CDSP_DOMAIN_ID)

typedef struct iree_hal_hexagon_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  iree_string_view_t identifier;
  iree_hal_hexagon_driver_options_t options;

  // + trailing identifier string storage
} iree_hal_hexagon_driver_t;

static const iree_hal_driver_vtable_t iree_hal_hexagon_driver_vtable;

static iree_hal_hexagon_driver_t *
iree_hal_hexagon_driver_cast(iree_hal_driver_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hexagon_driver_vtable);
  return (iree_hal_hexagon_driver_t *)base_value;
}

IREE_API_EXPORT iree_status_t iree_hal_hexagon_driver_create(
    iree_string_view_t identifier,
    const iree_hal_hexagon_driver_options_t *options,
    iree_allocator_t host_allocator, iree_hal_driver_t **out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_driver = NULL;

  // TODO(hexagon): verify options; this may be moved after any libraries are
  // loaded so the verification can use underlying implementation queries.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hexagon_driver_options_verify(options));

  iree_hal_hexagon_driver_t *driver = NULL;
  iree_host_size_t total_size = sizeof(*driver) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void **)&driver));
  iree_hal_resource_initialize(&iree_hal_hexagon_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(identifier, &driver->identifier,
                                    (char *)driver + total_size -
                                        identifier.size);

  // TODO(hexagon): if there are any string fields then they will need to be
  // retained as well (similar to the identifier they can be tagged on to the
  // end of the driver struct).
  memcpy(&driver->options, options, sizeof(*options));

  // TODO(hexagon): load libraries and query driver support from the system.
  // Devices need not be enumerated here if doing so is expensive; the
  // application may create drivers just to see if they are present but defer
  // device enumeration until the user requests one. Underlying implementations
  // can sometimes do bonkers static init stuff as soon as they are touched and
  // this code may want to do that on-demand instead.
  iree_status_t status = iree_ok_status();

  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t *)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t *)driver);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_hexagon_driver_destroy(iree_hal_driver_t *base_driver) {
  iree_hal_hexagon_driver_t *driver = iree_hal_hexagon_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(hexagon): if the driver loaded any libraries they should be closed
  // here.

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_hexagon_driver_query_available_devices(
    iree_hal_driver_t *base_driver, iree_allocator_t host_allocator,
    iree_host_size_t *out_device_info_count,
    iree_hal_device_info_t **out_device_infos) {
  // Query available hexagon devices and populate the output. Note
  // that unlike most IREE functions this allocates if required in order to
  // allow this to return uncached information. Uncached is preferred as it
  // allows devices that may come and go (power toggles, user visibility
  // toggles, etc) through a process lifetime to appear without needing a
  // full restart.

  iree_hal_hexagon_driver_t *driver = iree_hal_hexagon_driver_cast(base_driver);

  // Query all DSP domains if present or not.
  // If the API for querying is not present, assume no domain is present.
  static const iree_hal_hexagon_domain_id_t all_domain_ids[] = {
      ADSP_DOMAIN_ID, MDSP_DOMAIN_ID,  SDSP_DOMAIN_ID,
      CDSP_DOMAIN_ID, CDSP1_DOMAIN_ID,
  };
  static const unsigned int all_domain_ids_count =
      sizeof(all_domain_ids) / sizeof(all_domain_ids[0]);
  unsigned int detected_domain_ids_count = 0;
  iree_hal_hexagon_domain_id_t detected_domain_ids[all_domain_ids_count] =
      {}; // index is the same as in iree_hal_hexagon_domain_ids_names
  if (remote_handle_control) {
    for (unsigned int idx = 0; idx < all_domain_ids_count; ++idx) {
      // The Hexagon SDK example code polls for DOMAIN_SUPPORT. However, we need
      // to run unsigned modules, so we need to query for UNSIGNED_PD_SUPPORT,
      // in order to list only the DSP domains that support unsigned modules.
      struct remote_dsp_capability dsp_capability_domain = {
          all_domain_ids[idx], UNSIGNED_PD_SUPPORT, 0};
      int err =
          remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_domain,
                                sizeof(struct remote_dsp_capability));
      if (err == AEE_SUCCESS && dsp_capability_domain.capability != 0) {
        detected_domain_ids[detected_domain_ids_count++] = all_domain_ids[idx];
      }
    }
  }

  // Report a device for each domain that is present.
  // Allocate buffer for all device infos and fill with information.
  // Use domain Id as device ID and also report names.
  iree_host_size_t device_info_count = detected_domain_ids_count;
  iree_hal_device_info_t *device_infos = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, device_info_count * sizeof(iree_hal_device_info_t),
      &device_infos));
  for (int idx = 0; idx < detected_domain_ids_count; ++idx) {
    iree_hal_hexagon_fill_device_info(
        driver->identifier, detected_domain_ids[idx], &device_infos[idx]);
  }

  *out_device_info_count = device_info_count;
  *out_device_infos = device_infos;
  return iree_ok_status();
}

static iree_status_t
iree_hal_hexagon_driver_dump_device_info(iree_hal_driver_t *base_driver,
                                         iree_hal_device_id_t device_id,
                                         iree_string_builder_t *builder) {
  iree_hal_hexagon_driver_t *driver = iree_hal_hexagon_driver_cast(base_driver);

  // TODO(hexagon): add useful user-level information to the string builder for
  // the given device_id. This is used by the tools in features like
  // `--dump_devices` or may be used by hosting applications for diagnostics.
  (void)driver;

  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_device_options_parse(
    iree_host_size_t param_count, const iree_string_pair_t *params,
    iree_hal_hexagon_device_options_t *options) {

  for (int pi = 0; pi < (int)param_count; ++pi) {
    const iree_string_pair_t *param = &params[pi];
    // See README.md for a documentation of the options.
    if (iree_string_view_equal_case(param->key,
                                    iree_make_cstring_view("verbose"))) {
      IREE_RETURN_IF_ERROR(iree_hal_hexagon_parse_bool_from_string(
          param->value, &options->verbose));
      continue;
    }
    if (iree_string_view_equal_case(
            param->key, iree_make_cstring_view("dsp_status_notify"))) {
      IREE_RETURN_IF_ERROR(iree_hal_hexagon_parse_bool_from_string(
          param->value, &options->dsp_status_notify));
      continue;
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid option in URI");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_driver_create_device_by_id(
    iree_hal_driver_t *base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t *params,
    iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
  iree_hal_hexagon_driver_t *driver = iree_hal_hexagon_driver_cast(base_driver);

  // TODO(hexagon): use the provided params to overwrite the default options.
  // The format of the params is implementation-defined. The params strings can
  // be directly referenced if needed as the device creation is only allowed to
  // access them during the create call below.
  iree_hal_hexagon_device_options_t options =
      driver->options.default_device_options;
  IREE_RETURN_IF_ERROR(
      iree_hal_hexagon_device_options_parse(param_count, params, &options));

  // TODO(hexagon): implement creation by device_id; this is mostly used as
  // query_available_devices->create_device_by_id to avoid needing to expose
  // device paths (which may not always be 1:1). This skeleton only has a single
  // device so the ID is ignored.
  (void)driver;

  // device ID is Hexagon domain ID
  return iree_hal_hexagon_device_create(driver->identifier, device_id, &options,
                                        host_allocator, out_device);
}

static iree_status_t iree_hal_hexagon_driver_create_device_by_path(
    iree_hal_driver_t *base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t *params, iree_allocator_t host_allocator,
    iree_hal_device_t **out_device) {
  if (iree_string_view_is_empty(device_path)) {
    return iree_hal_hexagon_driver_create_device_by_id(
        base_driver, IREE_HAL_HEXAGON_DEVICE_ID_DEFAULT, param_count, params,
        host_allocator, out_device);
  }

  // Try parsing path as device ID.
  iree_hal_device_id_t device_id = 0;
  if (iree_string_view_atoi_uint32(device_path, &device_id)) {
    return iree_hal_hexagon_driver_create_device_by_id(
        base_driver, device_id, param_count, params, host_allocator,
        out_device);
  }

  // Try parsing path as device name. Use domain ID as device ID.
  iree_hal_hexagon_domain_id_t domain_id = 99999;
  if (iree_hal_hexagon_get_domain_id(device_path, &domain_id)) {
    return iree_hal_hexagon_driver_create_device_by_id(
        base_driver, domain_id, param_count, params, host_allocator,
        out_device);
  }

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unsupported device path");
}

static const iree_hal_driver_vtable_t iree_hal_hexagon_driver_vtable = {
    .destroy = iree_hal_hexagon_driver_destroy,
    .query_available_devices = iree_hal_hexagon_driver_query_available_devices,
    .dump_device_info = iree_hal_hexagon_driver_dump_device_info,
    .create_device_by_id = iree_hal_hexagon_driver_create_device_by_id,
    .create_device_by_path = iree_hal_hexagon_driver_create_device_by_path,
};
