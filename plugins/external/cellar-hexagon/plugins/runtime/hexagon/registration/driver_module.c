// Copyright 2025 RooflineAI GmbH

#include "hexagon/registration/driver_module.h"

#include "hexagon/api.h"
#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/base/string_view.h"

IREE_FLAG_LIST(
    string, hexagon_pmu_events,
    "Hexagon PMU events to configure for profiling. May be specified multiple\n"
    "times or as a comma-separated list. Values must be numeric IDs (decimal "
    "or 0x...). Up to 8 entries are used.");

static iree_status_t iree_hal_hexagon_apply_pmu_event_flags(
    iree_hal_hexagon_device_options_t *options) {
  const iree_flag_string_list_t list = FLAG_hexagon_pmu_events_list();

  uint32_t count = 0;
  for (iree_host_size_t li = 0; li < list.count; ++li) {
    iree_string_view_t remaining = list.values[li];
    while (true) {
      iree_string_view_t segment = iree_string_view_empty();
      intptr_t split_index =
          iree_string_view_split(remaining, ',', &segment, &remaining);
      segment = iree_string_view_trim(segment);
      if (!iree_string_view_is_empty(segment)) {
        if (count >= IREE_HAL_HEXAGON_PMU_COUNTERS) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "too many hexagon PMU events (max %u)",
                                  (unsigned int)IREE_HAL_HEXAGON_PMU_COUNTERS);
        }
        uint32_t id = 0;
        if (!iree_string_view_atoi_uint32(segment, &id)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "invalid hexagon PMU event id '%.*s'",
                                  (int)segment.size, segment.data);
        }
        options->pmu_event_ids[count++] = id;
      }
      if (split_index == -1)
        break;
    }
  }

  options->pmu_event_ids_count = (uint8_t)count;
  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_driver_factory_enumerate(
    void *self, iree_host_size_t *out_driver_info_count,
    const iree_hal_driver_info_t **out_driver_infos) {
  static const iree_hal_driver_info_t default_driver_info = {
      .driver_name = IREE_SVL("hexagon"),
      .full_name = IREE_SVL("Qualcomm Hexagon Driver"),
  };
  *out_driver_info_count = 1;
  *out_driver_infos = &default_driver_info;
  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_driver_factory_try_create(
    void *self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t **out_driver) {
  if (!iree_string_view_equal(driver_name, IREE_SV("hexagon"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  // TODO(hexagon): populate options from flags. This driver module file is only
  // used in native tools that have access to the flags library. Programmatic
  // creation of the driver and devices will bypass this file and pass the
  // options via this struct or key-value string parameters.
  iree_hal_hexagon_driver_options_t options;
  iree_hal_hexagon_driver_options_initialize(&options);
  iree_status_t status =
      iree_hal_hexagon_apply_pmu_event_flags(&options.default_device_options);
  if (!iree_status_is_ok(status))
    return status;

  status = iree_hal_hexagon_driver_create(driver_name, &options, host_allocator,
                                          out_driver);

  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_hexagon_driver_module_register(iree_hal_driver_registry_t *registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_hexagon_driver_factory_enumerate,
      .try_create = iree_hal_hexagon_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
