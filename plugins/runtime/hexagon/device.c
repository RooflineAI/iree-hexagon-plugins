// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "hexagon/device.h"

#include "hexagon/allocator.h"
#include "hexagon/api.h"
#include "hexagon/buffer.h"
#include "hexagon/channel.h"
#include "hexagon/command_buffer.h"
#include "hexagon/event.h"
#include "hexagon/executable_cache.h"
#include "hexagon/profiler.h"
#include "hexagon/semaphore.h"
#include "hexagon/serialize/bindings_serialize.h"
#include "hexagon/utils.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/arena.h"
#include "iree/base/status.h"
#include "iree/base/string_builder.h"
#include "iree/base/string_view.h"
#include "iree/base/tracing.h"
#include "iree/hal/allocator.h"
#include "iree/hal/channel_provider.h"
#include "iree/hal/resource.h"
#include "iree/hal/semaphore.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/queue_emulation.h"
#include "iree/hal/utils/queue_host_call_emulation.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>

// Hexagon SDK includes
#include "AEEStdErr.h"
#include "remote.h"
#include "rpcmem.h"

#include "hexagon_dsp.h"
#include "rpc_types.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_device_options_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_hexagon_device_options_initialize(
    iree_hal_hexagon_device_options_t *out_options) {
  memset(out_options, 0, sizeof(*out_options));
  // TODO(hexagon): set defaults based on compiler configuration. Flags should
  // not be used as multiple devices may be configured within the process or the
  // hosting application may be authored in python/etc that does not use a flags
  // mechanism accessible here.
  out_options->profiler_extra_records_per_dispatch = 256;
}

static iree_status_t iree_hal_hexagon_device_options_verify(
    const iree_hal_hexagon_device_options_t *options) {
  // TODO(hexagon): verify that the parameters are within expected ranges and
  // any requested features are supported.
  if (options->pmu_event_ids_count > IREE_HAL_HEXAGON_PMU_COUNTERS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "too many PMU event IDs (%u, max %u)",
                            (unsigned int)options->pmu_event_ids_count,
                            (unsigned int)IREE_HAL_HEXAGON_PMU_COUNTERS);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_device_t
//===----------------------------------------------------------------------===//

struct iree_hal_hexagon_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;
  iree_hal_hexagon_domain_id_t domain_id;

  /// parsed device options, copied here by value due to short lifetime of
  /// options arg passed to init
  iree_hal_hexagon_device_options_t options;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t *device_allocator;

  // For now, the entire Hexagon device uses a single block pool.
  iree_arena_block_pool_t block_pool;

  // Proactor pool for async I/O. Retained for the lifetime of the device to
  // ensure proactor threads outlive all device resources (semaphores, etc.).
  iree_async_proactor_pool_t *proactor_pool;

  // Proactor selected from the pool for this device's async I/O operations.
  // Borrowed from the pool -- valid as long as the pool is retained.
  iree_async_proactor_t *proactor;

  /// Connection to DSP / DSP RPC session.
  rpc_session_handle_t rpc_session_handle;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t *channel_provider;

  // Topology information if this device is part of a multi-device topology.
  iree_hal_device_topology_info_t topology_info;

#if defined(IREE_HAL_HEXAGON_ENABLE_PROFILER)
  // Cached tracy GPU context id.
  uint8_t tracy_context_id;
  iree_tracing_context_t *tracy_plot_context;
#endif

  // + trailing identifier string storage
};

static const iree_hal_device_vtable_t iree_hal_hexagon_device_vtable;

static iree_hal_hexagon_device_t *
iree_hal_hexagon_device_cast(iree_hal_device_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hexagon_device_vtable);
  return (iree_hal_hexagon_device_t *)base_value;
}

#define LOOKUP_HEXAGON_DOMAINS                                                 \
  LOOKUP_HEXAGON_DOMAIN(ADSP_DOMAIN_ID, "ADSP", ADSP_DOMAIN)                   \
  LOOKUP_HEXAGON_DOMAIN(MDSP_DOMAIN_ID, "MDSP", MDSP_DOMAIN)                   \
  LOOKUP_HEXAGON_DOMAIN(SDSP_DOMAIN_ID, "SDSP", SDSP_DOMAIN)                   \
  LOOKUP_HEXAGON_DOMAIN(CDSP_DOMAIN_ID, "CDSP", CDSP_DOMAIN)                   \
  LOOKUP_HEXAGON_DOMAIN(CDSP1_DOMAIN_ID, "CDSP1", CDSP1_DOMAIN)

int iree_hal_hexagon_get_domain_name(iree_hal_hexagon_domain_id_t domain_id,
                                     const char **name) {
  switch (domain_id) {
#define LOOKUP_HEXAGON_DOMAIN(id, dom_name, uri_suffix)                        \
  case id:                                                                     \
    *name = dom_name;                                                          \
    return 1;
    LOOKUP_HEXAGON_DOMAINS
#undef LOOKUP_HEXAGON_DOMAIN
  default:
    return 0;
  }
}

const char *iree_hal_hexagon_get_domain_name_or_unknown(
    iree_hal_hexagon_domain_id_t domain_id) {
  const char *name = NULL;
  if (iree_hal_hexagon_get_domain_name(domain_id, &name)) {
    return name;
  }
  return "unknown";
}

int iree_hal_hexagon_get_domain_id(iree_string_view_t name,
                                   iree_hal_hexagon_domain_id_t *domain_id) {
#define LOOKUP_HEXAGON_DOMAIN(id, check_name, uri_suffix)                      \
  if (iree_string_view_equal(name, iree_make_cstring_view(check_name))) {      \
    *domain_id = id;                                                           \
    return 1;                                                                  \
  }
  LOOKUP_HEXAGON_DOMAINS
#undef LOOKUP_HEXAGON_DOMAIN
  return 0;
}

// Get the name of the directory (ending with a slash) of the currently running
// binary.
static iree_status_t get_binary_dir(iree_allocator_t allocator,
                                    char **binary_dir) {
  // build string "/proc/$$/exe"
  iree_string_builder_t proc_exe_builder;
  iree_string_builder_initialize(allocator, &proc_exe_builder);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      &proc_exe_builder, "/proc/%u/exe", (unsigned int)getpid()));
  iree_string_view_t end = {.data = "", .size = 1};
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(&proc_exe_builder, end));
  // proc_exe_builder contains a string that ends in '\0', so the .data of the
  // view is a valid C-string.
  const char *proc_exe = iree_string_builder_view(&proc_exe_builder).data;

  // Resolve symlink. Need to try with larger buffer till it fits.
  iree_host_size_t resolved_max_size = 64;
  char *resolved = NULL;
  while (1) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
        allocator, resolved_max_size, (void **)&resolved));
    ssize_t ret = readlink(proc_exe, resolved, resolved_max_size);
    if (ret < 0) {
      return IREE_HAL_HEXAGON_MAKE_STATUS_FROM_ERRNO(
          "resolving path of binary failed");
    }
    // resolved path fit buffer -> done
    //  - if ret == resolved_max_size, it is not clear if it got truncated or
    //    not, so treat as "did not fit"
    //  - readlink does not add a terminating 0 char, so add it, there is space
    //    in the buffer for it as ret is strictly less than resolved_max_size
    if (ret < resolved_max_size) {
      resolved[ret] = 0;
      break;
    }
    // buffer too small -> enlarge and try again
    // - fail if size reached 64K (way to big for file path), something is
    //   going very wrong if that happens
    iree_allocator_free(allocator, resolved);
    if (resolved_max_size >= 65536) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "resolved path of binary did not fit 64K");
    }
    resolved_max_size *= 2;
  }

  iree_string_builder_deinitialize(&proc_exe_builder);

  // Remove basename from binary path binary. Keep slash at end of directory.
  char *last_slash = strrchr(resolved, '/');
  if (!last_slash) {
    iree_allocator_free(allocator, resolved);
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no directory found in binary path '%s'", resolved);
  }
  last_slash[1] = 0;

  *binary_dir = resolved;
  return iree_ok_status();
}

// Convert a domain ID to a domain URI suffix.
//
// Return domain URI suffix in *domain_uri_suffix as pointer to string literal
//
// Return 1 on success, 0 on failure
static int domain_id_to_uri_suffix(iree_hal_hexagon_domain_id_t domain_id,
                                   const char **domain_uri_suffix) {
  switch (domain_id) {
#define LOOKUP_HEXAGON_DOMAIN(id, name, uri_suffix)                            \
  case id:                                                                     \
    *domain_uri_suffix = uri_suffix;                                           \
    return 1;
    LOOKUP_HEXAGON_DOMAINS
#undef LOOKUP_HEXAGON_DOMAIN
  }
  return 0;
}

void iree_hal_hexagon_fill_device_info(iree_hal_hexagon_domain_id_t domain_id,
                                       iree_hal_device_info_t *device_info) {
  device_info->device_id = domain_id;
  device_info->name = iree_make_cstring_view(
      iree_hal_hexagon_get_domain_name_or_unknown(domain_id));
  device_info->path = device_info->name;
}

iree_hal_hexagon_domain_id_t
iree_hal_hexagon_device_get_domain_id(iree_hal_hexagon_device_t *device) {
  return device->domain_id;
}

rpc_session_handle_t iree_hal_hexagon_device_get_rpc_session_handle(
    iree_hal_hexagon_device_t *device) {
  return device->rpc_session_handle;
}

int iree_hal_hexagon_pd_status_callback(void *context, int domain, int session,
                                        remote_rpc_status_flags_t status) {
  iree_hal_hexagon_device_t *device = (iree_hal_hexagon_device_t *)context;
  (void)device; // currently unused
  const char *dsp_status = "unknown";
  int nErr = AEE_EBADITEM;
  switch (status) {
  case FASTRPC_USER_PD_UP:
    dsp_status = "PD is up";
    nErr = AEE_SUCCESS;
    break;
  case FASTRPC_USER_PD_EXIT:
    dsp_status = "PD closed";
    nErr = AEE_SUCCESS;
    break;
  case FASTRPC_USER_PD_FORCE_KILL:
    dsp_status = "PD force kill";
    nErr = AEE_SUCCESS;
    break;
  case FASTRPC_USER_PD_EXCEPTION:
    dsp_status = "PD exception";
    nErr = AEE_SUCCESS;
    break;
  case FASTRPC_DSP_SSR:
    dsp_status = "DSP SSR";
    nErr = AEE_SUCCESS;
    break;
  case FASTRPC_INTERNAL_STATUS_RESERVED_1:
    dsp_status = "internal status reserved 1";
    nErr = AEE_SUCCESS;
    break;
  case FASTRPC_USERPD_TIMEOUT:
    dsp_status = "PD timeout";
    nErr = AEE_SUCCESS;
    break;
  }
  fprintf(stderr, "iree-run-module: hexagon: DSP status: %s\n", dsp_status);
  return nErr;
}

static iree_status_t iree_hal_hexagon_request_status_notifications(
    int domain_id, void *context,
    int (*notify_callback)(void *context, int domain, int session,
                           remote_rpc_status_flags_t status)) {
  // Query the DSP for status information support.
  struct remote_dsp_capability dsp_capability_status_notification_support;
  dsp_capability_status_notification_support.domain = (uint32_t)domain_id;
  dsp_capability_status_notification_support.attribute_ID =
      STATUS_NOTIFICATION_SUPPORT;
  dsp_capability_status_notification_support.capability = (uint32_t)0;
  int nErr = remote_handle_control(DSPRPC_GET_DSP_INFO,
                                   &dsp_capability_status_notification_support,
                                   sizeof(struct remote_dsp_capability));
  if ((nErr & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "FastRPC Capability API is not supported on this device");
  }
  if (nErr != AEE_SUCCESS) {
    return IREE_HAL_HEXAGON_MAKE_STATUS_FROM_DSP_ERR(
        nErr, "querying for DSP status notification support failed");
  }
  if (dsp_capability_status_notification_support.capability != 1) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no DSP status notification support available");
  }

  // Enable DSP status notifications.
  struct remote_rpc_notif_register notif = {
      .context = context, .domain = domain_id, .notifier_fn = notify_callback};
  nErr = remote_session_control(FASTRPC_REGISTER_STATUS_NOTIFICATIONS,
                                (void *)&notif, sizeof(notif));
  if (nErr != AEE_SUCCESS) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to enable DSP status notifications");
  }

  return iree_ok_status();
}

static iree_status_t
iree_hal_hexagon_device_set_up(iree_hal_hexagon_domain_id_t domain_id,
                               iree_hal_hexagon_device_t *device) {
  // TODO(hexagon): pass device handles and pool configuration to the allocator.
  // Some implementations may share allocators across multiple devices created
  // from the same driver.

  if (device->options.dsp_status_notify) {
    IREE_RETURN_IF_ERROR(iree_hal_hexagon_request_status_notifications(
        device->domain_id, (void *)device,
        iree_hal_hexagon_pd_status_callback));
  }

  IREE_RETURN_IF_ERROR(iree_hal_hexagon_allocator_create(
      device->host_allocator, device, &device->device_allocator));

  // Initialize block pool.
  iree_arena_block_pool_initialize(4096, device->host_allocator,
                                   &device->block_pool);

  // Configure the DSP to accept unsigned modules
  struct remote_rpc_control_unsigned_module control_unsigned_module;
  control_unsigned_module.domain = device->domain_id;
  control_unsigned_module.enable = 1;
  if (remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                             &control_unsigned_module,
                             sizeof(control_unsigned_module)) != AEE_SUCCESS) {
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "cannot enable unsigned modules on DSP");
  }

  // Assemble the URI for the DSP session.
  // The format is: <prefix> <so_file_name> "?" <func_name> "&" <mod_ver> "&"
  // <idl_ver> "&" <dom>
  //   prefix: "file:///" (yes 3 slashes here, not 2)
  //   so_file_name: path to the DSP library,
  //                 no slash in here means to look in predefined directories in
  //                 an $DSP_LIBRARY_PATH, an absolute path works
  //   func_name: name of the main entry function into the DSP library (e.g.
  //   <interface_name> "_skel_handle_invoke") mod_ver: "_modver=" <version>
  //   (e.g. "1.0") idl_ver: "_idlver=" <version> (e.g. "1.2.3") dom: _dom=<dsp
  //   type lower case> (e.g. "cdsp")
  // hexagon_dsp_URI contains everything without "&" <dom> and uses just the
  // library basename for <so_file_name>. If getting the absolute path of the
  // directory of the running binary works, insert this directory plus
  // "../lib/hexagon/", so loading the DSP library works without setting
  // DSP_LIBRARY_PATH.

  // Check prefix and cut it off.
  iree_string_view_t uri_main = iree_make_cstring_view(hexagon_dsp_URI);
  iree_string_view_t prefix = iree_make_cstring_view("file:///");
  if (!iree_string_view_consume_prefix(&uri_main, prefix)) {
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "DSP URI %.*s does not start with expected prefix",
                            (int)uri_main.size, uri_main.data);
  }

  // Obtain DSP domain suffix.
  const char *suffix = NULL;
  if (!domain_id_to_uri_suffix(device->domain_id, &suffix)) {
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "cannot convert DSP domain ID to URI suffix");
  }

  // Obtain directory of current binary.
  char *binary_dir = NULL;
  IREE_RETURN_IF_ERROR(get_binary_dir(device->host_allocator, &binary_dir));

  // Build full URI.
  iree_string_builder_t uri_builder;
  iree_string_builder_initialize(device->host_allocator, &uri_builder);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(&uri_builder, prefix));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&uri_builder, binary_dir));
  iree_allocator_free(device->host_allocator, binary_dir);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&uri_builder, "../lib/hexagon/"));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(&uri_builder, uri_main));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&uri_builder, suffix));
  iree_string_view_t end = {.data = "", .size = 1};
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(&uri_builder, end));
  // uri_builder contains a string that ends in '\0', so the .data of the view
  // is a valid C-string.
  const char *uri = iree_string_builder_view(&uri_builder).data;

  // Open the actual DSP RPC session.
  int dsp_open_ret = hexagon_dsp_open(uri, &device->rpc_session_handle);
  iree_string_builder_deinitialize(&uri_builder);
  if (dsp_open_ret != AEE_SUCCESS) {
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "opening RPC session with DSP failed");
  }

  return iree_ok_status();
}

iree_status_t iree_hal_hexagon_device_create(
    iree_string_view_t identifier, iree_hal_hexagon_domain_id_t domain_id,
    const iree_hal_hexagon_device_options_t *options,
    const iree_hal_device_create_params_t *create_params,
    iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_device = NULL;

  // Verify the parameters prior to creating resources.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hexagon_device_options_verify(options));

  iree_hal_hexagon_device_t *device = NULL;
  iree_host_size_t total_size = sizeof(*device) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void **)&device));
  iree_hal_resource_initialize(&iree_hal_hexagon_device_vtable,
                               &device->resource);
  iree_string_view_append_to_buffer(identifier, &device->identifier,
                                    (char *)device + total_size -
                                        identifier.size);
  device->domain_id = domain_id;
  device->options = *options; /// copy by value due to lifetime of options arg
  device->host_allocator = host_allocator;

  // Retain the proactor pool and acquire a proactor for this device. The pool
  // is retained for the device lifetime so proactor threads outlive all
  // device resources (semaphores).
  device->proactor_pool = create_params->proactor_pool;
  iree_async_proactor_pool_retain(device->proactor_pool);

#if defined(IREE_HAL_HEXAGON_ENABLE_PROFILER)
  device->tracy_context_id = IREE_HAL_HEXAGON_TRACY_CONTEXT_INVALID;
  device->tracy_plot_context = NULL;
#endif

  iree_status_t status =
      iree_async_proactor_pool_get(device->proactor_pool, 0, &device->proactor);

  if (iree_status_is_ok(status)) {
    status = iree_hal_hexagon_device_set_up(domain_id, device);
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t *)device;
  } else {
    iree_hal_device_release((iree_hal_device_t *)device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_hexagon_device_destroy(iree_hal_device_t *base_device) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(hexagon): release all implementation resources here. It's expected
  // that this is only called once all outstanding resources created with this
  // device have been released by the application and no work is outstanding. If
  // the implementation performs internal async operations those should be
  // shutdown and joined first.

  iree_hal_channel_provider_release(device->channel_provider);

  if (device->rpc_session_handle) {
    hexagon_dsp_close(device->rpc_session_handle);
  }

#if defined(IREE_HAL_HEXAGON_ENABLE_PROFILER)
  iree_tracing_context_free(device->tracy_plot_context);
#endif

  iree_async_proactor_pool_release(device->proactor_pool);
  iree_arena_block_pool_deinitialize(&device->block_pool);
  iree_hal_allocator_release(device->device_allocator);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t
iree_hal_hexagon_device_id(iree_hal_device_t *base_device) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);
  return iree_make_cstring_view(
      iree_hal_hexagon_get_domain_name_or_unknown(device->domain_id));
}

static iree_allocator_t
iree_hal_hexagon_device_host_allocator(iree_hal_device_t *base_device) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t *
iree_hal_hexagon_device_allocator(iree_hal_device_t *base_device) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);
  return device->device_allocator;
}

static void
iree_hal_hexagon_replace_device_allocator(iree_hal_device_t *base_device,
                                          iree_hal_allocator_t *new_allocator) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_hexagon_replace_channel_provider(
    iree_hal_device_t *base_device, iree_hal_channel_provider_t *new_provider) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t
iree_hal_hexagon_device_trim(iree_hal_device_t *base_device) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): if the device has any cached resources that can be safely
  // dropped here (unused pools/etc). This is usually called in low-memory
  // situations or when the HAL device will not be used for awhile (device
  // entering sleep mode or a low power state, etc).

  IREE_RETURN_IF_ERROR(iree_hal_allocator_trim(device->device_allocator));

  return iree_ok_status();
}

static iree_status_t
iree_hal_hexagon_device_query_i64(iree_hal_device_t *base_device,
                                  iree_string_view_t category,
                                  iree_string_view_t key, int64_t *out_value) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);
  *out_value = 0;

  // TODO(hexagon): implement additional queries. These are stubs for common
  // ones as used by the compiler. Targets may have their own, though, and
  // connect with them by emitting `hal.device.query` ops in programs or calling
  // the query method at runtime via the HAL API.

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    // NOTE: this is a fuzzy match and can allow a program to work with multiple
    // device implementations.
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    // NOTE: this is a fuzzy match and can allow multiple formats to be used
    // (this should return 1 for any format supported).
    // For now, accept exactly the "Hexagon instructions, no OS" one.
    // TODO(hexagon): match a format and return true.
    *out_value = iree_string_view_equal(
        key, iree_make_cstring_view("embedded-elf-hexagon"));
    return iree_ok_status();
  }

  // TODO(hexagon): return basic queries for concurrency to allow programs to
  // estimate potential utilization.
  if (iree_string_view_equal(category, IREE_SV("hal.device"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      *out_value = 1;
      return iree_ok_status();
    }
  } else if (iree_string_view_equal(category, IREE_SV("hal.dispatch"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      *out_value = 1;
      return iree_ok_status();
    }
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_hexagon_device_query_capabilities(
    iree_hal_device_t *base_device,
    iree_hal_device_capabilities_t *out_capabilities) {
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t *
iree_hal_hexagon_device_topology_info(iree_hal_device_t *base_device) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t
iree_hal_hexagon_device_refine_topology_edge(iree_hal_device_t *src_device,
                                             iree_hal_device_t *dst_device,
                                             iree_hal_topology_edge_t *edge) {
  (void)src_device;
  (void)dst_device;
  (void)edge;
  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_device_assign_topology_info(
    iree_hal_device_t *base_device,
    const iree_hal_device_topology_info_t *topology_info) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);
  device->topology_info = *topology_info;
  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_device_create_channel(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t **out_channel) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): pass any additional resources required to create the
  // channel. The device->channel_provider can be used to get default
  // rank/count, exchange IDs, etc as needed.
  (void)device;

  return iree_hal_hexagon_channel_create(
      params, iree_hal_device_host_allocator(base_device), out_channel);
}

static iree_status_t iree_hal_hexagon_device_create_command_buffer(
    iree_hal_device_t *base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t **out_command_buffer) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);
  // TODO(hexagon): pass any additional resources required to create the command
  // buffer. The implementation could pool command buffers here.
  return iree_hal_hexagon_command_buffer_create(
      iree_hal_device_allocator(base_device), mode, command_categories,
      queue_affinity, binding_capacity, device->host_allocator,
      device->rpc_session_handle, &device->block_pool,
      device->options.profiler_extra_records_per_dispatch, out_command_buffer);
}

static iree_status_t iree_hal_hexagon_device_create_event(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t **out_event) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): pass any additional resources required to create the event.
  // The implementation could pool events here.
  (void)device;

  return iree_hal_hexagon_event_create(
      queue_affinity, flags, iree_hal_device_host_allocator(base_device),
      out_event);
}

static iree_status_t iree_hal_hexagon_device_create_executable_cache(
    iree_hal_device_t *base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t **out_executable_cache) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): pass any additional resources required during executable
  // creation or cache management.
  (void)device;

  return iree_hal_hexagon_executable_cache_create(
      identifier, iree_hal_device_host_allocator(base_device),
      device->rpc_session_handle, out_executable_cache);
}

static iree_status_t iree_hal_hexagon_device_import_file(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t *handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t **out_file) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): if the implementation supports native file operations
  // definitely prefer that. The emulated file I/O present here as a default is
  // inefficient. The queue affinity specifies which queues may access the file
  // via read and write queue operations.
  return iree_hal_file_from_handle(
      iree_hal_device_allocator(base_device), queue_affinity, access, handle,
      device->proactor, iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_hexagon_device_create_semaphore(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t **out_semaphore) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): pass any additional resources required to create or track
  // the semaphore. The implementation could pool semaphores here.

  return iree_hal_hexagon_semaphore_create(
      device->proactor, queue_affinity, initial_value, flags,
      device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_hexagon_device_query_semaphore_compatibility(
    iree_hal_device_t *base_device, iree_hal_semaphore_t *semaphore) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): return the appropriate bits for the provided semaphore
  // indicating how it may be used with this device. The semaphore may have been
  // created or imported on this device or any other device from the same
  // driver. Certain implementations may allow semaphores from other drivers to
  // be used and those can be checked here (though the API to do this isn't
  // implemented yet).
  (void)device;
  iree_hal_semaphore_compatibility_t compatibility =
      IREE_HAL_SEMAPHORE_COMPATIBILITY_NONE;

  return compatibility;
}

static iree_status_t iree_hal_hexagon_device_queue_alloca(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t *pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t **IREE_RESTRICT out_buffer) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): perform an allocation of a transient buffer in queue order.
  // The allocation may be used on any queue set in the provided queue affinity.
  // Deallocation via queue_dealloca is preferred but users are allowed to
  // deallocate the buffer on the host via iree_hal_buffer_release even if they
  // allocated it with queue_alloca.

  (void)device; // not yet used

  // To get started, implement the functionality sequentially, outside of any
  // queue.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(
      wait_semaphore_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_RETURN_IF_ERROR(
      iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(base_device),
                                         params, allocation_size, out_buffer));
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_list_signal(signal_semaphore_list, /*frontier=*/NULL));
  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_device_queue_dealloca(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t *buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): perform a deallocation of the transient buffer in queue
  // order. Only buffers allocated with queue_alloca on the same device will be
  // passed. Note that different queues on the same device may have allocated
  // the buffer and if the same queue must deallocate it the implementation will
  // need to track that on the buffer. The user is allowed to deallocate the
  // buffer on the host via iree_hal_buffer_release even if they allocated it
  // with queue_alloca.

  (void)device; // not yet used

  // To get started, implement the functionality sequentially, outside of any
  // queue.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(
      wait_semaphore_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_allocator_deallocate_buffer(iree_hal_device_allocator(base_device),
                                       buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_list_signal(signal_semaphore_list, /*frontier=*/NULL));
  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_device_queue_fill(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void *pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  // TODO(hexagon): if a native queue fill operation is available use that
  // instead. The emulated fill creates a command buffer and executes it and
  // it's best if the extra recording/upload/allocation time can be avoided.
  return iree_hal_device_queue_emulated_fill(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      target_buffer, target_offset, length, pattern, pattern_length, flags);
}

static iree_status_t iree_hal_hexagon_device_queue_update(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void *source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  // TODO(hexagon): if a native queue update operation is available use that
  // instead. The emulated update creates a command buffer and executes it and
  // it's best if the extra recording/upload/allocation time can be avoided.
  // Since command buffers have a limited capacity for embedded data the
  // emulated version may need to allocate buffers, split the update into
  // multiple commands, or commit other sins a native implementation would be
  // able to avoid.
  return iree_hal_device_queue_emulated_update(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_buffer, target_offset, length,
      flags);
}

static iree_status_t iree_hal_hexagon_device_queue_copy(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t *source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  // TODO(hexagon): if a native queue copy operation is available use that
  // instead. The emulated copy creates a command buffer and executes it and
  // it's best if the extra recording/upload/allocation time can be avoided.
  return iree_hal_device_queue_emulated_copy(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_buffer, target_offset, length,
      flags);
}

static iree_status_t iree_hal_hexagon_device_queue_read(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t *source_file, uint64_t source_offset,
    iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  // TODO(hexagon): if native support for file operations are available then
  // definitely prefer those over the emulated implementation provided here by
  // default. The implementation performs allocations, creates semaphores, and
  // submits command buffers with host-device blocking behavior.

  // TODO: expose streaming chunk count/size options.
  iree_hal_file_transfer_options_t options = {
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_read_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_file, source_offset, target_buffer, target_offset, length, flags,
      options));
  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_device_queue_write(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t *source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t *target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  // TODO(hexagon): if native support for file operations are available then
  // definitely prefer those over the emulated implementation provided here by
  // default. The implementation performs allocations, creates semaphores, and
  // submits command buffers with host-device blocking behavior.

  // TODO: expose streaming chunk count/size options.
  iree_hal_file_transfer_options_t options = {
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_write_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_file, target_offset, length, flags,
      options));
  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_device_queue_host_call(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  // TODO(hexagon): if a native queue host call operation is available use that
  // instead. The emulated host call is horrendous and creates a new thread for
  // every requested host call. Even if native host call support is not
  // available an implementation should do _anything_ better than launching a
  // thread per call (polling threads, worker pools, etc).
  return iree_hal_device_queue_emulated_host_call(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      call, args, flags);
}

static iree_status_t iree_hal_hexagon_device_queue_dispatch(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t *executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  // TODO(hexagon): if a native queue dispatch operation is available use that
  // instead. The emulated dispatch creates a command buffer and executes it and
  // it's best if the extra recording/upload/allocation time can be avoided.
  return iree_hal_device_queue_emulated_dispatch(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      executable, function, config, constants, bindings, flags);
}

static void
iree_hal_hexagon_device_helper_unmap(const iree_hal_buffer_binding_t *bindings,
                                     iree_host_size_t binding_cnt) {
  for (iree_host_size_t idx = 0; idx < binding_cnt; ++idx) {
    iree_hal_buffer_binding_t binding = bindings[idx];
    if (iree_status_is_ok(iree_hal_buffer_binding_normalize(&binding))) {
      iree_hal_hexagon_buffer_unmap_from_dsp(binding.buffer);
    }
  }
}

static iree_status_t iree_hal_hexagon_device_queue_execute_cmd_buf(
    iree_hal_hexagon_device_t *device,
    iree_hal_command_buffer_t *command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  rpc_command_buffer_handle_t rpc_command_buffer_handle = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hexagon_command_buffer_get_rpc_handle(
      command_buffer, &rpc_command_buffer_handle));

  // Note: We cannot normalize buffer bindings in-place up-front here, because
  // the bindings passed to us here are const. So we have to normalize on the
  // fly where we use each binding.

  // Map all buffers in the binding table to the DSP.
  for (iree_host_size_t idx = 0; idx < binding_table.count; ++idx) {
    int fd = -1;
    iree_hal_buffer_binding_t binding = binding_table.bindings[idx];
    iree_status_t status = iree_hal_buffer_binding_normalize(&binding);
    if (iree_status_is_ok(status)) {
      status = iree_hal_hexagon_buffer_map_to_dsp(binding.buffer, &fd);
    }
    if (!iree_status_is_ok(status)) {
      // went wrong for one buffer, unmap the ones already mapped
      iree_hal_hexagon_device_helper_unmap(binding_table.bindings, idx);
      return status;
    }
  }

  // Get size of serialized binding data.
  iree_host_size_t bind_tab_size = 0;
  iree_status_t status =
      iree_hal_hexagon_bindings_serialize_prep(&binding_table, &bind_tab_size);
  if (!iree_status_is_ok(status)) {
    iree_hal_hexagon_device_helper_unmap(binding_table.bindings,
                                         binding_table.count);
    return status;
  }

  // Allocate RPC memory buffer for serialized binding data.
  if (bind_tab_size > INT_MAX /* max size supported by rpcmem_alloc() */) {
    iree_hal_hexagon_device_helper_unmap(binding_table.bindings,
                                         binding_table.count);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "serialized binding table too big");
  }
  uint8_t *bind_tab_data = rpcmem_alloc(
      RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)bind_tab_size);
  if (!bind_tab_data) {
    iree_hal_hexagon_device_helper_unmap(binding_table.bindings,
                                         binding_table.count);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "out of RPC memory");
  }

  // Serialize binding table to ARM/DSP data.
  iree_status_t status_serialize_exec =
      iree_hal_hexagon_bindings_serialize_exec(
          &binding_table, iree_hal_hexagon_buffer_get_fd_for_dsp, bind_tab_data,
          bind_tab_size);

  if (!iree_status_is_ok(status_serialize_exec)) {
    rpcmem_free(bind_tab_data);
    iree_hal_hexagon_device_helper_unmap(binding_table.bindings,
                                         binding_table.count);
    return status_serialize_exec;
  }

#if defined(IREE_HAL_HEXAGON_ENABLE_PROFILER)
  uint8_t *profiler_data = NULL;
  iree_host_size_t profiler_data_size = 0;
  iree_status_t status_profiler = iree_hal_hexagon_alloc_and_init_profiler_data(
      command_buffer, &device->options, &profiler_data, &profiler_data_size);
  if (!iree_status_is_ok(status_profiler)) {
    rpcmem_free(bind_tab_data);
    iree_hal_hexagon_device_helper_unmap(binding_table.bindings,
                                         binding_table.count);
    return status_profiler;
  }
#endif

#if defined(IREE_HAL_HEXAGON_ENABLE_PROFILER)
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "RPC call");
  // Call RPC on DSP to execute the kernel with profiler
  int dsp_err = hexagon_dsp_command_buffer_execute_profiler(
      device->rpc_session_handle, rpc_command_buffer_handle, bind_tab_data,
      bind_tab_size, profiler_data, profiler_data_size);
  rpcmem_free(bind_tab_data);
  iree_hal_hexagon_device_helper_unmap(binding_table.bindings,
                                       binding_table.count);

  if (dsp_err != AEE_SUCCESS) {
    rpcmem_free(profiler_data);
    iree_hal_hexagon_device_helper_unmap(binding_table.bindings,
                                         binding_table.count);
    return IREE_HAL_HEXAGON_MAKE_STATUS_FROM_DSP_ERR(
        dsp_err, "could not execute command buffer on DSP");
  }
  IREE_TRACE_ZONE_END(z0);
#else
  // Call RPC on DSP to execute the kernel.
  int dsp_err = hexagon_dsp_command_buffer_execute(
      device->rpc_session_handle, rpc_command_buffer_handle, bind_tab_data,
      bind_tab_size);

  rpcmem_free(bind_tab_data);
  iree_hal_hexagon_device_helper_unmap(binding_table.bindings,
                                       binding_table.count);

  if (dsp_err != AEE_SUCCESS) {
    return IREE_HAL_HEXAGON_MAKE_STATUS_FROM_DSP_ERR(
        dsp_err, "could not execute command buffer on DSP");
  }
#endif

#if defined(IREE_HAL_HEXAGON_ENABLE_PROFILER)
  iree_status_t status_exporting = iree_hal_hexagon_export_profiler_data(
      device->host_allocator, profiler_data, &device->tracy_context_id,
      &device->tracy_plot_context);

  rpcmem_free(profiler_data);
  iree_hal_hexagon_device_helper_unmap(binding_table.bindings,
                                       binding_table.count);

  if (!iree_status_is_ok(status_exporting)) {
    return status_exporting;
  }
#endif

  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_device_queue_execute(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t *command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): implement a wait, execute, and signal queue operation. The
  // queue affinity can be used to determine which top-level execution resources
  // are to be used when executing and it can be assumed that all resources
  // required for execution are accessible on those queues. If more than one
  // queue is specified the implementation may use any it prefers from the set.

  // TODO(hexagon): an optional binding table is provided for indirect command
  // buffers (those who have a binding_capacity > 0). The binding table must be
  // captured by the implementation as they may be mutated or freed by the
  // caller immediately after this call returns.

  // TODO(hexagon): do this async - callers may be submitting work to multiple
  // devices or queues on the same device from the same thread and blocking here
  // will prevent both concurrency and pipelining.

  // To get started, implement the functionality sequentially, outside of any
  // queue.

  // Wait for all semaphores for which we need to wait.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(
      wait_semaphore_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  // It is possible for command_buffer to be NULL. In this case, there is
  // nothing to be executed. Just wait skip the execution part. (Still handle
  // the semaphores.)
  if (command_buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_hexagon_device_queue_execute_cmd_buf(
        device, command_buffer, binding_table, flags));
  }

  // Signal all semaphores we can signal now.
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_list_signal(signal_semaphore_list, /*frontier=*/NULL));

  return iree_ok_status();
}

static iree_status_t
iree_hal_hexagon_device_queue_flush(iree_hal_device_t *base_device,
                                    iree_hal_queue_affinity_t queue_affinity) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): though buffering queue operations is not recommended if any
  // buffering has been performed it must be flushed here. Callers may be
  // indicating that they are about to suspend themselves waiting for submitted
  // work to complete.
  (void)device;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "queue flush not implemented");

  return status;
}

static iree_status_t iree_hal_hexagon_device_profiling_begin(
    iree_hal_device_t *base_device,
    const iree_hal_device_profiling_options_t *options) {
  // Currently, allocations are happening before command buffer execution, when
  // we already know the number of dispatches and can therefore allocate the
  // necessary size. This assumes we only have a single command buffer being
  // executed simultaneously (this is also assumed by the PMU configuration in
  // the DSP side).
  // While working with synchronous command buffer execution, this
  // is ok, but will need to be changed in the future in case we switch to
  // multiple command buffers running asynchronously. An example of this is
  // available in the Vulkan runtime.
  // These functions are meant to be used with fixed size and global allocations
  // per benchmark

  return iree_ok_status();
}

static iree_status_t
iree_hal_hexagon_device_profiling_flush(iree_hal_device_t *base_device) {
  // Read iree_hal_hexagon_device_profiler_begin.

  return iree_ok_status();
}

static iree_status_t
iree_hal_hexagon_device_profiling_end(iree_hal_device_t *base_device) {
  // Read iree_hal_hexagon_device_profiler_begin.

  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_hexagon_device_vtable = {
    .destroy = iree_hal_hexagon_device_destroy,
    .id = iree_hal_hexagon_device_id,
    .host_allocator = iree_hal_hexagon_device_host_allocator,
    .device_allocator = iree_hal_hexagon_device_allocator,
    .replace_device_allocator = iree_hal_hexagon_replace_device_allocator,
    .replace_channel_provider = iree_hal_hexagon_replace_channel_provider,
    .trim = iree_hal_hexagon_device_trim,
    .query_i64 = iree_hal_hexagon_device_query_i64,
    .query_capabilities = iree_hal_hexagon_device_query_capabilities,
    .topology_info = iree_hal_hexagon_device_topology_info,
    .refine_topology_edge = iree_hal_hexagon_device_refine_topology_edge,
    .assign_topology_info = iree_hal_hexagon_device_assign_topology_info,
    .create_channel = iree_hal_hexagon_device_create_channel,
    .create_command_buffer = iree_hal_hexagon_device_create_command_buffer,
    .create_event = iree_hal_hexagon_device_create_event,
    .create_executable_cache = iree_hal_hexagon_device_create_executable_cache,
    .import_file = iree_hal_hexagon_device_import_file,
    .create_semaphore = iree_hal_hexagon_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_hexagon_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_hexagon_device_queue_alloca,
    .queue_dealloca = iree_hal_hexagon_device_queue_dealloca,
    .queue_fill = iree_hal_hexagon_device_queue_fill,
    .queue_update = iree_hal_hexagon_device_queue_update,
    .queue_copy = iree_hal_hexagon_device_queue_copy,
    .queue_read = iree_hal_hexagon_device_queue_read,
    .queue_write = iree_hal_hexagon_device_queue_write,
    .queue_host_call = iree_hal_hexagon_device_queue_host_call,
    .queue_dispatch = iree_hal_hexagon_device_queue_dispatch,
    .queue_execute = iree_hal_hexagon_device_queue_execute,
    .queue_flush = iree_hal_hexagon_device_queue_flush,
    .profiling_begin = iree_hal_hexagon_device_profiling_begin,
    .profiling_flush = iree_hal_hexagon_device_profiling_flush,
    .profiling_end = iree_hal_hexagon_device_profiling_end,
};
