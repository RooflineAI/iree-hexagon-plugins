// Copyright 2025 RooflineAI GmbH
#include "hexagon/device.h"

#include "hexagon/allocator.h"
#include "hexagon/api.h"
#include "hexagon/channel.h"
#include "hexagon/command_buffer.h"
#include "hexagon/event.h"
#include "hexagon/executable.h"
#include "hexagon/executable_cache.h"
#include "hexagon/semaphore.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/queue_emulation.h"
#include "iree/hal/utils/queue_host_call_emulation.h"

// Hexagon SDK includes
#include <AEEStdErr.h>
#include <iree/base/string_view.h>
#include <remote.h>

#include "hexagon_dsp.h"
#include "rpc_types_impl.h"

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
}

static iree_status_t iree_hal_hexagon_device_options_verify(
    const iree_hal_hexagon_device_options_t *options) {
  // TODO(hexagon): verify that the parameters are within expected ranges and
  // any requested features are supported.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hexagon_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;
  iree_hal_hexagon_domain_id_t domain_id;

  /// parsed device options, copied here by value due to short lifetime of
  /// options arg passed to init
  iree_hal_hexagon_device_options_t options;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t *device_allocator;

  /// Connection to DSP / DSP RPC session.
  iree_hal_hexagon_rpc_session_t rpc_session;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t *channel_provider;

  // + trailing identifier string storage
} iree_hal_hexagon_device_t;

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

void iree_hal_hexagon_fill_device_info(iree_string_view_t identifier,
                                       iree_hal_hexagon_domain_id_t domain_id,
                                       iree_hal_device_info_t *device_info) {
  device_info->device_id = domain_id;
  device_info->name = iree_make_cstring_view(
      iree_hal_hexagon_get_domain_name_or_unknown(domain_id));
  device_info->path = device_info->name;
  device_info->identifier = identifier;
  device_info->device_index = domain_id;
}

int iree_hal_hexagon_pd_status_callback(void *context, int domain, int session,
                                        remote_rpc_status_flags_t status) {
  iree_hal_hexagon_device_t *device = (iree_hal_hexagon_device_t *)context;
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
  }
  fprintf(stderr, "iree-run-module: hexagon: DSP status: %s\n", dsp_status);
  return nErr;
}

static iree_status_t iree_hal_hexagon_request_status_notifications(
    int domain_id, void *context,
    int (*notify_callback)(void *context, int domain, int session,
                           remote_rpc_status_flags_t status)) {
  if (!remote_handle_control) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "DSP API not available");
  }

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
    return iree_make_status(
        IREE_STATUS_UNKNOWN,
        "querying for DSP status notification support failed");
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
      device->host_allocator, &device->device_allocator));

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
  const char *suffix = NULL;
  if (!domain_id_to_uri_suffix(device->domain_id, &suffix)) {
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "cannot convert DSP domain ID to URI suffix");
  }
  iree_host_size_t uri_sz = strlen(hexagon_dsp_URI) + strlen(suffix) + 1;
  char *uri = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(device->host_allocator, uri_sz, &uri));
  strcpy(uri, hexagon_dsp_URI);
  strcat(uri, suffix);

  // Open the actual DSP RPC session.
  int dsp_open_ret = hexagon_dsp_open(uri, &device->rpc_session.rpc_handle);
  iree_allocator_free(device->host_allocator, uri);
  if (dsp_open_ret != AEE_SUCCESS) {
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "opening RPC session with DSP failed");
  }
  device->rpc_session.rpc_is_open = 1;

  return iree_ok_status();
}

iree_status_t iree_hal_hexagon_device_create(
    iree_string_view_t identifier, iree_hal_hexagon_domain_id_t domain_id,
    const iree_hal_hexagon_device_options_t *options,
    iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
  IREE_ASSERT_ARGUMENT(options);
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

  iree_status_t status = iree_hal_hexagon_device_set_up(domain_id, device);

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

  if (device->rpc_session.rpc_is_open) {
    hexagon_dsp_close(device->rpc_session.rpc_handle);
  }

  iree_hal_allocator_release(device->device_allocator);
  iree_hal_channel_provider_release(device->channel_provider);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t
iree_hal_hexagon_device_id(iree_hal_device_t *base_device) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);
  return iree_make_cstring_view(
      iree_hal_hexagon_get_domain_name_or_unknown(device->domain_id));
}

static iree_hal_device_info_t
iree_hal_hexagon_device_info(iree_hal_device_t *base_device) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);
  iree_hal_device_info_t device_info;
  iree_hal_hexagon_fill_device_info(device->identifier, device->domain_id,
                                    &device_info);
  return device_info;
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
      out_command_buffer);
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
    iree_loop_t loop, iree_hal_executable_cache_t **out_executable_cache) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): pass any additional resources required during executable
  // creation or cache management.
  (void)device;

  return iree_hal_hexagon_executable_cache_create(
      identifier, iree_hal_device_host_allocator(base_device),
      out_executable_cache);
}

static iree_status_t iree_hal_hexagon_device_import_file(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t *handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t **out_file) {
  // TODO(hexagon): if the implementation supports native file operations
  // definitely prefer that. The emulated file I/O present here as a default is
  // inefficient. The queue affinity specifies which queues may access the file
  // via read and write queue operations.
  return iree_hal_file_from_handle(
      iree_hal_device_allocator(base_device), queue_affinity, access, handle,
      iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_hexagon_device_create_semaphore(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t **out_semaphore) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): pass any additional resources required to create or track
  // the semaphore. The implementation could pool semaphores here.
  (void)device;

  return iree_hal_hexagon_semaphore_create(queue_affinity, initial_value, flags,
                                           device->host_allocator,
                                           out_semaphore);
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
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t **IREE_RESTRICT out_buffer) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): perform an allocation of a transient buffer in queue order.
  // The allocation may be used on any queue set in the provided queue affinity.
  // Deallocation via queue_dealloca is preferred but users are allowed to
  // deallocate the buffer on the host via iree_hal_buffer_release even if they
  // allocated it with queue_alloca.
  (void)device;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "queue alloca not implemented");

  return status;
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
  (void)device;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "queue dealloca not implemented");

  return status;
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
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_read_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_file, source_offset, target_buffer, target_offset, length, flags,
      options));
  return loop_status;
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
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_write_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_file, target_offset, length, flags,
      options));
  return loop_status;
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
    iree_hal_executable_t *executable, int32_t entry_point,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  // TODO(hexagon): if a native queue dispatch operation is available use that
  // instead. The emulated dispatch creates a command buffer and executes it and
  // it's best if the extra recording/upload/allocation time can be avoided.
  return iree_hal_device_queue_emulated_dispatch(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      executable, entry_point, config, constants, bindings, flags);
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

  (void)device;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "queue execute not implemented");

  return status;
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

static iree_status_t iree_hal_hexagon_device_wait_semaphores(
    iree_hal_device_t *base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout,
    iree_hal_wait_flags_t flags) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): implement multi-wait as either an ALL (AND) or ANY (OR)
  // operation. Semaphores are expected to be compatible with the device today
  // and may come from other device instances provided by the same driver or
  // have been imported by a device instance.

  // TODO(hexagon): if any semaphore has a failure status set return
  // `iree_status_from_code(IREE_STATUS_ABORTED)`. Avoid a full status as it may
  // capture a backtrace and allocate and callers are expected to follow up a
  // failed wait with a query to get the status.

  // TODO(hexagon): prefer having a fast-path for if the semaphores are
  // known-signaled in user-mode. This can usually avoid syscalls/ioctls and
  // potential context switches in polling cases.

  // TODO(hexagon): check for `iree_timeout_is_immediate(timeout)` and return
  // immediately if the condition is not satisfied before waiting with
  // `iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED)`. Prefer the raw code
  // status instead of a full status object as immediate timeouts are used when
  // polling and a full status may capture a backtrace and allocate.

  (void)device;
  iree_status_t status = iree_make_status(
      IREE_STATUS_UNIMPLEMENTED, "semaphore multi-wait not implemented");

  return status;
}

static iree_status_t iree_hal_hexagon_device_profiling_begin(
    iree_hal_device_t *base_device,
    const iree_hal_device_profiling_options_t *options) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): set implementation-defined device or global profiling modes.
  // This will be paired with a profiling_end call at some point in the future.
  // Hosting applications may periodically call profiling_flush.
  (void)device;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "device profiling not implemented");

  return status;
}

static iree_status_t
iree_hal_hexagon_device_profiling_flush(iree_hal_device_t *base_device) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): flush if needed. May be no-op. Any accumulated profiling
  // information should be carried across the flush but the event can be used to
  // reclaim resources or perform other expensive bookkeeping. Benchmarks, for
  // example, are expected to call this periodically with their timing
  // suspended.
  (void)device;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "device profiling not implemented");

  return status;
}

static iree_status_t
iree_hal_hexagon_device_profiling_end(iree_hal_device_t *base_device) {
  iree_hal_hexagon_device_t *device = iree_hal_hexagon_device_cast(base_device);

  // TODO(hexagon): unset whatever profiling_begin set, if anything. May be
  // no-op.
  (void)device;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "device profiling not implemented");

  return status;
}

static const iree_hal_device_vtable_t iree_hal_hexagon_device_vtable = {
    .destroy = iree_hal_hexagon_device_destroy,
    .id = iree_hal_hexagon_device_id,
    .info = iree_hal_hexagon_device_info,
    .host_allocator = iree_hal_hexagon_device_host_allocator,
    .device_allocator = iree_hal_hexagon_device_allocator,
    .replace_device_allocator = iree_hal_hexagon_replace_device_allocator,
    .replace_channel_provider = iree_hal_hexagon_replace_channel_provider,
    .trim = iree_hal_hexagon_device_trim,
    .query_i64 = iree_hal_hexagon_device_query_i64,
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
    .wait_semaphores = iree_hal_hexagon_device_wait_semaphores,
    .profiling_begin = iree_hal_hexagon_device_profiling_begin,
    .profiling_flush = iree_hal_hexagon_device_profiling_flush,
    .profiling_end = iree_hal_hexagon_device_profiling_end,
};
