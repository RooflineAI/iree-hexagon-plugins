// Copyright 2025 RooflineAI GmbH

#include "hexagon/allocator.h"

#include <iree/base/bitfield.h>
#include <iree/base/status.h>
#include <iree/hal/allocator.h>
#include <iree/hal/buffer.h>
#include <iree/hal/queue.h>

#include "hexagon/api.h"
#include "hexagon/buffer.h"
#include "hexagon/device.h"
#include "hexagon/mem_alloc.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_allocator_t
//===----------------------------------------------------------------------===//

// TODO(hexagon): use one ID per address space or pool - each shows as a
// different track in tracing tools.
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char *IREE_HAL_HEXAGON_ALLOCATOR_ID =
    "{Qualcomm Hexagon} unpooled";
#endif // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

typedef struct iree_hal_hexagon_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_hexagon_device_t *device; ///< unowned

  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_hexagon_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_hexagon_allocator_vtable;

static iree_hal_hexagon_allocator_t *
iree_hal_hexagon_allocator_cast(iree_hal_allocator_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hexagon_allocator_vtable);
  return (iree_hal_hexagon_allocator_t *)base_value;
}

static void
iree_hal_hexagon_buffer_release_callback(void *user_data,
                                         iree_hal_buffer_t *buffer) {
  iree_hal_hexagon_allocator_t *allocator =
      iree_hal_hexagon_allocator_cast((iree_hal_allocator_t *)user_data);
  void *impl_ptr = iree_hal_hexagon_buffer_impl_ptr(buffer);
  (void)impl_ptr;

  // TODO(hexagon): if the buffer was imported then this accounting may need to
  // be conditional depending on the implementation.
  bool was_imported = false;
  if (!was_imported) {
    IREE_TRACE_FREE_NAMED(IREE_HAL_HEXAGON_ALLOCATOR_ID, impl_ptr);
    IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
        &allocator->statistics, iree_hal_buffer_memory_type(buffer),
        iree_hal_buffer_allocation_size(buffer)));
  }
}

iree_status_t
iree_hal_hexagon_allocator_create(iree_allocator_t host_allocator,
                                  iree_hal_hexagon_device_t *device,
                                  iree_hal_allocator_t **out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_allocator = NULL;

  iree_hal_hexagon_allocator_t *allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void **)&allocator));
  iree_hal_resource_initialize(&iree_hal_hexagon_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;
  allocator->device = device;

  // TODO(hexagon): query device heaps, supported features (concurrent
  // access/etc), and prepare any pools that will be used during allocation.
  // It's expected that most failures that occur after creation are allocation
  // request-specific so preparing here will help keep the errors more
  // localized.
  // For now, there are no pools to be set up, so nothing to be done here.
  iree_status_t status = iree_ok_status();

  if (iree_status_is_ok(status)) {
    *out_allocator = (iree_hal_allocator_t *)allocator;
  } else {
    iree_hal_allocator_release((iree_hal_allocator_t *)allocator);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_hexagon_allocator_destroy(
    iree_hal_allocator_t *IREE_RESTRICT base_allocator) {
  IREE_ASSERT_ARGUMENT(base_allocator);
  iree_hal_hexagon_allocator_t *allocator =
      iree_hal_hexagon_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(allocator->host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_hexagon_allocator_host_allocator(
    const iree_hal_allocator_t *IREE_RESTRICT base_allocator) {
  iree_hal_hexagon_allocator_t *allocator =
      (iree_hal_hexagon_allocator_t *)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_hexagon_allocator_trim(
    iree_hal_allocator_t *IREE_RESTRICT base_allocator) {
  iree_hal_hexagon_allocator_t *allocator =
      (iree_hal_hexagon_allocator_t *)base_allocator;

  // TODO(hexagon): if the allocator is retaining any unused resources they
  // should be dropped here. If the underlying implementation has pools or
  // caches it should be notified that a trim is requested. This is called in
  // low-memory situations or when IREE is not going to be used for awhile (low
  // power modes or suspension).
  (void)allocator;

  return iree_ok_status();
}

static void iree_hal_hexagon_allocator_query_statistics(
    iree_hal_allocator_t *IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t *IREE_RESTRICT out_statistics) {
  IREE_STATISTICS({
    iree_hal_hexagon_allocator_t *allocator =
        iree_hal_hexagon_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
    // TODO(hexagon): update statistics (merge).
  });
}

static iree_status_t iree_hal_hexagon_allocator_select_mem_kind(
    iree_hal_memory_type_t memory_type,
    iree_hal_hexagon_mem_kind_t *out_mem_kind) {
  // Memory does not need to be device-visible.
  // -> Use host allocator memory. It is automatically cached and coherent, so
  // no need to check for those flags.
  if (!iree_any_bit_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    *out_mem_kind = IREE_HAL_HEXAGON_MEM_KIND_HOST;
    return iree_ok_status();
  }

  // Memory does not need to be host-visble.
  // -> Use device HAP_malloc memory.
  if (!iree_any_bit_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    *out_mem_kind = IREE_HAL_HEXAGON_MEM_KIND_DEVICE_HAP;
    return iree_ok_status();
  }

  // Memory needs to be host-visible and device-visible.
  // -> Use RPC memory. mem_alloc.c uses RPCMEM_DEFAULT_FLAGS, which results in
  // cached and coherent memory according to
  // Hexagon_SDK/6.3.0.0/docs/software/os/os_support_dsp.html#cache-management
  *out_mem_kind = IREE_HAL_HEXAGON_MEM_KIND_RPCMEM;
  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_allocator_query_memory_heaps(
    iree_hal_allocator_t *IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t *IREE_RESTRICT heaps,
    iree_host_size_t *IREE_RESTRICT out_count) {
  iree_hal_hexagon_allocator_t *allocator =
      iree_hal_hexagon_allocator_cast(base_allocator);

  // TODO(hexagon): return heap information. This is called at least once with a
  // capacity that may be 0 (indicating a query for the total count) and the
  // heaps should only be populated if capacity is sufficient to store all of
  // them.
  (void)allocator;
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "heap query not implemented");

  return status;
}

static iree_hal_buffer_compatibility_t
iree_hal_hexagon_allocator_query_buffer_compatibility(
    iree_hal_allocator_t *IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t *IREE_RESTRICT params,
    iree_device_size_t *IREE_RESTRICT allocation_size) {
  iree_hal_hexagon_allocator_t *allocator =
      iree_hal_hexagon_allocator_cast(base_allocator);

  // TODO(hexagon): set compatibility rules based on the implementation.
  // Note that the user may have requested that the allocator place the
  // allocation based on whatever is optimal for the indicated usage by
  // including the IREE_HAL_MEMORY_TYPE_OPTIMAL flag. It's still required that
  // the implementation meet all the requirements but it is free to place it in
  // either host or device memory so long as the appropriate bits are updated to
  // indicate where it landed.
  (void)allocator;
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_NONE;

  // We are now optimal.
  params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;

  // Compatibility depends on selected memory kind, so find memory kind first
  // and fill compatibility based on selected kind. If there is no kind of
  // memory for the requested type, there is no compatibility with anything.
  iree_hal_hexagon_mem_kind_t mem_kind = 0;
  if (iree_status_is_ok(iree_hal_hexagon_allocator_select_mem_kind(
          params->type, &mem_kind))) {
    switch (mem_kind) {
    case IREE_HAL_HEXAGON_MEM_KIND_HOST:
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;
      // TODO: Add other things possible with host allocator memory.
      break;
    case IREE_HAL_HEXAGON_MEM_KIND_RPCMEM:
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
                       IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH |
                       IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
      // TODO: Add other things possible with RPC memory.
      break;
    case IREE_HAL_HEXAGON_MEM_KIND_DEVICE_HAP:
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
                       IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH |
                       IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
      // TODO: Add other things possible with device HAP_malloc memory.
      break;
    }
  }

  // Guard against the corner case where the requested buffer size is 0. The
  // application is unlikely to do anything when requesting a 0-byte buffer; but
  // it can happen in real world use cases. So we should at least not crash.
  if (*allocation_size == 0)
    *allocation_size = 4;

  return compatibility;
}

static iree_status_t iree_hal_hexagon_allocator_allocate_buffer(
    iree_hal_allocator_t *IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t *IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t **IREE_RESTRICT out_buffer) {
  iree_hal_hexagon_allocator_t *allocator =
      iree_hal_hexagon_allocator_cast(base_allocator);

  // Coerce options into those required by the current device.
  iree_hal_buffer_params_t compat_params = *params;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_hexagon_allocator_query_buffer_compatibility(
          base_allocator, &compat_params, &allocation_size);
  if (!iree_all_bits_set(compatibility,
                         IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    // TODO(benvanik): make a helper for this.
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t temp0, temp1, temp2;
    iree_string_view_t memory_type_str =
        iree_hal_memory_type_format(params->type, &temp0);
    iree_string_view_t usage_str =
        iree_hal_buffer_usage_format(params->usage, &temp1);
    iree_string_view_t compatibility_str =
        iree_hal_buffer_compatibility_format(compatibility, &temp2);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters; "
        "memory_type=%.*s, usage=%.*s, compatibility=%.*s",
        (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
        usage_str.data, (int)compatibility_str.size, compatibility_str.data);
#else
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters");
#endif // IREE_STATUS_MODE
  }

  // TODO(hexagon): allocate the underlying device memory. The impl_ptr is just
  // used for accounting and can be an opaque value (handle/etc) so long as it
  // is consistent between the alloc and free and unique to the buffer while it
  // is live. An example iree_hal_hexagon_buffer_wrap is provided that can be
  // used for implementations that are managing memory using underlying
  // allocators and just wrapping those device pointers in the HAL buffer type.
  // Other implementations that require more tracking can provide their own
  // buffer types that do such tracking for them.
  iree_hal_buffer_placement_t placement = {
      .device = (iree_hal_device_t *)allocator->device,
      .queue_affinity = compat_params.queue_affinity
                            ? compat_params.queue_affinity
                            : IREE_HAL_QUEUE_AFFINITY_ANY,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE};
  iree_hal_memory_type_t memory_type = compat_params.type;
  iree_hal_memory_access_t allowed_access = compat_params.access;
  iree_hal_buffer_usage_t allowed_usage = compat_params.usage;

  // Select kind of Hexagon memory based on memory type.
  iree_hal_hexagon_mem_kind_t mem_kind = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hexagon_allocator_select_mem_kind(
      compat_params.type, &mem_kind));

  iree_hal_hexagon_domain_id_t domain_id =
      iree_hal_hexagon_device_get_domain_id(allocator->device);
  rpc_session_handle_t rpc_session_handle =
      iree_hal_hexagon_device_get_rpc_session_handle(allocator->device);
  iree_hal_hexagon_mem_alloc_t *alloc = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_hexagon_mem_alloc_create(
      allocator->host_allocator, domain_id, rpc_session_handle, mem_kind,
      allocation_size, &alloc));

  // Allocate an IREE buffer data structure and put the allocated memory into
  // it.
  iree_hal_buffer_t *buffer = NULL;
  // This release callback takes care of profiling statistics
  iree_hal_buffer_release_callback_t release_callback = {
      .fn = iree_hal_hexagon_buffer_release_callback,
      .user_data = (void *)base_allocator,
  };
  iree_status_t status = iree_hal_hexagon_buffer_wrap(
      alloc, placement, memory_type, allowed_access, allowed_usage,
      allocation_size,
      /* byte_offset */ 0, /* byte_length */ allocation_size, release_callback,
      allocator->host_allocator, allocator->device, &buffer);

  if (iree_status_is_ok(status)) {
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_HEXAGON_ALLOCATOR_ID,
                           iree_hal_hexagon_mem_alloc_impl_ptr(alloc),
                           allocation_size);
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, compat_params.type, allocation_size));
    *out_buffer = buffer;
  } else {
    if (buffer)
      iree_hal_buffer_release(buffer);
  }
  iree_hal_hexagon_mem_alloc_release(alloc);
  return status;
}

static void iree_hal_hexagon_allocator_deallocate_buffer(
    iree_hal_allocator_t *IREE_RESTRICT base_allocator,
    iree_hal_buffer_t *IREE_RESTRICT base_buffer) {
  // NOTE: tracing/statistics are handled by the buffer release callback.
  iree_hal_buffer_destroy(base_buffer);
}

static iree_status_t iree_hal_hexagon_allocator_import_buffer(
    iree_hal_allocator_t *IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t *IREE_RESTRICT params,
    iree_hal_external_buffer_t *IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t **IREE_RESTRICT out_buffer) {
  iree_hal_hexagon_allocator_t *allocator =
      iree_hal_hexagon_allocator_cast(base_allocator);

  // Coerce options into those required by the current device.
  iree_hal_buffer_params_t compat_params = *params;
  iree_device_size_t allocation_size = external_buffer->size;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_hexagon_allocator_query_buffer_compatibility(
          base_allocator, &compat_params, &allocation_size);
  if (!iree_all_bits_set(compatibility,
                         IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE)) {
    // TODO(benvanik): make a helper for this.
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t temp0, temp1, temp2;
    iree_string_view_t memory_type_str =
        iree_hal_memory_type_format(params->type, &temp0);
    iree_string_view_t usage_str =
        iree_hal_buffer_usage_format(params->usage, &temp1);
    iree_string_view_t compatibility_str =
        iree_hal_buffer_compatibility_format(compatibility, &temp2);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot import a buffer with the given parameters; "
        "memory_type=%.*s, usage=%.*s, compatibility=%.*s",
        (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
        usage_str.data, (int)compatibility_str.size, compatibility_str.data);
#else
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot import a buffer with the given parameters");
#endif // IREE_STATUS_MODE
  }

  // TODO(hexagon): switch on external_buffer->type and import the buffer. See
  // the headers for more information on semantics. Most implementations can
  // service IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION by just wrapping
  // the underlying device pointer. Those that can service
  // IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION may be able to avoid a lot of
  // additional copies when moving data around between host and device or across
  // devices from different drivers.
  (void)allocator;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "external buffer type not supported");

  return status;
}

static iree_status_t iree_hal_hexagon_allocator_export_buffer(
    iree_hal_allocator_t *IREE_RESTRICT base_allocator,
    iree_hal_buffer_t *IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t *IREE_RESTRICT out_external_buffer) {
  iree_hal_hexagon_allocator_t *allocator =
      iree_hal_hexagon_allocator_cast(base_allocator);

  // TODO(hexagon): switch on requested_type and export as appropriate. Most
  // implementations can service IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION
  // by just exposing the underlying device pointer. Those that can service
  // IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION may be able to avoid a lot of
  // additional copies when moving data around between host and device or across
  // devices from different drivers.
  (void)allocator;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "external buffer type not supported");
}

static const iree_hal_allocator_vtable_t iree_hal_hexagon_allocator_vtable = {
    .destroy = iree_hal_hexagon_allocator_destroy,
    .host_allocator = iree_hal_hexagon_allocator_host_allocator,
    .trim = iree_hal_hexagon_allocator_trim,
    .query_statistics = iree_hal_hexagon_allocator_query_statistics,
    .query_memory_heaps = iree_hal_hexagon_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        iree_hal_hexagon_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_hexagon_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_hexagon_allocator_deallocate_buffer,
    .import_buffer = iree_hal_hexagon_allocator_import_buffer,
    .export_buffer = iree_hal_hexagon_allocator_export_buffer,
};
