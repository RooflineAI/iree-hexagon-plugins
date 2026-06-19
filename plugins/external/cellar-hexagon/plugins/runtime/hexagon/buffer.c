// Copyright 2025 RooflineAI GmbH

#include "hexagon/buffer.h"

#include <iree/base/allocator.h>
#include <iree/base/status.h>
#include <iree/hal/buffer.h>

#include "hexagon/api.h"
#include "hexagon/mem_alloc.h"
#include "hexagon/serialize/rpc_types.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hexagon_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;
  iree_hal_buffer_release_callback_t release_callback;
  iree_hal_hexagon_mem_alloc_t *alloc;
} iree_hal_hexagon_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_hexagon_buffer_vtable;

static bool iree_hal_hexagon_buffer_isa(iree_hal_buffer_t *base_buffer) {
  return iree_hal_resource_is(&base_buffer->resource,
                              &iree_hal_hexagon_buffer_vtable);
}

static iree_hal_hexagon_buffer_t *
iree_hal_hexagon_buffer_cast(iree_hal_buffer_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hexagon_buffer_vtable);
  return (iree_hal_hexagon_buffer_t *)base_value;
}

static const iree_hal_hexagon_buffer_t *
iree_hal_hexagon_buffer_const_cast(const iree_hal_buffer_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hexagon_buffer_vtable);
  return (const iree_hal_hexagon_buffer_t *)base_value;
}

iree_status_t iree_hal_hexagon_buffer_wrap(
    iree_hal_hexagon_mem_alloc_t *alloc, iree_hal_buffer_placement_t placement,
    iree_hal_memory_type_t memory_type, iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_hexagon_device_t *device,
    iree_hal_buffer_t **out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  iree_hal_hexagon_buffer_t *buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void **)&buffer));
  iree_hal_buffer_initialize(placement, &buffer->base, allocation_size,
                             byte_offset, byte_length, memory_type,
                             allowed_access, allowed_usage,
                             &iree_hal_hexagon_buffer_vtable, &buffer->base);
  buffer->host_allocator = host_allocator;
  buffer->release_callback = release_callback;
  buffer->alloc = NULL;

  // TODO(hexagon): retain or take ownership of provided handles/pointers/etc.
  // Implementations may want to pass in an internal buffer type discriminator
  // if there are multiple or use different top-level iree_hal_buffer_t
  // implementations.
  if (alloc) {
    buffer->alloc = alloc;
    iree_hal_hexagon_mem_alloc_retain(alloc);
  }
  iree_status_t status = iree_ok_status();

  if (iree_status_is_ok(status)) {
    *out_buffer = &buffer->base;
  } else {
    iree_hal_buffer_release(&buffer->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void *iree_hal_hexagon_buffer_impl_ptr(iree_hal_buffer_t *base_buffer) {
  iree_hal_hexagon_buffer_t *buffer = iree_hal_hexagon_buffer_cast(base_buffer);
  return iree_hal_hexagon_mem_alloc_impl_ptr(buffer->alloc);
}

iree_status_t iree_hal_hexagon_buffer_map_to_dsp(iree_hal_buffer_t *base_buffer,
                                                 int *out_fd) {
  if (!iree_hal_hexagon_buffer_isa(base_buffer)) {
    *out_fd = -1;
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "non-Hexagon buffers cannot be mapped to DSP");
  }
  return iree_hal_hexagon_mem_alloc_map(
      iree_hal_hexagon_buffer_cast(base_buffer)->alloc, out_fd);
}

iree_status_t
iree_hal_hexagon_buffer_get_fd_for_dsp(iree_hal_buffer_t *base_buffer,
                                       int *out_fd) {
  if (!iree_hal_hexagon_buffer_isa(base_buffer)) {
    *out_fd = -1;
    return iree_make_status(
        IREE_STATUS_INCOMPATIBLE,
        "non-Hexagon buffers do not have a file descriptor for the DSP");
  }
  return iree_hal_hexagon_mem_alloc_get_fd(
      iree_hal_hexagon_buffer_cast(base_buffer)->alloc, out_fd);
}

iree_status_t
iree_hal_hexagon_buffer_unmap_from_dsp(iree_hal_buffer_t *base_buffer) {
  if (!iree_hal_hexagon_buffer_isa(base_buffer)) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "non-Hexagon buffers cannot be unmapped from DSP");
  }
  return iree_hal_hexagon_mem_alloc_unmap(
      iree_hal_hexagon_buffer_cast(base_buffer)->alloc);
}

static void iree_hal_hexagon_buffer_destroy(iree_hal_buffer_t *base_buffer) {
  iree_hal_hexagon_buffer_t *buffer = iree_hal_hexagon_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Optionally call a release callback when the buffer is destroyed. Not all
  // implementations may require this but it's cheap and provides additional
  // flexibility.
  if (buffer->release_callback.fn) {
    buffer->release_callback.fn(buffer->release_callback.user_data,
                                base_buffer);
  }

  iree_hal_hexagon_mem_alloc_release(buffer->alloc);
  buffer->alloc = NULL;

  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_hexagon_buffer_map_range(
    iree_hal_buffer_t *base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t *mapping) {
  iree_hal_hexagon_buffer_t *buffer = iree_hal_hexagon_buffer_cast(base_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(base_buffer),
      mapping_mode == IREE_HAL_MAPPING_MODE_PERSISTENT
          ? IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
          : IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));

  // TODO(hexagon): perform mapping as described. Note that local-to-buffer
  // range adjustment may be required. The resulting mapping is populated with
  // standard information such as contents indicating the host addressable
  // memory range of the mapped buffer and implementation-specific information
  // if additional resources are required. iree_hal_buffer_emulated_map_range
  // can be used by implementations that have no way of providing host pointers
  // at a large cost (alloc + device->host transfer on map and host->device
  // transfer + dealloc on umap). Try not to use that.

  // On Hexagon, there is currently no need to do an actual mapping - either the
  // memory is mapped to host already (host memory and RPC memory) or it is not
  // possible to map the memory to host (device memory).

  // Get host span - if available.
  iree_byte_span_t host_span;
  IREE_RETURN_IF_ERROR(
      iree_hal_hexagon_mem_alloc_get_host_span(buffer->alloc, &host_span));

  // Check requested offset and size.
  if (local_byte_offset > host_span.data_length ||
      local_byte_offset + local_byte_length > host_span.data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "access beyond end of buffer requested");
  }

  // Return a mapping to the requested part of the buffer.
  mapping->contents = iree_make_byte_span(host_span.data + local_byte_offset,
                                          local_byte_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_buffer_unmap_range(
    iree_hal_buffer_t *base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t *mapping) {
  iree_hal_hexagon_buffer_t *buffer = iree_hal_hexagon_buffer_cast(base_buffer);

  // TODO(hexagon): reverse of map_range. Note that cache invalidation is
  // explicit via invalidate_range and need not be performed here. If using
  // emulated mapping this must call iree_hal_buffer_emulated_unmap_range to
  // release the transient resources.

  // On Hexagon, there is currently no need to do an actual mapping - and thus
  // no actual unmapping. See _map_range().
  (void)buffer;

  return iree_ok_status();
}

static iree_status_t
iree_hal_hexagon_buffer_invalidate_range(iree_hal_buffer_t *base_buffer,
                                         iree_device_size_t local_byte_offset,
                                         iree_device_size_t local_byte_length) {
  iree_hal_hexagon_buffer_t *buffer = iree_hal_hexagon_buffer_cast(base_buffer);

  // TODO(hexagon): invalidate the range if required by the buffer. Writes on
  // the device are expected to be visible to the host after this returns.
  (void)buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "buffer mapping not implemented");

  return status;
}

static iree_status_t
iree_hal_hexagon_buffer_flush_range(iree_hal_buffer_t *base_buffer,
                                    iree_device_size_t local_byte_offset,
                                    iree_device_size_t local_byte_length) {
  iree_hal_hexagon_buffer_t *buffer = iree_hal_hexagon_buffer_cast(base_buffer);

  // TODO(hexagon): flush the range if required by the buffer. Writes on the
  // host are expected to be visible to the device after this returns.
  (void)buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "buffer mapping not implemented");

  return status;
}

static const iree_hal_buffer_vtable_t iree_hal_hexagon_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_hexagon_buffer_destroy,
    .map_range = iree_hal_hexagon_buffer_map_range,
    .unmap_range = iree_hal_hexagon_buffer_unmap_range,
    .invalidate_range = iree_hal_hexagon_buffer_invalidate_range,
    .flush_range = iree_hal_hexagon_buffer_flush_range,
};
