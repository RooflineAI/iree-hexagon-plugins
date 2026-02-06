// Copyright 2025 RooflineAI GmbH

#include "hexagon/command_buffer.h"

#include "AEEStdErr.h"
#include "hexagon/buffer.h"
#include "hexagon/executable.h"
#include "hexagon/serialize/cmd_barrier_serialize.h"
#include "hexagon/serialize/cmd_copy_serialize.h"
#include "hexagon/serialize/cmd_dispatch_serialize.h"
#include "hexagon/serialize/cmd_fill_serialize.h"
#include "hexagon/serialize/command_buffer_serialize.h"
#include "hexagon/serialize/command_buffer_types.h"
#include "hexagon/serialize/rpc_types.h"
#include "hexagon/utils.h"
#include "hexagon_dsp.h"
#include "iree/base/allocator.h"
#include "iree/base/api.h"
#include "iree/base/config.h"
#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/resource_set.h"
#include "rpcmem.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_command_buffer_t
//
// Command buffer entries (like dispatches and execution barriers) get added
// one by one. This happens between "begin" and "end" calls. When "end" has
// returned, the command buffer needs to be ready on the device for execution.
// There are two approaches to implement this:
// (1) Keep the device side in sync all the time during recording. This is done
//     by CUDA and Vulkan.
// (2) Record the contents on the host side. Update the device side on "end".
//     This is done by Metal and here for Hexagon.
// The approach (2) is chosen, because moving the data to the DSP is rather
// complicated/expensive:
//  - ARM host and DSP data layouts don't match. Pointers on one side don't have
//    a meaning on the other side. This means data has to be serialized.
//    See arm_dsp subdir for data structures used in serialization.
//    See command_buffer_serialize.c for implementation of the serialization.
//  - The RPC call to get a chunk of data from the ARM host to the DSP is rather
//    expensive and causes a latency. So the number of RPC calls shall be
//    kept low.
//===----------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t
    iree_hal_hexagon_command_buffer_vtable;

static iree_hal_hexagon_command_buffer_t *
iree_hal_hexagon_command_buffer_cast(iree_hal_command_buffer_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hexagon_command_buffer_vtable);
  return (iree_hal_hexagon_command_buffer_t *)base_value;
}

static void iree_hal_hexagon_command_buffer_clear_recorded(
    iree_hal_hexagon_command_buffer_t *command_buffer) {
  iree_hal_hexagon_command_buffer_entry_t *entry = command_buffer->first_entry;
  while (entry != NULL) {
    iree_hal_hexagon_command_buffer_entry_t *entry_to_free = entry;
    entry = entry->next;
    iree_allocator_free(command_buffer->host_allocator, entry_to_free);
  }
  command_buffer->first_entry = NULL;
  command_buffer->last_entry = NULL;
}

static void iree_hal_hexagon_command_buffer_clear(
    iree_hal_hexagon_command_buffer_t *command_buffer) {
  // destroy command buffer on DSP side
  if (command_buffer->rpc_command_buffer_handle) {
    hexagon_dsp_command_buffer_destroy(
        command_buffer->rpc_session_handle,
        command_buffer->rpc_command_buffer_handle);
    command_buffer->rpc_command_buffer_handle = 0;
  }
  // clear list of recorded entries
  iree_hal_hexagon_command_buffer_clear_recorded(command_buffer);
}

iree_status_t iree_hal_hexagon_command_buffer_create(
    iree_hal_allocator_t *device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator, rpc_session_handle_t rpc_session_handle,
    iree_arena_block_pool_t *block_pool,
    iree_hal_command_buffer_t **out_command_buffer) {
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_command_buffer = NULL;

  iree_hal_hexagon_command_buffer_t *command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*command_buffer) +
                                iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                            (void **)&command_buffer));
  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t *)command_buffer + sizeof(*command_buffer),
      &iree_hal_hexagon_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->rpc_session_handle = rpc_session_handle;
  // Starts at 1, because index 0 is used for the topmost record for
  // synchronization
  command_buffer->profiling_entries = 1;

  // TODO(hexagon): allocate any additional resources for managing command
  // buffer state. Some implementations may have their own command
  // buffer/command list APIs this can route to or may need to implement it all
  // themselves using iree_arena_t/block pools. Implementations should also
  // retain any resources used during the recording and can use
  // iree_hal_resource_set_t* to make that easier.
  iree_status_t status = iree_ok_status();

  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
    status = iree_hal_resource_set_allocate(block_pool,
                                            &command_buffer->resource_set);
  }

  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_release(&command_buffer->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_hexagon_command_buffer_destroy(
    iree_hal_command_buffer_t *base_command_buffer) {
  iree_hal_hexagon_command_buffer_t *command_buffer =
      iree_hal_hexagon_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(hexagon): release any implementation resources and
  // iree_hal_resource_set_t.
  iree_hal_hexagon_command_buffer_clear(command_buffer);

  if (command_buffer->resource_set) {
    iree_hal_resource_set_free(command_buffer->resource_set);
    command_buffer->resource_set = NULL;
  }

  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_hexagon_command_buffer_isa(
    iree_hal_command_buffer_t *base_command_buffer) {
  return iree_hal_resource_is(&base_command_buffer->resource,
                              &iree_hal_hexagon_command_buffer_vtable);
}

iree_status_t iree_hal_hexagon_command_buffer_get_rpc_handle(
    iree_hal_command_buffer_t *base_command_buffer,
    rpc_command_buffer_handle_t *out_rpc_handle) {
  if (!iree_hal_hexagon_command_buffer_isa(base_command_buffer)) {
    return iree_make_status(
        IREE_STATUS_INCOMPATIBLE,
        "non-Hexagon command buffers do not have RPC handle");
  }
  iree_hal_hexagon_command_buffer_t *command_buffer =
      iree_hal_hexagon_command_buffer_cast(base_command_buffer);
  *out_rpc_handle = command_buffer->rpc_command_buffer_handle;
  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_command_buffer_begin(
    iree_hal_command_buffer_t *base_command_buffer) {
  iree_hal_hexagon_command_buffer_t *command_buffer =
      iree_hal_hexagon_command_buffer_cast(base_command_buffer);

  // TODO(hexagon): if the implementation needs to route the begin to the
  // implementation it can be done here. Note that creation may happen much
  // earlier than recording and any expensive work should be deferred until this
  // point to make profiling easier.

  // Nothing needs to be done here for Hexagon.
  (void)command_buffer;

  iree_status_t status = iree_ok_status();

  return status;
}

static iree_status_t iree_hal_hexagon_command_buffer_end(
    iree_hal_command_buffer_t *base_command_buffer) {
  iree_hal_hexagon_command_buffer_t *command_buffer =
      iree_hal_hexagon_command_buffer_cast(base_command_buffer);

  // TODO(hexagon): if recording requires multiple passes any fixup/linking can
  // happen here. Recording-only resources are no longer needed after this point
  // and can be disposed.

  // serialize the command buffer entries to a buffer with the ARM/DSP layout
  // data structure and create a DSP side command buffer

  // get size of serialized data, also check and count entries
  iree_host_size_t cmd_buf_size = 0;
  uint32_t num_entries = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hexagon_command_buffer_serialize_prep(
      command_buffer, &cmd_buf_size, &num_entries));

  // allocate RPC memory buffer for ARM/DSP data structures
  if (cmd_buf_size > INT_MAX /* max size supported by rpcmem_alloc() */) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "serialized command buffer too big");
  }
  uint8_t *cmd_buf_data = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                       RPCMEM_DEFAULT_FLAGS, (int)cmd_buf_size);
  if (!cmd_buf_data) {
    return iree_make_status(IREE_STATUS_UNKNOWN, "rpcmem_alloc returned NULL");
  }

  // serialize to ARM/DSP data
  iree_status_t status_serialize_exec =
      iree_hal_hexagon_command_buffer_serialize_exec(
          command_buffer, iree_hal_hexagon_buffer_get_dsp_vaddr, num_entries,
          cmd_buf_data, cmd_buf_size);
  if (!iree_status_is_ok(status_serialize_exec)) {
    rpcmem_free(cmd_buf_data);
    return status_serialize_exec;
  }

  // Create DSP side command buffer.
  // The RPC memory cmd_buf_data gets made visible to the DSP during the
  // duration of the PRC call. The data needs to be copied on DSP side in order
  // to retain it.
  int dsp_err = hexagon_dsp_command_buffer_create(
      command_buffer->rpc_session_handle, cmd_buf_data, cmd_buf_size,
      &command_buffer->rpc_command_buffer_handle);
  rpcmem_free(cmd_buf_data);
  if (dsp_err != AEE_SUCCESS) {
    return IREE_HAL_HEXAGON_MAKE_STATUS_FROM_DSP_ERR(
        dsp_err, "could not create command buffer on DSP");
  }

  // clear recorded entries (no more needed from here)
  iree_hal_hexagon_command_buffer_clear_recorded(command_buffer);

  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t *base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t *location) {
  iree_hal_hexagon_command_buffer_t *command_buffer =
      iree_hal_hexagon_command_buffer_cast(base_command_buffer);

  // TODO(hexagon): begin a nested debug group (push) if the implementation
  // has a way to insert markers. This is informational and can be ignored.
  (void)command_buffer;

  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_command_buffer_end_debug_group(
    iree_hal_command_buffer_t *base_command_buffer) {
  iree_hal_hexagon_command_buffer_t *command_buffer =
      iree_hal_hexagon_command_buffer_cast(base_command_buffer);

  // TODO(hexagon): end a nested debug group (pop). Always called 1:1 in stack
  // order with begin_debug_group.
  (void)command_buffer;

  return iree_ok_status();
}

/// append command to command buffer
static void iree_hal_hexagon_command_buffer_append(
    iree_hal_hexagon_command_buffer_t *command_buffer,
    iree_hal_hexagon_command_buffer_entry_t *entry) {
  if (command_buffer->last_entry) {
    command_buffer->last_entry->next = entry;
  } else {
    command_buffer->first_entry = entry;
  }
  command_buffer->last_entry = entry;
}

static iree_status_t iree_hal_hexagon_command_buffer_execution_barrier(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t *memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t *buffer_barriers) {
  iree_hal_hexagon_command_buffer_t *command_buffer =
      iree_hal_hexagon_command_buffer_cast(base_command_buffer);

  // TODO(hexagon): barriers split the execution sequence into all operations
  // that did happen before the barrier and all that will happen after. In
  // implementations that have no concurrency this can be a no-op. This is
  // effectively just a signal_event followed by a wait_event.

  if (buffer_barrier_count != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Hexagon execution barriers do not support "
                            "barriers on specific buffers");
  }

  // get size of serialized barrier command
  iree_host_size_t cmd_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hexagon_cmd_barrier_serialize_prep(&cmd_size));

  // allocate buffer for command buffer entry and serialized command
  iree_hal_hexagon_command_buffer_entry_t *cmd_barrier_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      command_buffer->host_allocator, sizeof(*cmd_barrier_entry) + cmd_size,
      (void **)&cmd_barrier_entry));
  // initialize new entry
  cmd_barrier_entry->next = NULL;
  cmd_barrier_entry->size = cmd_size;
  uint8_t *cmd_data = (uint8_t *)cmd_barrier_entry + sizeof(*cmd_barrier_entry);

  // serialize barrier command
  iree_status_t status_serialize =
      iree_hal_hexagon_cmd_barrier_serialize_exec(cmd_data, cmd_size);
  if (!iree_status_is_ok(status_serialize)) {
    iree_allocator_free(command_buffer->host_allocator, cmd_barrier_entry);
    return status_serialize;
  }

  // append barrier command to command buffer
  iree_hal_hexagon_command_buffer_append(command_buffer, cmd_barrier_entry);
  command_buffer->profiling_entries++;

  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_command_buffer_signal_event(
    iree_hal_command_buffer_t *base_command_buffer, iree_hal_event_t *event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_hexagon_command_buffer_t *command_buffer =
      iree_hal_hexagon_command_buffer_cast(base_command_buffer);

  // TODO(hexagon): WIP API and may change; signals the given event allowing
  // waiters to proceed.
  (void)command_buffer;
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "events not implemented");

  return status;
}

static iree_status_t iree_hal_hexagon_command_buffer_reset_event(
    iree_hal_command_buffer_t *base_command_buffer, iree_hal_event_t *event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_hexagon_command_buffer_t *command_buffer =
      iree_hal_hexagon_command_buffer_cast(base_command_buffer);

  // TODO(hexagon): WIP API and may change; resets the given event to
  // unsignaled.
  (void)command_buffer;
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "events not implemented");

  return status;
}

static iree_status_t iree_hal_hexagon_command_buffer_wait_events(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t **events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t *memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t *buffer_barriers) {
  iree_hal_hexagon_command_buffer_t *command_buffer =
      iree_hal_hexagon_command_buffer_cast(base_command_buffer);

  // TODO(hexagon): WIP API and may change; waits on the list of events and
  // enacts the specified set of barriers. Implementations without
  // fine-grained tracking can treat this as an execution_barrier and ignore
  // the memory/buffer barriers provided.
  (void)command_buffer;
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "events not implemented");

  return status;
}

static iree_status_t iree_hal_hexagon_command_buffer_advise_buffer(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  iree_hal_hexagon_command_buffer_t *command_buffer =
      iree_hal_hexagon_command_buffer_cast(base_command_buffer);

  // TODO(hexagon): WIP API and may change; this is likely to become an
  // madvise-like command that can be used to control prefetching and other
  // cache behavior. The current discard behavior is a hint that the buffer
  // contents will never be used again and that if they are in a cache they
  // need not be written back to global memory.
  (void)command_buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "discard buffer not implemented");

  return status;
}

/// Fill a buffer with a repeating byte pattern.
/// A pattern of length 1 (i.e. a single byte) means filling the
/// whole part of the buffer (starting at offset, length given by byte_length)
/// with this byte (pattern[0]).
/// A longer pattern (e.g., the 4 bytes (in hex) 11 22 33 44) means writing
/// the given sequence of bytes repeatedly to the part of the buffer. For
/// example, for offset 8 and length 12, this would result in the following
/// data in the buffer if it was zero-filled before:
///   00 00 00 00 00 00 00 00  11 22 33 44 11 22 33 44
///   11 22 33 44 11 22 33 44  11 22 33 44 11 22 33 44
///   00 00 00 00 00 00 00 00  ...
static iree_status_t iree_hal_hexagon_command_buffer_fill_buffer(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void *pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_hexagon_command_buffer_t *command_buffer =
      iree_hal_hexagon_command_buffer_cast(base_command_buffer);

  // TODO(hexagon): memset on the buffer. The pattern_length is 1, 2, or 4
  // bytes. Note that the buffer may be a reference to a binding table slot in
  // which case it will be provided during submission to a queue.

  // check that pattern is non-empty and fits the static buffer
  //  - comments in IREE HAL state that there is an upper limit of 4
  if (pattern_length == 0 || pattern_length > 4) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "unsupported pattern length %" PRIu64,
                            pattern_length);
  }

  // get size of serialized fill command
  iree_host_size_t cmd_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hexagon_cmd_fill_serialize_prep(&cmd_size));

  // allocate buffer for command buffer entry and serialized command
  iree_hal_hexagon_command_buffer_entry_t *cmd_fill_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(command_buffer->host_allocator,
                                             sizeof(*cmd_fill_entry) + cmd_size,
                                             (void **)&cmd_fill_entry));
  // initialize new entry
  cmd_fill_entry->next = NULL;
  cmd_fill_entry->size = cmd_size;
  uint8_t *cmd_data = (uint8_t *)cmd_fill_entry + sizeof(*cmd_fill_entry);

  // serialize fill command
  iree_status_t status_serialize = iree_hal_hexagon_cmd_fill_serialize_exec(
      pattern_length, pattern, &target_ref,
      iree_hal_hexagon_buffer_get_dsp_vaddr, cmd_data, cmd_size);
  if (!iree_status_is_ok(status_serialize)) {
    iree_allocator_free(command_buffer->host_allocator, cmd_fill_entry);
    return status_serialize;
  }

  // retain target buffer
  iree_status_t status_retain = iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &target_ref.buffer);
  if (!iree_status_is_ok(status_retain)) {
    iree_allocator_free(command_buffer->host_allocator, cmd_fill_entry);
    return status_retain;
  }

  // append fill command to command buffer
  iree_hal_hexagon_command_buffer_append(command_buffer, cmd_fill_entry);
  command_buffer->profiling_entries++;

  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_command_buffer_update_buffer(
    iree_hal_command_buffer_t *base_command_buffer, const void *source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_hexagon_command_buffer_t *command_buffer =
      iree_hal_hexagon_command_buffer_cast(base_command_buffer);

  // TODO(hexagon): embed and copy a small (~64KB) chunk of host memory to the
  // target buffer. The source_buffer contents must be captured as they may
  // change/be freed after this call completes.
  // Note that the target buffer may be a reference to a binding table slot in
  // which case it will be provided during submission to a queue.
  (void)command_buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "update buffer not implemented");

  return status;
}

static iree_status_t iree_hal_hexagon_command_buffer_copy_buffer(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_hexagon_command_buffer_t *command_buffer =
      iree_hal_hexagon_command_buffer_cast(base_command_buffer);

  // TODO(hexagon): memcpy between two buffers. The buffers must both be
  // device-visible but may reside on either the host or device.
  // Note that either buffer may be a reference to a binding table slot in
  // which case it will be provided during submission to a queue.

  // get size of serialized copy command
  iree_host_size_t cmd_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hexagon_cmd_copy_serialize_prep(&cmd_size));

  // allocate buffer for command buffer entry and serialized command
  iree_hal_hexagon_command_buffer_entry_t *cmd_copy_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(command_buffer->host_allocator,
                                             sizeof(*cmd_copy_entry) + cmd_size,
                                             (void **)&cmd_copy_entry));
  // initialize new entry
  cmd_copy_entry->next = NULL;
  cmd_copy_entry->size = cmd_size;
  uint8_t *cmd_data = (uint8_t *)cmd_copy_entry + sizeof(*cmd_copy_entry);

  // serialize copy command
  iree_status_t status_serialize = iree_hal_hexagon_cmd_copy_serialize_exec(
      &source_ref, &target_ref, iree_hal_hexagon_buffer_get_dsp_vaddr, cmd_data,
      cmd_size);
  if (!iree_status_is_ok(status_serialize)) {
    iree_allocator_free(command_buffer->host_allocator, cmd_copy_entry);
    return status_serialize;
  }

  // retain source and target buffers
  iree_hal_buffer_t *retain_buffers[] = {source_ref.buffer, target_ref.buffer};
  iree_status_t status_retain = iree_hal_resource_set_insert(
      command_buffer->resource_set, IREE_ARRAYSIZE(retain_buffers),
      retain_buffers);
  if (!iree_status_is_ok(status_retain)) {
    iree_allocator_free(command_buffer->host_allocator, cmd_copy_entry);
    return status_retain;
  }

  // append copy command to command buffer
  iree_hal_hexagon_command_buffer_append(command_buffer, cmd_copy_entry);
  command_buffer->profiling_entries++;

  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_command_buffer_collective(
    iree_hal_command_buffer_t *base_command_buffer, iree_hal_channel_t *channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  iree_hal_hexagon_command_buffer_t *command_buffer =
      iree_hal_hexagon_command_buffer_cast(base_command_buffer);

  // TODO(hexagon): perform the collective operation defined by op. See the
  // headers for more information. The channel is fixed for a particular
  // recording but note that either buffer may be a reference to a binding
  // table slot in which case it will be provided during submission to a
  // queue.
  (void)command_buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "collectives not implemented");

  return status;
}

static iree_status_t iree_hal_hexagon_command_buffer_dispatch(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_executable_t *executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_hexagon_command_buffer_t *command_buffer =
      iree_hal_hexagon_command_buffer_cast(base_command_buffer);

  // TODO(hexagon): dispatch the specified executable entry point with the
  // given workgroup count directly or indirectly based on flags. The
  // constants must be copied into the command buffer as they may be mutated
  // or freed after this call returns. Note that any of the bindings may be
  // references to binding table slots in which case they will be provided
  // during submission to a queue.
  //
  // If an INDIRECT_PARAMETERS flag is set then the workgroup count is stored
  // in the given workgroup count buffer as a uint32_t[3]. The workgroup count
  // may change up until the barrier immediately prior to the dispatch.
  //
  // Arguments are are in the HAL ABI form with constants and bindings unless
  // CUSTOM_ARGUMENTS or INDIRECT_ARGUMENTS are specified, in which case they
  // are passed through directly either from the inlined constants provided to
  // this call or a device visible buffer.

  if (iree_hal_dispatch_uses_custom_arguments(flags)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "custom arguments (direct/indirect) are not supported on Hexagon");
  }
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect parameters are not supported on Hexagon");
  }

  rpc_executable_handle_t rpc_executable_handle =
      iree_hal_hexagon_executable_get_rpc_executable(executable);
  if (!rpc_executable_handle) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "non-Hexagon executables are not supported on Hexagon");
  }

  // get size of serialized dispatch command
  iree_host_size_t cmd_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hexagon_cmd_dispatch_serialize_prep(
      &constants, &bindings, &cmd_size));

  // allocate buffer for command buffer entry and serialized command
  iree_hal_hexagon_command_buffer_entry_t *cmd_dispatch_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      command_buffer->host_allocator, sizeof(*cmd_dispatch_entry) + cmd_size,
      (void **)&cmd_dispatch_entry));
  // initialize new entry
  cmd_dispatch_entry->next = NULL;
  cmd_dispatch_entry->size = cmd_size;
  uint8_t *cmd_data =
      (uint8_t *)cmd_dispatch_entry + sizeof(*cmd_dispatch_entry);

  // serialize dispatch command
  iree_status_t status_serialize = iree_hal_hexagon_cmd_dispatch_serialize_exec(
      rpc_executable_handle, export_ordinal, &config, &constants, &bindings,
      iree_hal_hexagon_buffer_get_dsp_vaddr, cmd_data, cmd_size);
  if (!iree_status_is_ok(status_serialize)) {
    iree_allocator_free(command_buffer->host_allocator, cmd_dispatch_entry);
    return status_serialize;
  }

  // retain direct buffers
  iree_status_t status_retain = iree_hal_resource_set_insert_strided(
      command_buffer->resource_set, bindings.count, bindings.values,
      offsetof(iree_hal_buffer_ref_t, buffer), sizeof(iree_hal_buffer_ref_t));
  if (!iree_status_is_ok(status_retain)) {
    iree_allocator_free(command_buffer->host_allocator, cmd_dispatch_entry);
    return status_retain;
  }

  // append dispatch command to command buffer
  iree_hal_hexagon_command_buffer_append(command_buffer, cmd_dispatch_entry);
  // One for the dispatch, one for the kernel, two for cache management
  command_buffer->profiling_entries += 4;

  return iree_ok_status();
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_hexagon_command_buffer_vtable = {
        .destroy = iree_hal_hexagon_command_buffer_destroy,
        .begin = iree_hal_hexagon_command_buffer_begin,
        .end = iree_hal_hexagon_command_buffer_end,
        .begin_debug_group = iree_hal_hexagon_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_hexagon_command_buffer_end_debug_group,
        .execution_barrier = iree_hal_hexagon_command_buffer_execution_barrier,
        .signal_event = iree_hal_hexagon_command_buffer_signal_event,
        .reset_event = iree_hal_hexagon_command_buffer_reset_event,
        .wait_events = iree_hal_hexagon_command_buffer_wait_events,
        .advise_buffer = iree_hal_hexagon_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_hexagon_command_buffer_fill_buffer,
        .update_buffer = iree_hal_hexagon_command_buffer_update_buffer,
        .copy_buffer = iree_hal_hexagon_command_buffer_copy_buffer,
        .collective = iree_hal_hexagon_command_buffer_collective,
        .dispatch = iree_hal_hexagon_command_buffer_dispatch,
};
