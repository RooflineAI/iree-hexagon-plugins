// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "HAP_mem.h"
#include "hexagon/arm_dsp/bindings.h"
#include "hexagon/arm_dsp/cmd_buf.h"
#include "hexagon/arm_dsp/profiler.h"
#include "hexagon/dsp/align.h"
#include "hexagon/dsp/device.h"
#include "hexagon/dsp/executable.h"
#include "hexagon/dsp/pmu/hexagon_pmu.h"
#include "hexagon/dsp/power_mode.h"
#include "hexagon/dsp/rt/async_runtime.h"
#include "hexagon/dsp/rt/dispatch_state.h"
#include "hexagon/dsp/rt/executable_library.h"
#include "hexagon/dsp/rt/profiler.h"
#include "hexagon/dsp/rt/runtime_state.h"
#include "hexagon/dsp/rt/vtcm_partition.h"
#include "hexagon_dsp.h"
#include "qurt.h"

/// data about a command buffer
typedef struct hexagon_dsp_command_buffer_s {
  /// handle of owning RPC session
  remote_handle64 rpc_handle;
  uint8_t *cmd_buf_data; ///< contains hexagon_rt_arm_dsp_cmd_buf
  size_t cmd_buf_size;
  /// deduplicated direct buffer references (fd != -1) used by any command,
  /// needed for cache maintenance at command buffer level
  hexagon_rt_arm_dsp_binding_t *direct_bufs;
  uint32_t num_direct_bufs;
} hexagon_dsp_command_buffer_t;

/**
 * Macro to read type T from a buffer with serialized data (pointer P, size S).
 * Provides pointer to deserialized value as V.
 * Returns with error from calling function if size is too small.
 */
#define PEEK_SERIALIZED(P, S, T, V)                                            \
  if (S < sizeof(T)) {                                                         \
    return AEE_EINCOMPLETEITEM;                                                \
  }                                                                            \
  const T *V = (const T *)P;

/**
 * Macro to read type T from a buffer with serialized data (pointer P, size S).
 * Provides pointer to deserialized value as V.
 * Advances pointer P, reduces size S.
 * Returns with error from calling function if size is too small.
 */
#define READ_SERIALIZED(P, S, T, V)                                            \
  PEEK_SERIALIZED(P, S, T, V)                                                  \
  P += sizeof(T);                                                              \
  S -= sizeof(T);

/**
 * Macro to read type T from a buffer with serialized data (pointer P, size S).
 * Provides pointer to deserialized value as V (mutable).
 * Advances pointer P, reduces size S.
 * Returns with error from calling function if size is too small.
 */
#define READ_SERIALIZED_MUT(P, S, T, V)                                        \
  if (S < sizeof(T)) {                                                         \
    return AEE_EINCOMPLETEITEM;                                                \
  }                                                                            \
  T *V = (T *)P;                                                               \
  P += sizeof(T);                                                              \
  S -= sizeof(T);

/**
 * @brief Record a direct buffer reference (fd != -1) in bufs, merging it with
 *        overlapping or adjacent ranges of the same fd. On bufs == NULL: only
 *        counts (upper bound).
 * @param[in] ref buffer reference to record, ignored if indirect (fd == -1)
 * @param[in,out] bufs array to append to, NULL to only count
 * @param[in,out] num_bufs number of entries in bufs (or counted so far)
 */
static void hexa_cmd_buf_add_direct_buf(const hexagon_rt_arm_dsp_buf_ref_t *ref,
                                        hexagon_rt_arm_dsp_binding_t *bufs,
                                        uint32_t *num_bufs) {
  if (ref->fd == -1) {
    return; // indirect, covered by binding table
  }
  if (bufs) {
    // merge with all overlapping or adjacent ranges of the same fd
    uint64_t begin = ref->offset;
    uint64_t end = ref->offset + ref->length;
    uint32_t idx = 0;
    while (idx < *num_bufs) {
      uint64_t idx_begin = bufs[idx].offset;
      uint64_t idx_end = bufs[idx].offset + bufs[idx].length;
      if (bufs[idx].fd != ref->fd || idx_begin > end || begin > idx_end) {
        ++idx;
        continue;
      }
      // absorb entry into the new range and remove
      // re-check idx as it now holds a different entry
      begin = idx_begin < begin ? idx_begin : begin;
      end = idx_end > end ? idx_end : end;
      // remove, by moving the last entry in its place
      // no-op if only one entry
      --*num_bufs;
      bufs[idx] = bufs[*num_bufs];
    }
    bufs[*num_bufs].fd = ref->fd;
    bufs[*num_bufs].offset = begin;
    bufs[*num_bufs].length = end - begin;
  }
  ++*num_bufs;
}

/**
 * @brief Walk command buffer and collect all direct buffer references.
 * @param[in] cmd_buf_data pointer to serialized command buffer data
 * @param[in] cmd_buf_size size of serialized command buffer data
 * @param[out] bufs array to store merged direct buffer references in,
 *                  NULL to only count (without merging)
 * @param[out] num_bufs number of entries (to be) stored in bufs
 * @retval AEE_SUCCESS for success
 */
static int hexa_cmd_buf_collect_direct_bufs(const uint8_t *cmd_buf_data,
                                            int cmd_buf_size,
                                            hexagon_rt_arm_dsp_binding_t *bufs,
                                            uint32_t *num_bufs) {
  *num_bufs = 0;
  READ_SERIALIZED(cmd_buf_data, cmd_buf_size, hexagon_rt_arm_dsp_cmd_buf_t,
                  cmd_buf)
  for (uint32_t idx_entry = 0; idx_entry < cmd_buf->num_entries; ++idx_entry) {
    PEEK_SERIALIZED(cmd_buf_data, cmd_buf_size, hexagon_rt_arm_dsp_cmd_base_t,
                    cmd_base)
    switch ((hexagon_rt_arm_dsp_cmd_type_enum_t)cmd_base->cmd_type) {

    case HEXAGON_RT_ARM_DSP_CMD_DISPATCH: {
      READ_SERIALIZED(cmd_buf_data, cmd_buf_size,
                      hexagon_rt_arm_dsp_cmd_dispatch_t, cmd_dispatch)
      for (uint16_t c = 0; c < cmd_dispatch->constant_count; ++c) {
        READ_SERIALIZED(cmd_buf_data, cmd_buf_size, hexagon_rt_arm_dsp_con_t,
                        con)
        (void)con;
      }
      for (uint32_t b = 0; b < cmd_dispatch->num_bindings; ++b) {
        READ_SERIALIZED(cmd_buf_data, cmd_buf_size,
                        hexagon_rt_arm_dsp_buf_ref_t, buf_ref)
        hexa_cmd_buf_add_direct_buf(buf_ref, bufs, num_bufs);
      }
      break;
    }

    case HEXAGON_RT_ARM_DSP_CMD_BARRIER: {
      READ_SERIALIZED(cmd_buf_data, cmd_buf_size,
                      hexagon_rt_arm_dsp_cmd_barrier_t, cmd_barrier)
      (void)cmd_barrier;
      break;
    }

    case HEXAGON_RT_ARM_DSP_CMD_COPY: {
      READ_SERIALIZED(cmd_buf_data, cmd_buf_size, hexagon_rt_arm_dsp_cmd_copy_t,
                      cmd_copy)
      hexa_cmd_buf_add_direct_buf(&cmd_copy->src, bufs, num_bufs);
      hexa_cmd_buf_add_direct_buf(&cmd_copy->trgt, bufs, num_bufs);
      break;
    }

    case HEXAGON_RT_ARM_DSP_CMD_FILL: {
      READ_SERIALIZED(cmd_buf_data, cmd_buf_size, hexagon_rt_arm_dsp_cmd_fill_t,
                      cmd_fill)
      hexa_cmd_buf_add_direct_buf(&cmd_fill->trgt, bufs, num_bufs);
      break;
    }

    default:
      FARF(
          RUNTIME_HIGH,
          "HEXAGON-RUNTIME-ERROR: unknown/invalid command in command buffer\n");
      return AEE_EBADITEM;
    }
  }
  return AEE_SUCCESS;
}

/**
 * @brief Perform cache operation on all given buffers.
 * @param[in] bufs buffers (fd, offset, length), entries with fd -1 are skipped
 * @param[in] num_bufs number of entries in bufs
 * @param[in] op cache operation (e.g. QURT_MEM_CACHE_INVALIDATE)
 * @retval AEE_SUCCESS for success
 */
static int hexa_cmd_buf_cache_op(const hexagon_rt_arm_dsp_binding_t *bufs,
                                 uint32_t num_bufs, qurt_mem_cache_op_t op) {
  for (uint32_t idx = 0; idx < num_bufs; ++idx) {
    if (bufs[idx].fd == -1 || bufs[idx].length == 0) {
      continue;
    }
    uint8_t *dsp_vaddr = NULL;
    uint64_t paddr = 0;
    int err = HAP_mmap_get(bufs[idx].fd, (void **)&dsp_vaddr, &paddr);
    if (err != AEE_SUCCESS) {
      return err;
    }
    err = qurt_mem_cache_clean((qurt_addr_t)(dsp_vaddr + bufs[idx].offset),
                               bufs[idx].length, op, QURT_MEM_DCACHE);
    HAP_mmap_put(bufs[idx].fd);
    if (err != QURT_EOK) {
      // according to doc, the only error is QURT_EVAL - invalid cache type
      return AEE_EFAILED;
    }
  }
  return AEE_SUCCESS;
}

/**
 * @brief Create a command buffer (called when ARM host side finalizes command
 * buffer)
 * @param[in] rpc_handle handle of DSP RPC session
 * @param[in] cmd_buf_data command buffer data of type
 *                         hexagon_rt_arm_dsp_cmd_buf_t
 * @param[in] cmd_buf_size size of command buffer data
 * @param[out] command_buffer_handle handle of the loaded executable
 * @retval AEE_SUCCESS for success
 */
int hexagon_dsp_command_buffer_create(remote_handle64 rpc_handle,
                                      const uint8 *cmd_buf_data,
                                      int cmd_buf_size,
                                      int64 *command_buffer_handle) {
  // allocate internal data structure used to mange command buffer
  hexagon_dsp_command_buffer_t *command_buffer = NULL;
  int err = HAP_malloc(sizeof(hexagon_dsp_command_buffer_t),
                       (void **)&command_buffer);
  if (err != AEE_SUCCESS) {
    return err;
  }
  if (!command_buffer) {
    return AEE_ENOMEMORY;
  }
  command_buffer->rpc_handle = rpc_handle;

  // Store local copy of command buffer data.
  // The buffer cmd_buf_data lives only (on DSP side) during the duration of
  // the PRC call. The data needs to be copied in order to retain it.
  command_buffer->cmd_buf_size = cmd_buf_size;
  command_buffer->cmd_buf_data = NULL;
  err = HAP_malloc(cmd_buf_size, (void **)&command_buffer->cmd_buf_data);
  if (err != AEE_SUCCESS) {
    free(command_buffer);
    return err;
  }
  if (!command_buffer->cmd_buf_data) {
    free(command_buffer);
    return AEE_ENOMEMORY;
  }
  memcpy(command_buffer->cmd_buf_data, cmd_buf_data, cmd_buf_size);

  // Collect direct buffer references once, so cache maintenance at execution
  // time does not need to walk the command buffer. First pass counts (upper
  // bound), second pass fills and deduplicates.
  command_buffer->direct_bufs = NULL;
  command_buffer->num_direct_bufs = 0;
  uint32_t max_direct_bufs = 0;
  // count only
  err = hexa_cmd_buf_collect_direct_bufs(command_buffer->cmd_buf_data,
                                         cmd_buf_size, NULL, &max_direct_bufs);
  if (err == AEE_SUCCESS && max_direct_bufs > 0) {
    err = HAP_malloc(max_direct_bufs * sizeof(hexagon_rt_arm_dsp_binding_t),
                     (void **)&command_buffer->direct_bufs);
    if (err == AEE_SUCCESS && !command_buffer->direct_bufs) {
      err = AEE_ENOMEMORY;
    }
    if (err == AEE_SUCCESS) {
      // collect
      err = hexa_cmd_buf_collect_direct_bufs(
          command_buffer->cmd_buf_data, cmd_buf_size,
          command_buffer->direct_bufs, &command_buffer->num_direct_bufs);
    }
  }
  if (err != AEE_SUCCESS) {
    HAP_free(command_buffer->direct_bufs);
    HAP_free(command_buffer->cmd_buf_data);
    HAP_free(command_buffer);
    return err;
  }

  // return pointer to internal data structure as handle
  *command_buffer_handle = (int64)command_buffer;
  return AEE_SUCCESS;
}

/**
 * @brief Destroy command buffer.
 * @param[in] rpc_handle handle of DSP RPC session
 * @param[in] command_buffer_handle handle of the command buffer
 * @retval AEE_SUCCESS for success, should always succeed
 */
int hexagon_dsp_command_buffer_destroy(remote_handle64 rpc_handle,
                                       int64 command_buffer_handle) {
  hexagon_dsp_command_buffer_t *command_buffer =
      (hexagon_dsp_command_buffer_t *)command_buffer_handle;
  // free command buffer data
  HAP_free(command_buffer->direct_bufs);
  HAP_free(command_buffer->cmd_buf_data);
  // free management data structure
  HAP_free(command_buffer);
  return AEE_SUCCESS;
}

/// resolved buffer reference
typedef struct hexa_cmd_buf_res_buf_s {
  int fd;
  uint8_t *dsp_vaddr;
  uint64_t offset;
  uint64_t length;
} hexa_cmd_buf_res_buf_t;

/**
 * @brief Resolve a buffer reference to an actual RPCmem file descriptor,
 *        offset and length. Map the file descriptor to get a virtual address.
 * @param[in] buf_ref buffer reference to resolve, might use "slot" to point to
 *                    buffer in binding table
 * @param[in] bind_tab binding_table
 * @param[in] bind_tab_num_ent number of entries in binding table
 * @param[out] out_res_buf resolved buffer reference
 * @retval AEE_SUCCESS for success
 *
 * Dispatches and transfers in a command buffer use "buffer references" to
 * refer to the buffers to be used as inputs/outputs. A buffer reference can
 * be either "direct" or "indirect".
 * A direct buffer reference refers to a fixed buffer that is known already
 * at command buffer recording time and is the same for each execution of the
 * command buffer.
 * An indirect buffer reference refers to a buffer that will be supplied via a
 * "binding table" when the command buffer is executed. At command buffer
 * recording time, the indirect buffers references point to a "slot" of the
 * binding table (i.e. refer to an entry of it by index). A (potentially
 * different) binding table is passed to each execution of a command buffer.
 * Each entry refers to an actual buffer - or more precisely - to a part of an
 * actual buffer (using additional offset and length).
 * This function takes a buffer reference (direct or indirect) and the binding
 * table and converts it to a DSP vaddr of a buffer, the overall offset into it
 * and the length of the memory range to use.
 */
static int
hexa_cmd_buf_resolve_and_map(const hexagon_rt_arm_dsp_buf_ref_t *buf_ref,
                             const hexagon_rt_arm_dsp_binding_t *bind_tab,
                             uint32_t bind_tab_num_ent,
                             hexa_cmd_buf_res_buf_t *out_res_buf) {
  // direct buffer reference -> use this buffer
  if (buf_ref->fd != -1) {
    out_res_buf->fd = buf_ref->fd;
    out_res_buf->offset = buf_ref->offset;
    out_res_buf->length = buf_ref->length;
  }
  // indirect buffer reference -> use entry from binding table
  else {
    if (buf_ref->slot > bind_tab_num_ent) {
      return AEE_EBADITEM; // access behind end of binding table
    }
    const hexagon_rt_arm_dsp_binding_t *bind_entry = &bind_tab[buf_ref->slot];
    /* The structure of the binding table entry and the buffer ref in memory
     * is as follows:
     *
     * mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmDDDDDDDDDDDDDDmmmmmm
     *                          ||<- offset ->||<- length ->|
     * |<- bind_entry->offset ->||<- bind_entry->length --------->|
     * |<- new offset computed here --------->|
     * ^
     * |
     * \--- bind_entry->buffer_dsp_vaddr
     *
     * m: memory provided by buffer in binding table entry
     * D: part of memory used by buffer reference of dispatch
     */
    out_res_buf->fd = bind_entry->fd;
    if (buf_ref->offset > bind_entry->length) {
      return AEE_EBADITEM; // offset to behind binding table entry's buffer
    }
    if (buf_ref->offset + buf_ref->length > bind_entry->length) {
      return AEE_EBADITEM; // buffer ref extends to behind bind tabs's buffer
    }
    out_res_buf->offset = bind_entry->offset +
                          buf_ref->offset; // overall offset into bind tab's buf
    out_res_buf->length = buf_ref->length;
  }

  // map the buffer to a DSP virtual address
  uint64_t paddr = 0;
  return HAP_mmap_get(out_res_buf->fd, (void **)&out_res_buf->dsp_vaddr,
                      &paddr);
}

/// Unmap the file descriptors in mapped_fds.
static void hexa_cmd_buf_unmap(const int *mapped_fds, uint32_t mapped_fds_cnt) {
  for (uint32_t idx = 0; idx < mapped_fds_cnt; ++idx) {
    HAP_mmap_put(mapped_fds[idx]);
  }
}

/// Completion notification from workers to the master.
typedef struct hexa_cmd_buf_done_s {
  qurt_mutex_t mutex;
  qurt_cond_t cond;
  /// dispatches completed by workers and not yet finished by the master
  uint32_t num_completed;
} hexa_cmd_buf_done_t;

/**
 * A dispatch, prepared by the master thread (the one walking the command
 * buffer) and executed either by the master or by a worker thread.
 */
typedef struct hexa_cmd_buf_task_s {
  /// dispatch state passed to the dispatch function - must be aligned at 16
  /// bytes, because dispatch function declarations assume that alignment
  HEXAGON_ALIGNAS(16)
  hexagon_dsp_extended_dispatch_state_t extended_dispatch_state;
  /// runtime state the extended dispatch state points to
  hexagon_rt_state_t runtime_state;
  /// VTCM partition the runtime state points to, acquired by the master
  hexagon_rt_vtcm_partition_t vtcm_partition;
  hexagon_executable_dispatch_fn_t dispatch_func;
  const hexagon_executable_environment_t *environment;
  /// dispatch name for profiling and diagnostics, may be NULL
  const char *func_name;
  /// allocation holding the arrays the dispatch state points to (constants and
  /// binding pointers) and mapped_fds, freed when finished
  uint8_t *dispatch_arrays;
  /// file descriptors mapped for the bindings, unmapped when finished
  int *mapped_fds;
  uint32_t num_buf_refs;
  /// completion notification of the master
  hexa_cmd_buf_done_t *done_notify;
  /// set by the worker once the dispatch has been executed, guarded by
  /// done_notify->mutex
  bool done;
  /// submitted to a worker and not finished yet, master only
  bool in_flight;
} hexa_cmd_buf_task_t;

/**
 * Scheduler state of one command buffer execution, only visible to the master,
 * except that done_notify is exposed to the workers for notification
 */
typedef struct hexa_cmd_buf_sched_s {
  /// max. number of dispatches executing at the same time (workers + master)
  uint32_t max_concurrent;
  /// task slots, one per execution slot (max_concurrent)
  hexa_cmd_buf_task_t *tasks;
  /// number of tasks in flight (submitted to workers and not finished yet)
  uint32_t num_in_flight;
  /// index of a task slot that is not in flight, max_concurrent if all task
  /// slots are in flight
  uint32_t next_free;
  /// notified by a worker each time it completes a dispatch
  hexa_cmd_buf_done_t done_notify;
  /// Bookkeeping about VTCM usage:
  /// size of the global VTCM pool
  size_t vtcm_pool_size;
  /// sum of the sizes of all partitions currently acquired
  size_t vtcm_in_use;
} hexa_cmd_buf_sched_t;

/**
 * @brief Prepare a dispatch command buffer entry for execution: resolve and
 *        map buffers and build the dispatch state in a free task slot. Does
 *        not acquire VTCM.
 * @param[out] task free task slot to prepare, to be finished by
 *                  hexa_cmd_buf_finish_dispatch on success
 * @param[in,out] cmd_buf_data pointer to pointer to serialized dispatch command
 *                data, updated by amount of processed data
 * @param[in,out] cmd_buf_size pointer to size of serialized command buffer
 *                data, updated by amount of processed data
 * @param[in] bind_tab binding_table
 * @param[in] bind_tab_num_ent number of entries in binding table
 * @param[in] prof_context profiler context, NULL when tracing is disabled
 * @param[out] out_vtcm_bytes VTCM the dispatch declared,
 *                            HEXAGON_RT_VTCM_PARTITION_ALL if unknown
 * @param[out] out_uses_hmx whether the dispatch (possibly) uses HMX
 * @retval AEE_SUCCESS for success
 */
static int hexa_cmd_buf_prepare_dispatch(
    hexa_cmd_buf_task_t *task, const uint8_t **cmd_buf_data, int *cmd_buf_size,
    const hexagon_rt_arm_dsp_binding_t *bind_tab, uint32_t bind_tab_num_ent,
    hexagon_rt_prof_context_t *prof_context, size_t *out_vtcm_bytes,
    bool *out_uses_hmx) {
  PEEK_SERIALIZED(*cmd_buf_data, *cmd_buf_size,
                  hexagon_rt_arm_dsp_cmd_dispatch_t, cmd_dispatch)
  int64_t executable_handle = cmd_dispatch->executable_handle;
  uint32_t export_ordinal = cmd_dispatch->export_ordinal;
  uint32_t num_buf_refs = cmd_dispatch->num_bindings;

  // Check the whole entry is present, so reading it below cannot fail after
  // the arrays have been allocated.
  size_t entry_size =
      sizeof(hexagon_rt_arm_dsp_cmd_dispatch_t) +
      cmd_dispatch->constant_count * sizeof(hexagon_rt_arm_dsp_con_t) +
      num_buf_refs * sizeof(hexagon_rt_arm_dsp_buf_ref_t);
  if ((size_t)*cmd_buf_size < entry_size) {
    return AEE_EINCOMPLETEITEM;
  }
  *cmd_buf_data += sizeof(hexagon_rt_arm_dsp_cmd_dispatch_t);
  *cmd_buf_size -= sizeof(hexagon_rt_arm_dsp_cmd_dispatch_t);

  // Obtain dispatch function pointer.
  hexagon_executable_dispatch_fn_t dispatch_func = NULL;
  int err = hexagon_dsp_executable_get_dispatch_func(
      executable_handle, export_ordinal, &dispatch_func);
  if (err != AEE_SUCCESS) {
    return err;
  }
  const hexagon_executable_environment_t *environment = NULL;
  err = hexagon_dsp_executable_get_environment(executable_handle, &environment);
  if (err != AEE_SUCCESS) {
    return err;
  }

  // Allocate buffer for nested arrays in dispatch state.
  // The dispatch state contains some pointers, which are arrays semantically.
  // These pointers need to point to actual arrays, which are allocated here.
  // To reduce the number of HAL_malloc calls, the arrays are allocated in the
  // same buffer and the start addresses are computed based on their size.
  // Storage for the RPCmem file descriptors to unmap at the end is also
  // allocated in the same memory block.
  size_t constants_offset = 0;
  size_t constants_size = cmd_dispatch->constant_count * sizeof(uint32_t);
  size_t binding_ptrs_offset =
      HEXAGON_ALIGN_SIZE_FOR_TYPE(constants_offset + constants_size, void *);
  size_t binding_ptrs_size = num_buf_refs * sizeof(void *);
  size_t binding_lengths_offset = HEXAGON_ALIGN_SIZE_FOR_TYPE(
      binding_ptrs_offset + binding_ptrs_size, size_t);
  size_t binding_lengths_size = num_buf_refs * sizeof(size_t);
  size_t mapped_fds_offset = HEXAGON_ALIGN_SIZE_FOR_TYPE(
      binding_lengths_offset + binding_lengths_size, int);
  size_t mapped_fds_size = num_buf_refs * sizeof(int);
  uint8_t *dispatch_arrays = NULL;
  err = HAP_malloc(mapped_fds_offset + mapped_fds_size,
                   (void **)&dispatch_arrays);
  if (err != AEE_SUCCESS) {
    return err;
  }
  if (!dispatch_arrays) {
    return AEE_ENOMEMORY;
  }
  uint32_t *constants = (uint32_t *)(dispatch_arrays + constants_offset);
  void **binding_ptrs = (void **)(dispatch_arrays + binding_ptrs_offset);
  size_t *binding_lengths =
      (size_t *)(dispatch_arrays + binding_lengths_offset);
  int *mapped_fds = (int *)(dispatch_arrays + mapped_fds_offset);

  // Build dispatch state.
  // Hexagon uses a custom extended dispatch state that carries a pointer to the
  // runtime state, which some dispatches access via the dispatch state
  // argument.
  task->vtcm_partition = (hexagon_rt_vtcm_partition_t){};
  task->runtime_state.prof_context = prof_context;
  task->runtime_state.vtcm_partition = &task->vtcm_partition;
  task->extended_dispatch_state.dispatch_state =
      (hexagon_executable_dispatch_state_t){
          .workgroup_size_x = cmd_dispatch->workgroup_size_x,
          .workgroup_size_y = cmd_dispatch->workgroup_size_y,
          .workgroup_size_z = cmd_dispatch->workgroup_size_z,
          .workgroup_count_x = cmd_dispatch->workgroup_count_x,
          .workgroup_count_y = cmd_dispatch->workgroup_count_y,
          .workgroup_count_z = cmd_dispatch->workgroup_count_z,
          .constants = constants,
          .binding_ptrs = binding_ptrs,
      };
  task->extended_dispatch_state.runtime_state = &task->runtime_state;
  task->dispatch_func = dispatch_func;
  task->environment = environment;
  task->func_name = NULL;
  task->dispatch_arrays = dispatch_arrays;
  task->mapped_fds = mapped_fds;
  task->num_buf_refs = 0;

  // Obtain constants from serialized data
  for (uint16_t c = 0; c < cmd_dispatch->constant_count; ++c) {
    READ_SERIALIZED(*cmd_buf_data, *cmd_buf_size, hexagon_rt_arm_dsp_con_t, con)
    constants[c] = con->value;
  }

  // Resolve buffer references and fill buffer pointers and lengths in
  // dispatch state.
  for (uint32_t idx_buf_ref = 0; idx_buf_ref < num_buf_refs; ++idx_buf_ref) {
    READ_SERIALIZED(*cmd_buf_data, *cmd_buf_size, hexagon_rt_arm_dsp_buf_ref_t,
                    buf_ref)
    hexa_cmd_buf_res_buf_t res_buf = {};
    err = hexa_cmd_buf_resolve_and_map(buf_ref, bind_tab, bind_tab_num_ent,
                                       &res_buf);
    if (err != AEE_SUCCESS) {
      hexa_cmd_buf_unmap(mapped_fds, idx_buf_ref);
      HAP_free(dispatch_arrays);
      return err;
    }
    binding_ptrs[idx_buf_ref] = (void *)(res_buf.dsp_vaddr + res_buf.offset);
    binding_lengths[idx_buf_ref] = res_buf.length;
    mapped_fds[idx_buf_ref] = res_buf.fd;
    ++task->num_buf_refs;
  }

  // Note that this function is also called when profiler is disabled, but it
  // is not very expensive. We do not care if it fails either.
  hexagon_dsp_executable_get_dispatch_func_name(
      executable_handle, export_ordinal, &task->func_name);

  // The dispatch allocates VTCM only from its own partition, sized by the VTCM
  // the compiler declared for it. Without a declaration it gets all of VTCM.
  // Without valid resource information, the dispatch might use HMX.
  const hexagon_executable_dispatch_attrs_t *attrs = NULL;
  *out_vtcm_bytes = HEXAGON_RT_VTCM_PARTITION_ALL;
  *out_uses_hmx = true;
  if (hexagon_dsp_executable_get_dispatch_attrs(
          executable_handle, export_ordinal, &attrs) == AEE_SUCCESS &&
      attrs &&
      (attrs->hexagon_flags & HEXAGON_DISPATCH_FLAG_RESOURCES_VALID) != 0) {
    *out_uses_hmx =
        (attrs->hexagon_flags & HEXAGON_DISPATCH_FLAG_USES_HMX) != 0;
    if (attrs->vtcm_bytes != (uint64_t)INT64_MAX) {
      *out_vtcm_bytes = (size_t)attrs->vtcm_bytes;
    }
  }

  return AEE_SUCCESS;
}

/**
 * @brief Execute all workgroups of a prepared dispatch.
 * @param[in] task prepared task with acquired VTCM partition
 *
 * Runs on the master or on a worker thread.
 */
static void hexa_cmd_buf_run_dispatch(hexa_cmd_buf_task_t *task) {
  const hexagon_executable_dispatch_state_t *dispatch_state =
      &task->extended_dispatch_state.dispatch_state;

  hexagon_rt_prof_record_t *prof_dispatch_record = profiler_measurement_start(
      task->runtime_state.prof_context, KERNEL, task->func_name);

  // for now, run all workgroups sequentially
  for (uint16_t wg_id_z = 0; wg_id_z < dispatch_state->workgroup_count_z;
       ++wg_id_z) {
    for (uint32_t wg_id_y = 0; wg_id_y < dispatch_state->workgroup_count_y;
         ++wg_id_y) {
      for (uint32_t wg_id_x = 0; wg_id_x < dispatch_state->workgroup_count_x;
           ++wg_id_x) {

        // build workgroup state - must be aligned at 16 bytes, because dispatch
        // function declarations assumes that alignment
        HEXAGON_ALIGNAS(16)
        hexagon_executable_workgroup_state_t workgroup_state = {
            .workgroup_id_x = wg_id_x,
            .workgroup_id_y = wg_id_y,
            .workgroup_id_z = wg_id_z,
            .processor_id = 0,
        };

        // call dispatch function
        task->dispatch_func(task->environment, dispatch_state,
                            &workgroup_state);
      }
    }
  }

  profiler_measurement_finish_and_record(prof_dispatch_record);
}

/**
 * @brief Release the resources of an executed dispatch and return its task
 *        slot. Master only, as it returns the VTCM partition to the pool.
 * @param[in,out] sched scheduler state
 * @param[in] task_idx index of the executed task, free again afterwards
 */
static void hexa_cmd_buf_finish_dispatch(hexa_cmd_buf_sched_t *sched,
                                         uint32_t task_idx) {
  hexa_cmd_buf_task_t *task = &sched->tasks[task_idx];
  sched->vtcm_in_use -= task->vtcm_partition.size;
  hexagon_rt_release_vtcm_partition(&task->vtcm_partition);
  hexa_cmd_buf_unmap(task->mapped_fds, task->num_buf_refs);
  HAP_free(task->dispatch_arrays);
  task->in_flight = false;
  sched->next_free = task_idx < sched->next_free ? task_idx : sched->next_free;
}

/// Worker thread entry point: execute the dispatch and notify the master.
static void hexa_cmd_buf_task_worker(void *arg) {
  hexa_cmd_buf_task_t *task = (hexa_cmd_buf_task_t *)arg;
  hexa_cmd_buf_run_dispatch(task);
  // notify master of completion and mark this task completed
  hexa_cmd_buf_done_t *done_notify = task->done_notify;
  qurt_mutex_lock(&done_notify->mutex);
  task->done = true;
  ++done_notify->num_completed;
  qurt_cond_signal(&done_notify->cond);
  qurt_mutex_unlock(&done_notify->mutex);
}

/**
 * @brief Wait until a dispatch running on a worker has completed and finish
 *        it. Must only be called with dispatches in flight.
 * @param[in,out] sched scheduler state
 *
 * A worker sets done and counts the completion under the same lock, so each
 * counted completion matches a task that is done and not finished yet.
 */
static void hexa_cmd_buf_sched_wait_one(hexa_cmd_buf_sched_t *sched) {
  assert(sched->num_in_flight > 0);
  hexa_cmd_buf_done_t *done_notify = &sched->done_notify;
  uint32_t task_idx = sched->max_concurrent;
  qurt_mutex_lock(&done_notify->mutex);
  while (done_notify->num_completed == 0) {
    qurt_cond_wait(&done_notify->cond, &done_notify->mutex);
  }
  --done_notify->num_completed;
  // search which task completed
  for (uint32_t idx = 0; idx < sched->max_concurrent; ++idx) {
    if (sched->tasks[idx].in_flight && sched->tasks[idx].done) {
      task_idx = idx;
      break;
    }
  }
  qurt_mutex_unlock(&done_notify->mutex);
  --sched->num_in_flight;
  hexa_cmd_buf_finish_dispatch(sched, task_idx);
}

/// Wait until all dispatches running on workers have completed.
static void hexa_cmd_buf_sched_drain(hexa_cmd_buf_sched_t *sched) {
  while (sched->num_in_flight > 0) {
    hexa_cmd_buf_sched_wait_one(sched);
  }
}

/**
 * @brief Execute dispatch command buffer entry: prepare it, wait for a free
 *        execution slot and enough VTCM, then run it on the master (HMX or
 *        sequential execution) or submit it to a worker.
 * @param[in,out] sched scheduler state
 * @param[in,out] cmd_buf_data pointer to pointer to serialized dispatch command
 *                data, updated by amount of processed data
 * @param[in,out] cmd_buf_size pointer to size of serialized command buffer
 *                data, updated by amount of processed data
 * @param[in] bind_tab binding_table
 * @param[in] bind_tab_num_ent number of entries in binding table
 * @param[in] prof_context profiler context, NULL when tracing is
 * disabled
 * @retval AEE_SUCCESS for success
 */
static int hexa_cmd_buf_exec_dispatch(
    hexa_cmd_buf_sched_t *sched, const uint8_t **cmd_buf_data,
    int *cmd_buf_size, const hexagon_rt_arm_dsp_binding_t *bind_tab,
    uint32_t bind_tab_num_ent, hexagon_rt_prof_context_t *prof_context) {
  // Wait for a free execution slot, which comes with a free task slot.
  // in single threaded execution, this is a no-op,
  // since the master slot is free if it executes this code here
  while (sched->num_in_flight >= sched->max_concurrent) {
    hexa_cmd_buf_sched_wait_one(sched);
  }
  // next_free may change while waiting for VTCM below
  uint32_t task_idx = sched->next_free;
  hexa_cmd_buf_task_t *task = &sched->tasks[task_idx];
  size_t vtcm_bytes = 0;
  bool uses_hmx = true;
  int err = hexa_cmd_buf_prepare_dispatch(task, cmd_buf_data, cmd_buf_size,
                                          bind_tab, bind_tab_num_ent,
                                          prof_context, &vtcm_bytes, &uses_hmx);
  if (err != AEE_SUCCESS) {
    return err;
  }
  const char *name = task->func_name ? task->func_name : "(unnamed)";

  // A dispatch that does not even fit into the empty pool can never run.
  if (vtcm_bytes != HEXAGON_RT_VTCM_PARTITION_ALL &&
      vtcm_bytes > sched->vtcm_pool_size) {
    FARF(ERROR,
         "HEXAGON-DISPATCH-RESOURCES-VIOLATION name=%s declared=%llu "
         "vtcm_pool=%llu",
         name, (unsigned long long)vtcm_bytes,
         (unsigned long long)sched->vtcm_pool_size);
    abort();
  }

  // Wait until the partition can be carved out of the pool.
  while (!hexagon_rt_try_acquire_vtcm_partition(vtcm_bytes, task->func_name,
                                                &task->vtcm_partition)) {
    size_t vtcm_free = sched->vtcm_pool_size - sched->vtcm_in_use;
    if (vtcm_free >= task->vtcm_partition.size) {
      FARF(RUNTIME_HIGH,
           "HEXAGON-RUNTIME-WARNING: VTCM fragmented, no contiguous %llu "
           "bytes for %s although %llu bytes are free",
           (unsigned long long)task->vtcm_partition.size, name,
           (unsigned long long)vtcm_free);
    }
    if (sched->num_in_flight == 0) {
      // Nothing of this command buffer holds VTCM anymore, but it still does
      // not fit: someone else holds VTCM.
      FARF(ERROR,
           "HEXAGON-RUNTIME-ERROR: no contiguous %llu bytes of VTCM for %s "
           "although no dispatch is in flight",
           (unsigned long long)task->vtcm_partition.size, name);
      abort();
    }
    // wait for another task to free up some VTCM
    hexa_cmd_buf_sched_wait_one(sched);
  }
  sched->vtcm_in_use += task->vtcm_partition.size;

  // The master holds the HMX lock, so HMX dispatches run here. Workers that
  // are already running keep going in the meantime.
  if (uses_hmx || sched->max_concurrent == 1) {
    hexa_cmd_buf_run_dispatch(task);
    hexa_cmd_buf_finish_dispatch(sched, task_idx);
    return AEE_SUCCESS;
  }

  // give the dispatch to a worker to execute
  task->done = false;
  task->in_flight = true;
  ++sched->num_in_flight;
  hexagon_rt_async_execute(hexa_cmd_buf_task_worker, task);
  // Advance next_free to the next task that is not in flight. If there is
  // none, all execution slots are taken: it ends at max_concurrent and
  // wait_one sets it.
  while (sched->next_free < sched->max_concurrent &&
         sched->tasks[sched->next_free].in_flight) {
    ++sched->next_free;
  }

  return AEE_SUCCESS;
}

/**
 * @brief Execute barrier command buffer entry: wait until all dispatches in
 *        flight have completed.
 * @param[in,out] sched scheduler state
 * @param[in,out] cmd_buf_data pointer to pointer to serialized barrier command
 *                data, updated by amount of processed data
 * @param[in,out] cmd_buf_size pointer to size of serialized command buffer
 *                data, updated by amount of processed data
 * @retval AEE_SUCCESS for success
 */
static int hexa_cmd_buf_exec_barrier(hexa_cmd_buf_sched_t *sched,
                                     const uint8_t **cmd_buf_data,
                                     int *cmd_buf_size) {
  READ_SERIALIZED(*cmd_buf_data, *cmd_buf_size,
                  hexagon_rt_arm_dsp_cmd_barrier_t, cmd_barrier)
  (void)cmd_barrier;
  hexa_cmd_buf_sched_drain(sched);
  return AEE_SUCCESS;
}

/**
 * @brief Execute copy command buffer entry.
 * @param[in,out] cmd_buf_data pointer to pointer to serialized copy command
 *                data, updated by amount of processed data
 * @param[in,out] cmd_buf_size pointer to size of serialized command buffer
 *                data, updated by amount of processed data
 * @param[in] bind_tab binding_table
 * @param[in] bind_tab_num_ent number of entries in binding table
 * @retval AEE_SUCCESS for success
 */
static int hexa_cmd_buf_exec_copy(const uint8_t **cmd_buf_data,
                                  int *cmd_buf_size,
                                  const hexagon_rt_arm_dsp_binding_t *bind_tab,
                                  uint32_t bind_tab_num_ent) {
  READ_SERIALIZED(*cmd_buf_data, *cmd_buf_size, hexagon_rt_arm_dsp_cmd_copy_t,
                  cmd_copy)
  hexa_cmd_buf_res_buf_t src = {};
  int err = hexa_cmd_buf_resolve_and_map(&cmd_copy->src, bind_tab,
                                         bind_tab_num_ent, &src);
  if (err != AEE_SUCCESS) {
    return err;
  }
  hexa_cmd_buf_res_buf_t dest = {};
  err = hexa_cmd_buf_resolve_and_map(&cmd_copy->trgt, bind_tab,
                                     bind_tab_num_ent, &dest);
  if (err != AEE_SUCCESS) {
    HAP_mmap_put(src.fd);
    return err;
  }

  // copy data (cache maintenance is done for whole command buffer)
  memcpy(dest.dsp_vaddr + dest.offset, src.dsp_vaddr + src.offset,
         dest.length < src.length ? dest.length : src.length);

  HAP_mmap_put(dest.fd);
  HAP_mmap_put(src.fd);
  return AEE_SUCCESS;
}

/**
 * @brief Execute fill command buffer entry.
 * @param[in,out] cmd_buf_data pointer to pointer to serialized fill command
 *                data, updated by amount of processed data
 * @param[in,out] cmd_buf_size pointer to size of serialized command buffer
 *                data, updated by amount of processed data
 * @param[in] bind_tab binding_table
 * @param[in] bind_tab_num_ent number of entries in binding table
 * @retval AEE_SUCCESS for success
 */
static int hexa_cmd_buf_exec_fill(const uint8_t **cmd_buf_data,
                                  int *cmd_buf_size,
                                  const hexagon_rt_arm_dsp_binding_t *bind_tab,
                                  uint32_t bind_tab_num_ent) {
  READ_SERIALIZED(*cmd_buf_data, *cmd_buf_size, hexagon_rt_arm_dsp_cmd_fill_t,
                  cmd_fill)
  hexa_cmd_buf_res_buf_t dest = {};
  int err = hexa_cmd_buf_resolve_and_map(&cmd_fill->trgt, bind_tab,
                                         bind_tab_num_ent, &dest);
  if (err != AEE_SUCCESS) {
    return err;
  }

  // fill buffer with pattern (cache maintenance is done for whole command
  // buffer)
  uint8_t p = 0;
  for (uint64_t i = 0; i < cmd_fill->trgt.length; ++i) {
    dest.dsp_vaddr[dest.offset + i] = cmd_fill->pattern[p];
    ++p;
    if (p >= cmd_fill->pattern_length) {
      p = 0;
    }
  }

  HAP_mmap_put(dest.fd);
  return AEE_SUCCESS;
}

/**
 * @brief Walk the command buffer and execute or schedule its commands.
 * @param[in,out] sched scheduler state
 * @param[in] cmd_buf_data pointer to serialized command buffer data
 * @param[in] cmd_buf_size size of serialized command buffer data
 * @param[in] bind_tab binding_table
 * @param[in] bind_tab_num_ent number of entries in binding table
 * @param[in] prof_context profiler context if tracing enabled, NULL otherwise
 * @retval AEE_SUCCESS for success
 *
 * Dispatches may still be in flight when this function returns.
 */
static int hexa_cmd_buf_walk_cmds(hexa_cmd_buf_sched_t *sched,
                                  const uint8_t *cmd_buf_data, int cmd_buf_size,
                                  const hexagon_rt_arm_dsp_binding_t *bind_tab,
                                  uint32_t bind_tab_num_ent,
                                  hexagon_rt_prof_context_t *prof_context) {
  // Process command buffer from serialized representation, entry by entry.
  READ_SERIALIZED(cmd_buf_data, cmd_buf_size, hexagon_rt_arm_dsp_cmd_buf_t,
                  cmd_buf)
  for (uint32_t idx_entry = 0; idx_entry < cmd_buf->num_entries; ++idx_entry) {
    PEEK_SERIALIZED(cmd_buf_data, cmd_buf_size, hexagon_rt_arm_dsp_cmd_base_t,
                    cmd_base)
    switch ((hexagon_rt_arm_dsp_cmd_type_enum_t)cmd_base->cmd_type) {

    case HEXAGON_RT_ARM_DSP_CMD_DISPATCH: {
      // measures preparation, waiting for resources and submission; the
      // execution itself is measured by the KERNEL zone
      hexagon_rt_prof_record_t *prof_record =
          profiler_measurement_start(prof_context, DISPATCH, NULL);
      int err =
          hexa_cmd_buf_exec_dispatch(sched, &cmd_buf_data, &cmd_buf_size,
                                     bind_tab, bind_tab_num_ent, prof_context);
      profiler_measurement_finish_and_record(prof_record);
      if (err != AEE_SUCCESS) {
        return err;
      }
      break;
    }

    case HEXAGON_RT_ARM_DSP_CMD_BARRIER: {
      hexagon_rt_prof_record_t *prof_record =
          profiler_measurement_start(prof_context, BARRIER, NULL);
      int err = hexa_cmd_buf_exec_barrier(sched, &cmd_buf_data, &cmd_buf_size);
      profiler_measurement_finish_and_record(prof_record);
      if (err != AEE_SUCCESS) {
        return err;
      }
      break;
    }

    // Copies and fills use no VTCM and are independent of the dispatches in
    // flight (commands between barriers are), so the master runs them inline.
    case HEXAGON_RT_ARM_DSP_CMD_COPY: {
      hexagon_rt_prof_record_t *prof_record =
          profiler_measurement_start(prof_context, COPY, NULL);
      int err = hexa_cmd_buf_exec_copy(&cmd_buf_data, &cmd_buf_size, bind_tab,
                                       bind_tab_num_ent);
      profiler_measurement_finish_and_record(prof_record);
      if (err != AEE_SUCCESS) {
        return err;
      }
      break;
    }

    case HEXAGON_RT_ARM_DSP_CMD_FILL: {
      hexagon_rt_prof_record_t *prof_record =
          profiler_measurement_start(prof_context, FILL, NULL);
      int err = hexa_cmd_buf_exec_fill(&cmd_buf_data, &cmd_buf_size, bind_tab,
                                       bind_tab_num_ent);
      profiler_measurement_finish_and_record(prof_record);
      if (err != AEE_SUCCESS) {
        return err;
      }
      break;
    }

    default:
      FARF(
          RUNTIME_HIGH,
          "HEXAGON-RUNTIME-ERROR: unknown/invalid command in command buffer\n");
      return AEE_EBADITEM;
    }
  }

  return AEE_SUCCESS;
}

/**
 * @brief Execute the commands of a command buffer (without cache maintenance).
 * @param[in] cmd_buf_data pointer to serialized command buffer data
 * @param[in] cmd_buf_size size of serialized command buffer data
 * @param[in] bind_tab binding_table
 * @param[in] bind_tab_num_ent number of entries in binding table
 * @param[in] max_concurrent max. number of dispatches executing at the same
 *                           time, 1 for sequential execution on this thread
 * @param[in] prof_context profiler context if tracing enabled, NULL otherwise
 * @retval AEE_SUCCESS for success
 *
 * The calling thread (the master) walks the command buffer. Dispatches that
 * do not use HMX are submitted to the AsyncRuntime worker threads as soon as
 * an execution slot and their VTCM partition are available; HMX dispatches
 * run on the master, which holds the HMX lock.
 * All dispatches have completed when this function returns, also on error.
 */
static int hexa_cmd_buf_exec_cmds(const uint8_t *cmd_buf_data, int cmd_buf_size,
                                  const hexagon_rt_arm_dsp_binding_t *bind_tab,
                                  uint32_t bind_tab_num_ent,
                                  uint32_t max_concurrent,
                                  hexagon_rt_prof_context_t *prof_context) {
  if (max_concurrent < 1) {
    max_concurrent = 1;
  }
  hexa_cmd_buf_sched_t sched = {
      .max_concurrent = max_concurrent,
      .tasks = NULL,
      .num_in_flight = 0,
      .next_free = 0,
      // Also binds the VTCM pool and the HMX lock to this thread before any
      // worker exists.
      .vtcm_pool_size = hexagon_rt_vtcm_pool_size(),
      .vtcm_in_use = 0,
  };

  // Allocate the task slots, one per dispatch that can execute at the same
  // time. HAP_malloc does not guarantee the alignment of the tasks,
  // over-allocate.
  const size_t task_align = HEXAGON_BIG_ALIGNOF(hexa_cmd_buf_task_t);
  uint8_t *tasks_allocation = NULL;
  int err =
      HAP_malloc(max_concurrent * sizeof(hexa_cmd_buf_task_t) + task_align,
                 (void **)&tasks_allocation);
  if (err != AEE_SUCCESS) {
    return err;
  }
  if (!tasks_allocation) {
    return AEE_ENOMEMORY;
  }
  // need to give it the aligned ptr, since the allocated ptr may not be
  // correctly aligned
  sched.tasks = (hexa_cmd_buf_task_t *)HEXAGON_ALIGN_SIZE_FOR_TYPE(
      (uintptr_t)tasks_allocation, hexa_cmd_buf_task_t);
  for (uint32_t idx = 0; idx < max_concurrent; ++idx) {
    memset(&sched.tasks[idx], 0, sizeof(sched.tasks[idx]));
    sched.tasks[idx].done_notify = &sched.done_notify;
  }

  qurt_mutex_init(&sched.done_notify.mutex);
  qurt_cond_init(&sched.done_notify.cond);
  sched.done_notify.num_completed = 0;

  err = hexa_cmd_buf_walk_cmds(&sched, cmd_buf_data, cmd_buf_size, bind_tab,
                               bind_tab_num_ent, prof_context);

  // wait fial all outstanding tasks to complete
  hexa_cmd_buf_sched_drain(&sched);

  qurt_cond_destroy(&sched.done_notify.cond);
  qurt_mutex_destroy(&sched.done_notify.mutex);
  HAP_free(tasks_allocation);
  return err;
}

/**
 * @brief Execute command buffer.
 * @param[in] command_buffer command buffer to execute
 * @param[in] bind_tab_data pointer to serialized binding table data
 * @param[in] bind_tab_size size of serialized binding table data
 * @param[in] prof_context profiler context if tracing enabled, NULL otherwise
 * @retval AEE_SUCCESS for success
 *
 * Cache maintenance is done once for the whole command buffer: all buffers
 * (binding table entries and direct buffer references) are invalidated before
 * the first command and flushed after the last one. The commands in between
 * all run on the DSP and thus share the same view of the caches.
 */
static int
hexa_cmd_buf_exec_buf(const hexagon_dsp_command_buffer_t *command_buffer,
                      const uint8_t *bind_tab_data, int bind_tab_size,
                      hexagon_rt_prof_context_t *prof_context) {
  // Set up binding table for access by index.
  // This is possible because serialized data is just header followed by array.
  READ_SERIALIZED(bind_tab_data, bind_tab_size,
                  hexagon_rt_arm_dsp_binding_tab_t, bind_tab_header);
  uint32_t bind_tab_num_ent = bind_tab_header->num_entries;
  if (bind_tab_size < bind_tab_num_ent * sizeof(hexagon_rt_arm_dsp_binding_t)) {
    return AEE_EINCOMPLETEITEM;
  }
  const hexagon_rt_arm_dsp_binding_t *bind_tab =
      (const hexagon_rt_arm_dsp_binding_t *)bind_tab_data;

  // invalidate cache of all buffers used by the command buffer
  hexagon_rt_prof_record_t *prof_invalidate_record =
      profiler_measurement_start(prof_context, MEMORY_MANAGEMENT, NULL);
  int err = hexa_cmd_buf_cache_op(bind_tab, bind_tab_num_ent,
                                  QURT_MEM_CACHE_INVALIDATE);
  if (err == AEE_SUCCESS) {
    err = hexa_cmd_buf_cache_op(command_buffer->direct_bufs,
                                command_buffer->num_direct_bufs,
                                QURT_MEM_CACHE_INVALIDATE);
  }
  profiler_measurement_finish_and_record(prof_invalidate_record);
  if (err != AEE_SUCCESS) {
    return err;
  }

  err = hexa_cmd_buf_exec_cmds(
      command_buffer->cmd_buf_data, command_buffer->cmd_buf_size, bind_tab,
      bind_tab_num_ent,
      hexagon_dsp_get_max_concurrent_dispatches(command_buffer->rpc_handle),
      prof_context);

  // flush cache of all buffers used by the command buffer, also on error to
  // publish whatever has been written so far
  hexagon_rt_prof_record_t *prof_flush_record =
      profiler_measurement_start(prof_context, MEMORY_MANAGEMENT, NULL);
  int err_flush =
      hexa_cmd_buf_cache_op(bind_tab, bind_tab_num_ent, QURT_MEM_CACHE_FLUSH);
  if (err_flush == AEE_SUCCESS) {
    err_flush = hexa_cmd_buf_cache_op(command_buffer->direct_bufs,
                                      command_buffer->num_direct_bufs,
                                      QURT_MEM_CACHE_FLUSH);
  }
  profiler_measurement_finish_and_record(prof_flush_record);

  return err != AEE_SUCCESS ? err : err_flush;
}

/**
 * @brief Execute a command buffer.
 * @param[in] rpc_handle handle of DSP RPC session
 * @param[in] command_buffer_handle handle of the command buffer
 * @param[in] binding_table_data serialized binding table used to resolve
 * buffer references
 * @param[in] binding_table_size size in bytes of serialized binding_table_data
 * (includes header and entries)
 * @retval AEE_SUCCESS for success
 */
int hexagon_dsp_command_buffer_execute(remote_handle64 rpc_handle,
                                       int64 command_buffer_handle,
                                       const uint8 *binding_table_data,
                                       int binding_table_size) {
  hexagon_dsp_command_buffer_t *command_buffer =
      (hexagon_dsp_command_buffer_t *)command_buffer_handle;
  int err = hexa_cmd_buf_exec_buf(command_buffer, binding_table_data,
                                  binding_table_size, NULL);
  return err;
}

/**
 * @brief Execute a command buffer while logging profiler information. The PMU
 * counters are not isolated when executing multiple command buffers at once and
 * will output garbage!
 * This rpc is duplicated in order to manage how NULL is controlled without
 * relying on the RPC mechanism (crashes when passing NULL).
 * Additionally, profiler buffer sizing is manually incremented during command
 * buffer creation, while the actual number of measurement zones is defined on
 * the DSP side. Any change to the profiler must be reflected on both sides!
 * @param[in] rpc_handle handle of DSP RPC session
 * @param[in] command_buffer_handle handle of the command buffer
 * @param[in] binding_table_data serialized binding table used to resolve
 * buffer references
 * @param[in] binding_table_size size in bytes of serialized binding_table_data
 * (includes header and entries)
 * @param[out] performance_log_data serialized performance log
 * @param[in] performance_log_size size in bytes of performance_log_data
 * (includes header and records)
 * @retval AEE_SUCCESS for success
 */
int hexagon_dsp_command_buffer_execute_profiler(remote_handle64 rpc_handle,
                                                int64 command_buffer_handle,
                                                const uint8 *binding_table_data,
                                                int binding_table_size,
                                                uint8 *performance_log_data,
                                                int performance_log_size) {
  hexagon_dsp_command_buffer_t *command_buffer =
      (hexagon_dsp_command_buffer_t *)command_buffer_handle;

  if (!performance_log_data) {
    FARF(
        RUNTIME_HIGH,
        "HEXAGON-RUNTIME-ERROR: Missing performance log, stopping execution\n");
    return AEE_EBADPARM;
  }

  READ_SERIALIZED_MUT(performance_log_data, performance_log_size,
                      hexagon_rt_prof_header_t, profiler_header)

  if (performance_log_size <
      profiler_header->num_records * sizeof(hexagon_rt_prof_record_t)) {
    return AEE_EINCOMPLETEITEM;
  }
  hexagon_rt_prof_record_t *profiler_records =
      (hexagon_rt_prof_record_t *)performance_log_data;

  // Setting the PMU unit, note that this assumes only one command buffer is
  // being executed at once!
  qurt_pmu_enable(0); // Reset PMU registers
  hexagon_pmu_configure(&profiler_header->pmu_event_ids);
  qurt_pmu_enable(1);

  hexagon_rt_prof_context_t prof_context;
  profiler_context_init(profiler_header, profiler_records, &prof_context);

  hexagon_rt_prof_record_t *prof_record =
      profiler_measurement_start(&prof_context, DSP_EXECUTION, NULL);

  int err = hexa_cmd_buf_exec_buf(command_buffer, binding_table_data,
                                  binding_table_size, &prof_context);

  profiler_measurement_finish_and_record(prof_record);

  profiler_context_deinit(&prof_context);

  return err;
}
