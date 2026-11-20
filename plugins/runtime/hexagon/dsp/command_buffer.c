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
#include "hexagon/arm_dsp/buf_flags.h"
#include "hexagon/arm_dsp/cmd_buf.h"
#include "hexagon/arm_dsp/profiler.h"
#include "hexagon/dsp/align.h"
#include "hexagon/dsp/device.h"
#include "hexagon/dsp/executable.h"
#include "hexagon/dsp/next_use.h"
#include "hexagon/dsp/pmu/hexagon_pmu.h"
#include "hexagon/dsp/power_mode.h"
#include "hexagon/dsp/residency.h"
#include "hexagon/dsp/rt/async_runtime.h"
#include "hexagon/dsp/rt/executable_library.h"
#include "hexagon/dsp/rt/profiler.h"
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
  /// for invalidating what the host may have written before each execution
  hexagon_rt_arm_dsp_binding_t *direct_bufs;
  uint32_t num_direct_bufs;
  /// buffer references (into cmd_buf_data) in the order the commands acquire
  /// them, the positions of the command order for choosing what to evict
  const hexagon_rt_arm_dsp_buf_ref_t **use_refs;
  uint32_t num_use_refs;
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
    uint32_t flags = ref->flags;
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
      flags &= bufs[idx].flags;
      // remove, by moving the last entry in its place
      // no-op if only one entry
      --*num_bufs;
      bufs[idx] = bufs[*num_bufs];
    }
    bufs[*num_bufs].fd = ref->fd;
    bufs[*num_bufs].offset = begin;
    bufs[*num_bufs].length = end - begin;
    bufs[*num_bufs].flags = flags;
  }
  ++*num_bufs;
}

/**
 * @brief Record a buffer reference: as use at the next position of the command
 *        order and, if direct, in the direct buffer references.
 * @param[in] ref buffer reference to record
 * @param[in,out] bufs see hexa_cmd_buf_add_direct_buf
 * @param[in,out] num_bufs see hexa_cmd_buf_add_direct_buf
 * @param[in,out] use_refs array to append ref to, NULL to only count
 * @param[in,out] num_use_refs number of entries in use_refs (or counted)
 */
static void
hexa_cmd_buf_add_buf_ref(const hexagon_rt_arm_dsp_buf_ref_t *ref,
                         hexagon_rt_arm_dsp_binding_t *bufs, uint32_t *num_bufs,
                         const hexagon_rt_arm_dsp_buf_ref_t **use_refs,
                         uint32_t *num_use_refs) {
  if (use_refs) {
    use_refs[*num_use_refs] = ref;
  }
  ++*num_use_refs;
  hexa_cmd_buf_add_direct_buf(ref, bufs, num_bufs);
}

/**
 * @brief Walk command buffer and collect all buffer references in the order
 *        they are acquired when executing, and the deduplicated direct ones.
 * @param[in] cmd_buf_data pointer to serialized command buffer data
 * @param[in] cmd_buf_size size of serialized command buffer data
 * @param[out] bufs array to store merged direct buffer references in,
 *                  NULL to only count (without merging)
 * @param[out] num_bufs number of entries (to be) stored in bufs
 * @param[out] use_refs array to store pointers to all buffer references in
 *                      (into cmd_buf_data), NULL to only count
 * @param[out] num_use_refs number of entries (to be) stored in use_refs
 * @retval AEE_SUCCESS for success
 */
static int hexa_cmd_buf_collect_buf_refs(
    const uint8_t *cmd_buf_data, int cmd_buf_size,
    hexagon_rt_arm_dsp_binding_t *bufs, uint32_t *num_bufs,
    const hexagon_rt_arm_dsp_buf_ref_t **use_refs, uint32_t *num_use_refs) {
  *num_bufs = 0;
  *num_use_refs = 0;
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
        hexa_cmd_buf_add_buf_ref(buf_ref, bufs, num_bufs, use_refs,
                                 num_use_refs);
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
      hexa_cmd_buf_add_buf_ref(&cmd_copy->src, bufs, num_bufs, use_refs,
                               num_use_refs);
      hexa_cmd_buf_add_buf_ref(&cmd_copy->trgt, bufs, num_bufs, use_refs,
                               num_use_refs);
      break;
    }

    case HEXAGON_RT_ARM_DSP_CMD_FILL: {
      READ_SERIALIZED(cmd_buf_data, cmd_buf_size, hexagon_rt_arm_dsp_cmd_fill_t,
                      cmd_fill)
      hexa_cmd_buf_add_buf_ref(&cmd_fill->trgt, bufs, num_bufs, use_refs,
                               num_use_refs);
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
 * @brief Report buffers that do not fit into the residency budget even with
 *        nothing else in use.
 * @return AEE_ENOMEMORY - AEE_EWOULDBLOCK must not reach the host, which
 *         would retry the execution forever
 */
static int hexa_cmd_buf_residency_exceeded(const char *what) {
  FARF(ERROR,
       "HEXAGON-RUNTIME-ERROR: the buffers of a %s do not fit into the DSP "
       "residency budget",
       what);
  return AEE_ENOMEMORY;
}

/**
 * @brief Invalidate the given buffers the host may have written since the last
 *        execution in the data cache, where mapped.
 * @param[in] residency residency cache of the RPC session
 * @param[in] bufs buffers (fd, offset, length, flags), entries with fd -1 and
 *                 the ones the host never writes are skipped
 * @param[in] num_bufs number of entries in bufs
 * @retval AEE_SUCCESS for success
 */
static int hexa_cmd_buf_host_sync(hexagon_dsp_residency_t *residency,
                                  const hexagon_rt_arm_dsp_binding_t *bufs,
                                  uint32_t num_bufs) {
  for (uint32_t idx = 0; idx < num_bufs; ++idx) {
    if (bufs[idx].fd == -1 ||
        (bufs[idx].flags & HEXAGON_RT_ARM_DSP_BUF_FLAG_HOST_NO_WRITE) != 0) {
      // skip if No write is set
      continue;
    }
    int err = hexagon_dsp_residency_host_sync(
        residency, bufs[idx].fd, bufs[idx].offset, bufs[idx].length);
    if (err != AEE_SUCCESS) {
      return err;
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
  command_buffer->use_refs = NULL;
  command_buffer->num_use_refs = 0;
  uint32_t max_direct_bufs = 0;
  uint32_t num_use_refs = 0;
  // count only
  err = hexa_cmd_buf_collect_buf_refs(command_buffer->cmd_buf_data,
                                      cmd_buf_size, NULL, &max_direct_bufs,
                                      NULL, &num_use_refs);
  if (err == AEE_SUCCESS && max_direct_bufs > 0) {
    err = HAP_malloc(max_direct_bufs * sizeof(hexagon_rt_arm_dsp_binding_t),
                     (void **)&command_buffer->direct_bufs);
    if (err == AEE_SUCCESS && !command_buffer->direct_bufs) {
      err = AEE_ENOMEMORY;
    }
  }
  if (err == AEE_SUCCESS && num_use_refs > 0) {
    err = HAP_malloc(num_use_refs * sizeof(*command_buffer->use_refs),
                     (void **)&command_buffer->use_refs);
    if (err == AEE_SUCCESS && !command_buffer->use_refs) {
      err = AEE_ENOMEMORY;
    }
    if (err == AEE_SUCCESS) {
      // collect
      err = hexa_cmd_buf_collect_buf_refs(
          command_buffer->cmd_buf_data, cmd_buf_size,
          command_buffer->direct_bufs, &command_buffer->num_direct_bufs,
          command_buffer->use_refs, &command_buffer->num_use_refs);
    }
  }
  if (err != AEE_SUCCESS) {
    HAP_free(command_buffer->use_refs);
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
  HAP_free(command_buffer->use_refs);
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
 * Where the master is in the command order of an execution, so the residency
 * cache can evict the buffer used furthest in the future (Belady).
 *
 * The order of buffer uses (use_refs) is fixed at command buffer creation,
 * but binding table slots resolve to file descriptors only per execution.
 * Under the residency budget nothing is evicted, so the next uses are only
 * computed, once per execution, when the first eviction asks for them.
 */
typedef struct hexa_cmd_buf_future_s {
  hexagon_dsp_residency_future_t base;
  const hexagon_dsp_command_buffer_t *command_buffer;
  const hexagon_rt_arm_dsp_binding_t *bind_tab;
  uint32_t bind_tab_num_ent;
  /// position of the next buffer use to acquire, counted up by each acquire
  /// and reset by the caller when it retries a command
  uint32_t position;
  /// uses with their file descriptors, sorted for
  /// hexagon_dsp_next_use_distance, NULL until the first eviction of the
  /// execution asks for them
  hexagon_dsp_use_t *sorted_uses;
  uint32_t num_sorted_uses;
  /// computing sorted_uses failed, evict least recently used first
  bool no_future;
} hexa_cmd_buf_future_t;

/// Resolve the file descriptor of each buffer use and sort them.
static void hexa_cmd_buf_future_compute(hexa_cmd_buf_future_t *future) {
  const hexagon_dsp_command_buffer_t *command_buffer = future->command_buffer;
  if (HAP_malloc(command_buffer->num_use_refs * sizeof(hexagon_dsp_use_t),
                 (void **)&future->sorted_uses) != AEE_SUCCESS ||
      !future->sorted_uses) {
    future->sorted_uses = NULL;
    future->no_future = true;
    return;
  }
  uint32_t num_uses = 0;
  for (uint32_t pos = 0; pos < command_buffer->num_use_refs; ++pos) {
    const hexagon_rt_arm_dsp_buf_ref_t *ref = command_buffer->use_refs[pos];
    int32_t fd = ref->fd;
    if (fd == -1) {
      if (ref->slot >= future->bind_tab_num_ent) {
        continue; // invalid, fails when executed
      }
      fd = future->bind_tab[ref->slot].fd;
    }
    future->sorted_uses[num_uses++] = (hexagon_dsp_use_t){fd, pos};
  }
  hexagon_dsp_uses_sort(future->sorted_uses, num_uses);
  future->num_sorted_uses = num_uses;
}

/// hexagon_dsp_residency_future_t::next_use for an execution
static uint64_t hexa_cmd_buf_next_use(void *user_data, int fd) {
  hexa_cmd_buf_future_t *future = (hexa_cmd_buf_future_t *)user_data;
  if (!future->sorted_uses && !future->no_future) {
    hexa_cmd_buf_future_compute(future);
  }
  if (future->no_future ||
      future->position >= future->command_buffer->num_use_refs) {
    return 0; // all equal, least recently used first
  }
  return hexagon_dsp_next_use_distance(
      future->sorted_uses, future->num_sorted_uses,
      future->command_buffer->num_use_refs, fd, future->position);
}

/**
 * @brief Resolve a buffer reference to an actual RPCmem file descriptor,
 *        offset and length. Map the file descriptor to get a virtual address.
 * @param[in,out] future position in the command order, for eviction, counted
 *                up by one
 * @param[in] buf_ref buffer reference to resolve, might use "slot" to point to
 *                    buffer in binding table
 * @param[in] bind_tab binding_table
 * @param[in] bind_tab_num_ent number of entries in binding table
 * @param[in] access how the command accesses the buffer, for cache maintenance
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
static int hexa_cmd_buf_resolve_and_map(
    hexagon_dsp_residency_t *residency, hexa_cmd_buf_future_t *future,
    const hexagon_rt_arm_dsp_buf_ref_t *buf_ref,
    const hexagon_rt_arm_dsp_binding_t *bind_tab, uint32_t bind_tab_num_ent,
    hexagon_dsp_cache_access_t access, hexa_cmd_buf_res_buf_t *out_res_buf) {
  // the commands acquire their buffers in the order
  // hexa_cmd_buf_collect_buf_refs recorded them, otherwise next-use distances
  // are off
  assert(future->position < future->command_buffer->num_use_refs &&
         future->command_buffer->use_refs[future->position] == buf_ref);
  // direct buffer reference -> use this buffer
  uint64_t alloc_size = 0;
  if (buf_ref->fd != -1) {
    out_res_buf->fd = buf_ref->fd;
    out_res_buf->offset = buf_ref->offset;
    out_res_buf->length = buf_ref->length;
    alloc_size = buf_ref->alloc_size;
  }
  // indirect buffer reference -> use entry from binding table
  else {
    if (buf_ref->slot >= bind_tab_num_ent) {
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
    // indirect refs cannot be normalized on the host, so a whole-buffer length
    // (IREE_HAL_WHOLE_BUFFER) means the rest of the binding table entry
    uint64_t length = buf_ref->length == UINT64_MAX
                          ? bind_entry->length - buf_ref->offset
                          : buf_ref->length;
    if (length > bind_entry->length - buf_ref->offset) {
      return AEE_EBADITEM; // buffer ref extends to behind bind tabs's buffer
    }
    out_res_buf->offset = bind_entry->offset +
                          buf_ref->offset; // overall offset into bind tab's buf
    out_res_buf->length = length;
    alloc_size = bind_entry->alloc_size;
  }

  // map the buffer to a DSP virtual address
  hexagon_dsp_residency_access_t residency_access = {
      .kind = access,
      .offset = out_res_buf->offset,
      .length = out_res_buf->length,
  };
  int err = hexagon_dsp_residency_acquire(
      residency, out_res_buf->fd, alloc_size, &future->base, &residency_access,
      &out_res_buf->dsp_vaddr);
  ++future->position;
  return err;
}

/// Unmap the file descriptors in mapped_fds.
static void hexa_cmd_buf_unmap(hexagon_dsp_residency_t *residency,
                               const int *mapped_fds, uint32_t mapped_fds_cnt) {
  for (uint32_t idx = 0; idx < mapped_fds_cnt; ++idx) {
    hexagon_dsp_residency_release(residency, mapped_fds[idx]);
  }
}

/// Completion notification from workers to the master.
typedef struct hexa_cmd_buf_done_s {
  qurt_mutex_t mutex;
  qurt_cond_t cond;
  /// dispatches completed by workers and not yet finished by the master
  uint32_t num_completed;
} hexa_cmd_buf_done_t;

/// Execution state of a dispatch task.
typedef enum hexa_cmd_buf_task_state_e {
  /// prepared and not executed yet, or still executing
  HEXA_CMD_BUF_TASK_RUNNING,
  /// all workgroups executed successfully
  HEXA_CMD_BUF_TASK_SUCCEEDED,
  /// a workgroup failed, the remaining workgroups were skipped
  HEXA_CMD_BUF_TASK_FAILED,
} hexa_cmd_buf_task_state_t;

/**
 * A dispatch, prepared by the master thread (the one walking the command
 * buffer) and executed either by the master or by a worker thread.
 */
typedef struct hexa_cmd_buf_task_s {
  /// dispatch state passed to the dispatch function - must be aligned at 16
  /// bytes, because dispatch function declarations assume that alignment
  HEXAGON_ALIGNAS(16)
  hexagon_executable_dispatch_state_t dispatch_state;
  /// VTCM partition the workgroup states point to, acquired by the master
  hexagon_rt_vtcm_partition_t vtcm_partition;
  hexagon_executable_dispatch_fn_t dispatch_func;
  const hexagon_executable_environment_t *environment;
  /// dispatch name for profiling and diagnostics, may be NULL
  const char *func_name;
  /// attributes the compiler recorded for the dispatch, may be NULL
  /// used by the main thread to check for necessary cache maintenance after
  /// completion
  const hexagon_executable_dispatch_attrs_t *attrs;
  /// allocation holding the arrays the dispatch state points to (constants,
  /// binding pointers and lengths) and mapped_fds: the slot's dispatches share
  /// it, grown to the largest, and it is freed with the slots.
  uint8_t *dispatch_arrays;
  size_t dispatch_arrays_size;
  /// file descriptors mapped for the bindings, unmapped when finished
  int *mapped_fds;
  uint32_t num_buf_refs;
  /// completion notification of the master
  hexa_cmd_buf_done_t *done_notify;
  /// set by the worker once the dispatch has been executed, guarded by
  /// done_notify->mutex
  hexa_cmd_buf_task_state_t state;
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
  /// notified by a worker each time it completes a dispatch
  hexa_cmd_buf_done_t done_notify;
  /// Bookkeeping about VTCM usage:
  /// size of the global VTCM pool
  size_t vtcm_pool_size;
  /// sum of the sizes of all partitions currently acquired
  size_t vtcm_in_use;
  /// first error of a finished dispatch, AEE_SUCCESS if none failed
  int status;
  /// residency cache of the RPC session, maps the buffers of the commands
  hexagon_dsp_residency_t *residency;
  /// position in the command order, for the residency cache
  hexa_cmd_buf_future_t future;
} hexa_cmd_buf_sched_t;

/// Alignment of the ranges a dispatch writes by bypassing the cache, so no
/// cache line they share with other data can be written back over the DMA's
/// result: the L2 line size, 128 bytes (QuRT has no API to query it). Bindings
/// get it with --iree-stream-resource-min-offset-alignment=128; buffers the
/// application passes in must be aligned by the application.
#define HEXA_CMD_BUF_BYPASS_WRITE_ALIGNMENT 128u

/**
 * @brief How a dispatch accesses binding idx_binding, from the binding masks
 * the compiler recorded. Without them, it may write through the cache.
 */
static hexagon_dsp_cache_access_t
hexa_cmd_buf_binding_access(const hexagon_executable_dispatch_attrs_t *attrs,
                            uint32_t idx_binding) {
  if (!attrs || idx_binding >= 64) {
    // unknown may do anything
    return HEXAGON_DSP_CACHE_ACCESS_CACHED_WRITE;
  }
  uint64_t bit = 1ull << idx_binding;
  bool read_only = (attrs->read_only_bindings & bit) != 0;
  if ((attrs->dma_bypass_bindings & bit) != 0) {
    return read_only ? HEXAGON_DSP_CACHE_ACCESS_BYPASS_READ
                     : HEXAGON_DSP_CACHE_ACCESS_BYPASS_WRITE;
  }
  return read_only ? HEXAGON_DSP_CACHE_ACCESS_CACHED_READ
                   : HEXAGON_DSP_CACHE_ACCESS_CACHED_WRITE;
}

/**
 * @brief Prepare a dispatch command buffer entry for execution: resolve and
 *        map buffers and build the dispatch state in a free task slot. Does
 *        not acquire VTCM.
 * @param[in] residency residency cache of the RPC session
 * @param[in,out] future position in the command order, for eviction
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
 * @param[out] out_threads maximum number of tasks the dispatch submits to the
 *                         thread pool, 1 if none, INT64_MAX if unknown
 * @retval AEE_SUCCESS for success
 * @retval AEE_EBADPARM if the dispatch writes a binding by bypassing the cache
 *                      that is not aligned to
 *                      HEXA_CMD_BUF_BYPASS_WRITE_ALIGNMENT
 */
static int hexa_cmd_buf_prepare_dispatch(
    hexagon_dsp_residency_t *residency, hexa_cmd_buf_future_t *future,
    hexa_cmd_buf_task_t *task, const uint8_t **cmd_buf_data, int *cmd_buf_size,
    const hexagon_rt_arm_dsp_binding_t *bind_tab, uint32_t bind_tab_num_ent,
    hexagon_rt_prof_context_t *prof_context, size_t *out_vtcm_bytes,
    bool *out_uses_hmx, uint64_t *out_threads) {
  PEEK_SERIALIZED(*cmd_buf_data, *cmd_buf_size,
                  hexagon_rt_arm_dsp_cmd_dispatch_t, cmd_dispatch)
  const hexagon_dsp_function_t *function =
      (const hexagon_dsp_function_t *)cmd_dispatch->function_handle;
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
  if (!function) {
    return AEE_EBADPARM;
  }
  hexagon_executable_dispatch_fn_t dispatch_func =
      hexagon_dsp_function_get_dispatch_func(function);
  const hexagon_executable_environment_t *environment =
      hexagon_dsp_function_get_environment(function);

  // Place the nested arrays of the dispatch state in the slot's buffer.
  // The dispatch state contains some pointers, which are arrays semantically.
  // These pointers need to point to actual arrays, which live in the task
  // slot's buffer, grown when this dispatch needs more than any before it.
  // The arrays share the buffer and the start addresses are computed based on
  // their size. Storage for the RPCmem file descriptors to unmap at the end is
  // also in the same buffer.
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
  size_t arrays_size = mapped_fds_offset + mapped_fds_size;
  int err = AEE_SUCCESS;
  if (arrays_size > task->dispatch_arrays_size) {
    // need to grow
    HAP_free(task->dispatch_arrays);
    task->dispatch_arrays = NULL;
    task->dispatch_arrays_size = 0;
    err = HAP_malloc(arrays_size, (void **)&task->dispatch_arrays);
    if (err != AEE_SUCCESS) {
      return err;
    }
    if (!task->dispatch_arrays) {
      return AEE_ENOMEMORY;
    }
    task->dispatch_arrays_size = arrays_size;
  }
  uint8_t *dispatch_arrays = task->dispatch_arrays;
  uint32_t *constants = (uint32_t *)(dispatch_arrays + constants_offset);
  void **binding_ptrs = (void **)(dispatch_arrays + binding_ptrs_offset);
  size_t *binding_lengths =
      (size_t *)(dispatch_arrays + binding_lengths_offset);
  int *mapped_fds = (int *)(dispatch_arrays + mapped_fds_offset);

  // Build dispatch state.
  // It also carries the runtime state (profiler context), which some
  // dispatches access via the dispatch state argument. The VTCM partition is
  // passed in the workgroup state.
  task->vtcm_partition = (hexagon_rt_vtcm_partition_t){};
  task->dispatch_state = (hexagon_executable_dispatch_state_t){
      .workgroup_size_x = cmd_dispatch->workgroup_size_x,
      .workgroup_size_y = cmd_dispatch->workgroup_size_y,
      .workgroup_size_z = cmd_dispatch->workgroup_size_z,
      .workgroup_count_x = cmd_dispatch->workgroup_count_x,
      .workgroup_count_y = cmd_dispatch->workgroup_count_y,
      .workgroup_count_z = cmd_dispatch->workgroup_count_z,
      .constants = constants,
      .binding_ptrs = binding_ptrs,
      .binding_lengths = binding_lengths,
      .prof_context = prof_context,
  };
  task->dispatch_func = dispatch_func;
  task->environment = environment;
  task->func_name = NULL;
  task->attrs = NULL;
  task->mapped_fds = mapped_fds;
  task->num_buf_refs = 0;

  // Obtain constants from serialized data
  for (uint16_t c = 0; c < cmd_dispatch->constant_count; ++c) {
    READ_SERIALIZED(*cmd_buf_data, *cmd_buf_size, hexagon_rt_arm_dsp_con_t, con)
    constants[c] = con->value;
  }

  // Resolve buffer references and fill buffer pointers and lengths in
  // dispatch state. This maps the buffers, which may unmap others first
  // and does the cache maintenance of each binding's access if necessary.
  const hexagon_executable_dispatch_attrs_t *attrs =
      hexagon_dsp_function_get_attrs(function);
  hexagon_rt_prof_record_t *prof_map_record =
      profiler_measurement_start(prof_context, MEMORY_MANAGEMENT, NULL);
  for (uint32_t idx_buf_ref = 0; idx_buf_ref < num_buf_refs; ++idx_buf_ref) {
    READ_SERIALIZED(*cmd_buf_data, *cmd_buf_size, hexagon_rt_arm_dsp_buf_ref_t,
                    buf_ref)
    hexa_cmd_buf_res_buf_t res_buf = {};
    err = hexa_cmd_buf_resolve_and_map(
        residency, future, buf_ref, bind_tab, bind_tab_num_ent,
        hexa_cmd_buf_binding_access(attrs, idx_buf_ref), &res_buf);
    if (err != AEE_SUCCESS) {
      profiler_measurement_finish_and_record(prof_map_record);
      hexa_cmd_buf_unmap(residency, mapped_fds, idx_buf_ref);
      return err;
    }
    binding_ptrs[idx_buf_ref] = (void *)(res_buf.dsp_vaddr + res_buf.offset);
    binding_lengths[idx_buf_ref] = res_buf.length;
    mapped_fds[idx_buf_ref] = res_buf.fd;
    // writing to un-aligned buf is an error
    if (hexa_cmd_buf_binding_access(attrs, idx_buf_ref) ==
            HEXAGON_DSP_CACHE_ACCESS_BYPASS_WRITE &&
        ((uintptr_t)binding_ptrs[idx_buf_ref] %
         HEXA_CMD_BUF_BYPASS_WRITE_ALIGNMENT) != 0) {
      const char *name = hexagon_dsp_function_get_name(function);
      FARF(ERROR,
           "HEXAGON-RUNTIME-ERROR: %s writes binding %u by bypassing the "
           "cache, but its offset %llu is not aligned to %u bytes",
           name ? name : "(unnamed)", (unsigned)idx_buf_ref,
           (unsigned long long)res_buf.offset,
           HEXA_CMD_BUF_BYPASS_WRITE_ALIGNMENT);
      profiler_measurement_finish_and_record(prof_map_record);
      hexa_cmd_buf_unmap(residency, mapped_fds, idx_buf_ref + 1);
      return AEE_EBADPARM;
    }
    ++task->num_buf_refs;
  }
  profiler_measurement_finish_and_record(prof_map_record);

  task->func_name = hexagon_dsp_function_get_name(function);
  task->attrs = attrs;

  // The dispatch allocates VTCM only from its own partition, sized by the VTCM
  // the compiler declared for it. Without a declaration it gets all of VTCM.
  // Without valid resource information, the dispatch might use HMX and
  // any number of threads.
  *out_vtcm_bytes = HEXAGON_RT_VTCM_PARTITION_ALL;
  *out_uses_hmx = true;
  *out_threads = (uint64_t)INT64_MAX;
  if (attrs &&
      (attrs->hexagon_flags & HEXAGON_DISPATCH_FLAG_RESOURCES_VALID) != 0) {
    *out_uses_hmx =
        (attrs->hexagon_flags & HEXAGON_DISPATCH_FLAG_USES_HMX) != 0;
    *out_threads = attrs->threads;
    if (attrs->vtcm_bytes != (uint64_t)INT64_MAX) {
      *out_vtcm_bytes = (size_t)attrs->vtcm_bytes;
    }
  }

  return AEE_SUCCESS;
}

/**
 * @brief Execute all workgroups of a prepared dispatch. Stops at the first
 *        workgroup that fails.
 * @param[in,out] task prepared task with acquired VTCM partition, which is
 *                     emptied before each workgroup
 * @return HEXA_CMD_BUF_TASK_SUCCEEDED or HEXA_CMD_BUF_TASK_FAILED
 *
 * Runs on the master or on a worker thread.
 */
static hexa_cmd_buf_task_state_t
hexa_cmd_buf_run_dispatch(hexa_cmd_buf_task_t *task) {
  const hexagon_executable_dispatch_state_t *dispatch_state =
      &task->dispatch_state;

  hexagon_rt_prof_record_t *prof_dispatch_record = profiler_measurement_start(
      task->dispatch_state.prof_context, KERNEL, task->func_name);

  // for now, run all workgroups sequentially
  for (uint16_t wg_id_z = 0; wg_id_z < dispatch_state->workgroup_count_z;
       ++wg_id_z) {
    for (uint32_t wg_id_y = 0; wg_id_y < dispatch_state->workgroup_count_y;
         ++wg_id_y) {
      for (uint32_t wg_id_x = 0; wg_id_x < dispatch_state->workgroup_count_x;
           ++wg_id_x) {

        // The dispatch declares the VTCM of one workgroup.
        // Each workgroup starts with an empty partition.
        hexagon_rt_vtcm_partition_reset(&task->vtcm_partition);

        // build workgroup state - must be aligned at 16 bytes, because dispatch
        // function declarations assumes that alignment
        HEXAGON_ALIGNAS(16)
        hexagon_executable_workgroup_state_t workgroup_state = {
            .workgroup_id_x = wg_id_x,
            .workgroup_id_y = wg_id_y,
            .workgroup_id_z = wg_id_z,
            .processor_id = 0,
            .vtcm_partition = &task->vtcm_partition,
        };

        // call dispatch function, non-zero return means failure
        int ret = task->dispatch_func(task->environment, dispatch_state,
                                      &workgroup_state);
        if (ret != 0) {
          FARF(ERROR,
               "HEXAGON-RUNTIME-ERROR: dispatch %s failed with %d in "
               "workgroup (%u, %u, %u)",
               task->func_name ? task->func_name : "(unnamed)", ret,
               (unsigned)wg_id_x, (unsigned)wg_id_y, (unsigned)wg_id_z);
          profiler_measurement_finish_and_record(prof_dispatch_record);
          return HEXA_CMD_BUF_TASK_FAILED;
        }
      }
    }
  }

  profiler_measurement_finish_and_record(prof_dispatch_record);
  return HEXA_CMD_BUF_TASK_SUCCEEDED;
}

/**
 * @brief Release the resources of an executed dispatch and return its task
 *        slot. Master only, as it returns the VTCM partition to the pool.
 *        Records the first failing dispatch in sched->status.
 * @param[in,out] sched scheduler state
 * @param[in] task_idx index of the executed task, free again afterwards
 */
static void hexa_cmd_buf_finish_dispatch(hexa_cmd_buf_sched_t *sched,
                                         uint32_t task_idx) {
  hexa_cmd_buf_task_t *task = &sched->tasks[task_idx];
  if (task->state == HEXA_CMD_BUF_TASK_FAILED) {
    sched->status = AEE_EFAILED;
  }
  sched->vtcm_in_use -= task->vtcm_partition.size;
  hexagon_rt_release_vtcm_partition(&task->vtcm_partition);
  // Safety net against prefetching: a line of a range the dispatch wrote by
  // bypassing the cache, fetched while the DMA was running, is stale, so drop
  // it.
  for (uint32_t idx = 0; idx < task->num_buf_refs; ++idx) {
    if (hexa_cmd_buf_binding_access(task->attrs, idx) ==
        HEXAGON_DSP_CACHE_ACCESS_BYPASS_WRITE) {
      int err = hexagon_dsp_residency_after_bypass_write(
          sched->residency, task->mapped_fds[idx],
          (const uint8_t *)task->dispatch_state.binding_ptrs[idx],
          task->dispatch_state.binding_lengths[idx]);
      if (err != AEE_SUCCESS && sched->status == AEE_SUCCESS) {
        sched->status = err;
      }
    }
  }
  hexa_cmd_buf_unmap(sched->residency, task->mapped_fds, task->num_buf_refs);
  task->in_flight = false;
}

/// Worker thread entry point: execute the dispatch and notify the master.
static void hexa_cmd_buf_task_worker(void *arg) {
  hexa_cmd_buf_task_t *task = (hexa_cmd_buf_task_t *)arg;
  hexa_cmd_buf_task_state_t state = hexa_cmd_buf_run_dispatch(task);
  // notify master of completion and mark this task completed
  hexa_cmd_buf_done_t *done_notify = task->done_notify;
  qurt_mutex_lock(&done_notify->mutex);
  task->state = state;
  ++done_notify->num_completed;
  qurt_cond_signal(&done_notify->cond);
  qurt_mutex_unlock(&done_notify->mutex);
}

/**
 * @brief Wait until a dispatch running on a worker has completed and finish
 *        it. Must only be called with dispatches in flight.
 * @param[in,out] sched scheduler state
 *
 * A worker sets the task state and counts the completion under the same lock,
 * so each counted completion matches a task that is done and not finished
 * yet.
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
    if (sched->tasks[idx].in_flight &&
        sched->tasks[idx].state != HEXA_CMD_BUF_TASK_RUNNING) {
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
  // measures preparation and waiting for resources, until the dispatch runs on
  // the master or is submitted to a worker; the execution itself is measured
  // by the KERNEL zone
  hexagon_rt_prof_record_t *prof_record =
      profiler_measurement_start(prof_context, SCHEDULE, NULL);

  // Wait for a free execution slot, which comes with a free task slot.
  // in single threaded execution, this is a no-op,
  // since the master slot is free if it executes this code here
  while (sched->num_in_flight >= sched->max_concurrent) {
    hexa_cmd_buf_sched_wait_one(sched);
  }
  // Pick the first task slot that is not in flight. There is one, since fewer
  // than max_concurrent tasks are in flight. Pick it only once: waiting for
  // VTCM below may free other slots, but this one is already being prepared.
  uint32_t task_idx = 0;
  while (sched->tasks[task_idx].in_flight) {
    ++task_idx;
  }
  assert(task_idx < sched->max_concurrent);
  hexa_cmd_buf_task_t *task = &sched->tasks[task_idx];
  size_t vtcm_bytes = 0;
  bool uses_hmx = true;
  uint64_t threads = (uint64_t)INT64_MAX;
  const uint8_t *entry_data = *cmd_buf_data;
  int entry_size = *cmd_buf_size;
  // current value to reset to, in case acquire buffers fails
  // this way we dont need to keep track on the amount of buffers that are
  // already mapped for the dispatches and can retry mapping all buffers
  uint32_t entry_position = sched->future.position;
  int err = hexa_cmd_buf_prepare_dispatch(
      sched->residency, &sched->future, task, cmd_buf_data, cmd_buf_size,
      bind_tab, bind_tab_num_ent, prof_context, &vtcm_bytes, &uses_hmx,
      &threads);
  // The buffers do not fit into the residency budget next to the ones of the
  // dispatches in flight: wait for one and prepare again.
  while (err == AEE_EWOULDBLOCK && sched->num_in_flight > 0) {
    sched->future.position = entry_position; // reset on acquire failure
    hexa_cmd_buf_sched_wait_one(sched);
    *cmd_buf_data = entry_data;
    *cmd_buf_size = entry_size;

    err = hexa_cmd_buf_prepare_dispatch(sched->residency, &sched->future, task,
                                        cmd_buf_data, cmd_buf_size, bind_tab,
                                        bind_tab_num_ent, prof_context,
                                        &vtcm_bytes, &uses_hmx, &threads);
  }
  if (err == AEE_EWOULDBLOCK) {
    err = hexa_cmd_buf_residency_exceeded("dispatch");
  }
  if (err != AEE_SUCCESS) {
    profiler_measurement_finish_and_record(prof_record);
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
  // are already running keep going in the meantime, except for a dispatch with
  // thread tasks, see above.
  if (uses_hmx || sched->max_concurrent == 1) {
    profiler_measurement_finish_and_record(prof_record);
    task->state = hexa_cmd_buf_run_dispatch(task);
    hexa_cmd_buf_finish_dispatch(sched, task_idx);
    // finish_dispatch has moved the error code to the sched state
    return sched->status;
  }

  // give the dispatch to a worker to execute
  task->state = HEXA_CMD_BUF_TASK_RUNNING;
  task->in_flight = true;
  ++sched->num_in_flight;
  hexagon_rt_async_execute(hexa_cmd_buf_task_worker, task);
  profiler_measurement_finish_and_record(prof_record);
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
 * @param[in,out] sched scheduler state
 * @param[in,out] cmd_buf_data pointer to pointer to serialized copy command
 *                data, updated by amount of processed data
 * @param[in,out] cmd_buf_size pointer to size of serialized command buffer
 *                data, updated by amount of processed data
 * @param[in] bind_tab binding_table
 * @param[in] bind_tab_num_ent number of entries in binding table
 * @retval AEE_SUCCESS for success
 */
static int hexa_cmd_buf_exec_copy(hexa_cmd_buf_sched_t *sched,
                                  const uint8_t **cmd_buf_data,
                                  int *cmd_buf_size,
                                  const hexagon_rt_arm_dsp_binding_t *bind_tab,
                                  uint32_t bind_tab_num_ent) {
  READ_SERIALIZED(*cmd_buf_data, *cmd_buf_size, hexagon_rt_arm_dsp_cmd_copy_t,
                  cmd_copy)
  // A buffer that does not fit into the residency budget next to the ones of
  // the dispatches in flight is mapped again once all of them have finished.
  hexa_cmd_buf_res_buf_t src = {};
  uint32_t entry_position = sched->future.position;
  int err = hexa_cmd_buf_resolve_and_map(
      sched->residency, &sched->future, &cmd_copy->src, bind_tab,
      bind_tab_num_ent, HEXAGON_DSP_CACHE_ACCESS_CACHED_READ, &src);
  if (err == AEE_EWOULDBLOCK) {
    hexa_cmd_buf_sched_drain(sched);
    sched->future.position = entry_position; // reset on acquire failure
    err = hexa_cmd_buf_resolve_and_map(
        sched->residency, &sched->future, &cmd_copy->src, bind_tab,
        bind_tab_num_ent, HEXAGON_DSP_CACHE_ACCESS_CACHED_READ, &src);
  }
  if (err != AEE_SUCCESS) {
    return err == AEE_EWOULDBLOCK ? hexa_cmd_buf_residency_exceeded("copy")
                                  : err;
  }
  hexa_cmd_buf_res_buf_t dest = {};
  entry_position = sched->future.position;
  err = hexa_cmd_buf_resolve_and_map(
      sched->residency, &sched->future, &cmd_copy->trgt, bind_tab,
      bind_tab_num_ent, HEXAGON_DSP_CACHE_ACCESS_CACHED_WRITE, &dest);
  if (err == AEE_EWOULDBLOCK) {
    hexa_cmd_buf_sched_drain(sched);
    sched->future.position = entry_position; // reset on acquire failure
    err = hexa_cmd_buf_resolve_and_map(
        sched->residency, &sched->future, &cmd_copy->trgt, bind_tab,
        bind_tab_num_ent, HEXAGON_DSP_CACHE_ACCESS_CACHED_WRITE, &dest);
  }
  if (err != AEE_SUCCESS) {
    hexagon_dsp_residency_release(sched->residency, src.fd);
    return err == AEE_EWOULDBLOCK ? hexa_cmd_buf_residency_exceeded("copy")
                                  : err;
  }

  // copy data (through the cache, see hexa_cmd_buf_exec_buf)
  memcpy(dest.dsp_vaddr + dest.offset, src.dsp_vaddr + src.offset,
         dest.length < src.length ? dest.length : src.length);

  hexagon_dsp_residency_release(sched->residency, dest.fd);
  hexagon_dsp_residency_release(sched->residency, src.fd);
  return AEE_SUCCESS;
}

/**
 * @brief Execute fill command buffer entry.
 * @param[in,out] sched scheduler state
 * @param[in,out] cmd_buf_data pointer to pointer to serialized fill command
 *                data, updated by amount of processed data
 * @param[in,out] cmd_buf_size pointer to size of serialized command buffer
 *                data, updated by amount of processed data
 * @param[in] bind_tab binding_table
 * @param[in] bind_tab_num_ent number of entries in binding table
 * @retval AEE_SUCCESS for success
 */
static int hexa_cmd_buf_exec_fill(hexa_cmd_buf_sched_t *sched,
                                  const uint8_t **cmd_buf_data,
                                  int *cmd_buf_size,
                                  const hexagon_rt_arm_dsp_binding_t *bind_tab,
                                  uint32_t bind_tab_num_ent) {
  READ_SERIALIZED(*cmd_buf_data, *cmd_buf_size, hexagon_rt_arm_dsp_cmd_fill_t,
                  cmd_fill)
  hexa_cmd_buf_res_buf_t dest = {};
  uint32_t entry_position = sched->future.position;
  int err = hexa_cmd_buf_resolve_and_map(
      sched->residency, &sched->future, &cmd_fill->trgt, bind_tab,
      bind_tab_num_ent, HEXAGON_DSP_CACHE_ACCESS_CACHED_WRITE, &dest);
  if (err == AEE_EWOULDBLOCK) {
    // retry once all dispatches have released their buffers
    hexa_cmd_buf_sched_drain(sched);
    sched->future.position = entry_position; // reset on acquire failure
    err = hexa_cmd_buf_resolve_and_map(
        sched->residency, &sched->future, &cmd_fill->trgt, bind_tab,
        bind_tab_num_ent, HEXAGON_DSP_CACHE_ACCESS_CACHED_WRITE, &dest);
  }
  if (err == AEE_EWOULDBLOCK) {
    err = hexa_cmd_buf_residency_exceeded("fill");
  }
  if (err != AEE_SUCCESS) {
    return err;
  }

  // fill buffer with pattern (through the cache, see hexa_cmd_buf_exec_buf)
  uint8_t p = 0;
  for (uint64_t i = 0; i < cmd_fill->trgt.length; ++i) {
    dest.dsp_vaddr[dest.offset + i] = cmd_fill->pattern[p];
    ++p;
    if (p >= cmd_fill->pattern_length) {
      p = 0;
    }
  }

  hexagon_dsp_residency_release(sched->residency, dest.fd);
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
 * Stops once a dispatch has failed and returns its error.
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
    // skip the rest of the command buffer once a dispatch has failed
    if (sched->status != AEE_SUCCESS) {
      return sched->status;
    }
    PEEK_SERIALIZED(cmd_buf_data, cmd_buf_size, hexagon_rt_arm_dsp_cmd_base_t,
                    cmd_base)
    switch ((hexagon_rt_arm_dsp_cmd_type_enum_t)cmd_base->cmd_type) {

    case HEXAGON_RT_ARM_DSP_CMD_DISPATCH: {
      // profiled by a SCHEDULE zone inside
      int err =
          hexa_cmd_buf_exec_dispatch(sched, &cmd_buf_data, &cmd_buf_size,
                                     bind_tab, bind_tab_num_ent, prof_context);
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
      int err = hexa_cmd_buf_exec_copy(sched, &cmd_buf_data, &cmd_buf_size,
                                       bind_tab, bind_tab_num_ent);
      profiler_measurement_finish_and_record(prof_record);
      if (err != AEE_SUCCESS) {
        return err;
      }
      break;
    }

    case HEXAGON_RT_ARM_DSP_CMD_FILL: {
      hexagon_rt_prof_record_t *prof_record =
          profiler_measurement_start(prof_context, FILL, NULL);
      int err = hexa_cmd_buf_exec_fill(sched, &cmd_buf_data, &cmd_buf_size,
                                       bind_tab, bind_tab_num_ent);
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

  // every recorded buffer use was acquired
  assert(sched->future.position == sched->future.command_buffer->num_use_refs);
  return AEE_SUCCESS;
}

/**
 * @brief Execute the commands of a command buffer (without cache maintenance).
 * @param[in] command_buffer command buffer to execute
 * @param[in] bind_tab binding_table
 * @param[in] bind_tab_num_ent number of entries in binding table
 * @param[in] max_concurrent max. number of dispatches executing at the same
 *                           time, 1 for sequential execution on this thread
 * @param[in] residency residency cache of the RPC session
 * @param[in] prof_context profiler context if tracing enabled, NULL otherwise
 * @retval AEE_SUCCESS for success
 *
 * The calling thread (the master) walks the command buffer. Dispatches that
 * do not use HMX are submitted to the AsyncRuntime worker threads as soon as
 * an execution slot and their VTCM partition are available; HMX dispatches
 * run on the master, which holds the HMX lock.
 * All dispatches have completed when this function returns, also on error.
 */
static int
hexa_cmd_buf_exec_cmds(const hexagon_dsp_command_buffer_t *command_buffer,
                       const hexagon_rt_arm_dsp_binding_t *bind_tab,
                       uint32_t bind_tab_num_ent, uint32_t max_concurrent,
                       hexagon_dsp_residency_t *residency,
                       hexagon_rt_prof_context_t *prof_context) {
  if (max_concurrent < 1) {
    max_concurrent = 1;
  }
  hexa_cmd_buf_sched_t sched = {
      .max_concurrent = max_concurrent,
      .tasks = NULL,
      .num_in_flight = 0,
      // Also binds the VTCM pool and the HMX lock to this thread before any
      // worker exists.
      .vtcm_pool_size = hexagon_rt_vtcm_pool_size(),
      .vtcm_in_use = 0,
      .status = AEE_SUCCESS,
      .residency = residency,
  };
  sched.future = (hexa_cmd_buf_future_t){
      .base = {.next_use = hexa_cmd_buf_next_use, .user_data = &sched.future},
      .command_buffer = command_buffer,
      .bind_tab = bind_tab,
      .bind_tab_num_ent = bind_tab_num_ent,
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

  err = hexa_cmd_buf_walk_cmds(&sched, command_buffer->cmd_buf_data,
                               command_buffer->cmd_buf_size, bind_tab,
                               bind_tab_num_ent, prof_context);

  // wait fial all outstanding tasks to complete
  hexa_cmd_buf_sched_drain(&sched);
  // report dispatches that failed after the last command was walked
  if (err == AEE_SUCCESS) {
    // dont override another error from walking the cmd buffer
    err = sched.status;
  }

  qurt_cond_destroy(&sched.done_notify.cond);
  qurt_mutex_destroy(&sched.done_notify.mutex);
  for (uint32_t idx = 0; idx < max_concurrent; ++idx) {
    HAP_free(sched.tasks[idx].dispatch_arrays);
  }
  HAP_free(sched.future.sorted_uses);
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
 * Cache maintenance: the buffers the host may have written (binding table
 * entries and direct buffer references) and that are still mapped from
 * earlier are invalidated before the first command. After the last command,
 * what was written to buffers the host may read is flushed. The commands in
 * between all go through the cache and thus share the same view of it. The
 * residency cache does the maintenance of each access, and flushes and
 * invalidates allocations before it unmaps them (see cache_state.h).
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
  hexagon_dsp_residency_t *residency =
      hexagon_dsp_get_residency(command_buffer->rpc_handle);

  // invalidate what the host may have written since the last execution
  hexagon_rt_prof_record_t *prof_invalidate_record =
      profiler_measurement_start(prof_context, MEMORY_MANAGEMENT, NULL);
  int err = hexa_cmd_buf_host_sync(residency, bind_tab, bind_tab_num_ent);
  if (err == AEE_SUCCESS) {
    err = hexa_cmd_buf_host_sync(residency, command_buffer->direct_bufs,
                                 command_buffer->num_direct_bufs);
  }
  profiler_measurement_finish_and_record(prof_invalidate_record);
  if (err != AEE_SUCCESS) {
    return err;
  }

  err = hexa_cmd_buf_exec_cmds(
      command_buffer, bind_tab, bind_tab_num_ent,
      hexagon_dsp_get_max_concurrent_dispatches(command_buffer->rpc_handle),
      residency, prof_context);

  hexagon_rt_prof_record_t *prof_flush_record =
      profiler_measurement_start(prof_context, MEMORY_MANAGEMENT, NULL);
  // flush everything, so the host can read, even on error
  int err_flush = hexagon_dsp_residency_end_execution(residency);
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
