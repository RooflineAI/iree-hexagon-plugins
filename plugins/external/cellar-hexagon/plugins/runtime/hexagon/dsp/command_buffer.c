// Copyright 2025 RooflineAI GmbH

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "HAP_mem.h"
#include "hexagon/arm_dsp/bindings.h"
#include "hexagon/arm_dsp/cmd_buf.h"
#include "hexagon/arm_dsp/profiling.h"
#include "hexagon/dsp/align.h"
#include "hexagon/dsp/executable.h"
#include "hexagon/dsp/pmu/hexagon_pmu.h"
#include "hexagon/dsp/profiling.h"
#include "hexagon_dsp.h"
#include "iree/hal/local/executable_library.h"
#include "qurt.h"

/// data about a command buffer
typedef struct hexagon_dsp_command_buffer_s {
  /// handle of owning RPC session
  remote_handle64 rpc_handle;
  uint8_t *cmd_buf_data; ///< contains hexagon_rt_arm_dsp_cmd_buf
  size_t cmd_buf_size;
} hexagon_dsp_command_buffer_t;

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
  HAP_free(command_buffer->cmd_buf_data);
  // free management data structure
  HAP_free(command_buffer);
  return AEE_SUCCESS;
}

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

/**
 * @brief Execute dispatch command buffer entry.
 * @param[in,out] cmd_buf_data pointer to pointer to serialized barrier command
 *                data, updated by amount of processed data
 * @param[in,out] cmd_buf_size pointer to size of serialized command buffer
 *                data, updated by amount of processed data
 * @param[in] bind_tab binding_table
 * @param[in] bind_tab_num_ent number of entries in binding table
 * @param[in] profiling_header profiling records header, null when tracing is
 * disabled
 * @param[in] profiling_records profiling records base, null when tracing is
 * disabled
 * @retval AEE_SUCCESS for success
 */
static int
hexa_cmd_buf_exec_dispatch(const uint8_t **cmd_buf_data, int *cmd_buf_size,
                           const hexagon_rt_arm_dsp_binding_t *bind_tab,
                           uint32_t bind_tab_num_ent,
                           hexagon_rt_prof_header_t *profiling_header,
                           hexagon_rt_prof_record_t *profiling_records) {
  READ_SERIALIZED(*cmd_buf_data, *cmd_buf_size,
                  hexagon_rt_arm_dsp_cmd_dispatch_t, cmd_dispatch)
  int64_t executable_handle = cmd_dispatch->executable_handle;
  uint32_t export_ordinal = cmd_dispatch->export_ordinal;
  uint32_t num_buf_refs = cmd_dispatch->num_bindings;

  // Obtain dispatch function pointer.
  iree_hal_executable_dispatch_v0_t dispatch_func = NULL;
  int err = hexagon_dsp_executable_get_dispatch_func(
      executable_handle, export_ordinal, &dispatch_func);
  if (err != AEE_SUCCESS) {
    return err;
  }

  // Allocate buffer for nested arrays in dispatch state.
  // The dispatch state contains some pointers, which are arrays semantically.
  // These pointers need to point to actual arrays, which are allocated here.
  // To reduce the number of HAL_malloc calls, the arrays are allocated in the
  // same buffer and the start addresses are computed based on their size.
  // The data is written to the arrays below the definition of the dispatch
  // data structure.
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

  // Build dispatch state - must be aligned at 16 bytes, because dispatch
  // function declarations assumes that alignment
  // FIXME: max_concurrency still hard coded to 1, need to be implemented
  // properly
  HEXAGON_ALIGNAS(16)
  iree_hal_executable_dispatch_state_v0_t dispatch_state = {
      .workgroup_size_x = cmd_dispatch->workgroup_size_x,
      .workgroup_size_y = cmd_dispatch->workgroup_size_y,
      .workgroup_size_z = cmd_dispatch->workgroup_size_z,
      .constant_count = cmd_dispatch->constant_count,
      .workgroup_count_x = cmd_dispatch->workgroup_count_x,
      .workgroup_count_y = cmd_dispatch->workgroup_count_y,
      .workgroup_count_z = cmd_dispatch->workgroup_count_z,
      .max_concurrency = 1,
      .binding_count = num_buf_refs,
      .constants = constants,
      .binding_ptrs = binding_ptrs,
      .binding_lengths = binding_lengths,
  };

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
  }

  profiler_measurement_start(profiling_header, profiling_records,
                             MEMORY_MANAGEMENT);
  // invalidate cache of buffers
  for (uint32_t idx_buf_ref = 0; idx_buf_ref < num_buf_refs; ++idx_buf_ref) {
    err = qurt_mem_cache_clean((qurt_addr_t)binding_ptrs[idx_buf_ref],
                               binding_lengths[idx_buf_ref],
                               QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    if (err != QURT_EOK) {
      // according to doc, the only error is QURT_EVAL - invalid cache type
      hexa_cmd_buf_unmap(mapped_fds, num_buf_refs);
      HAP_free(dispatch_arrays);
      return AEE_EFAILED;
    }
  }

  profiler_measurement_finish_and_record(profiling_header, profiling_records);

  // Note that this function is also called when profiling is disabled, but it
  // is not very expensive. We do not care if it fails either.
  const char *func_name = NULL;
  hexagon_dsp_executable_get_dispatch_func_name(executable_handle,
                                                export_ordinal, &func_name);

  profiler_measurement_start_extra_info(profiling_header, profiling_records,
                                        KERNEL, func_name);

  HEXAGON_ALIGNAS(16)
  iree_hal_executable_environment_v0_t environment = {0};

  // for now, run all workgroups sequentially
  for (uint16_t wg_id_z = 0; wg_id_z < dispatch_state.workgroup_count_z;
       ++wg_id_z) {
    for (uint32_t wg_id_y = 0; wg_id_y < dispatch_state.workgroup_count_y;
         ++wg_id_y) {
      for (uint32_t wg_id_x = 0; wg_id_x < dispatch_state.workgroup_count_x;
           ++wg_id_x) {

        // build workgroup state - must be aligned at 16 bytes, because dispatch
        // function declarations assumes that alignment
        HEXAGON_ALIGNAS(16)
        iree_hal_executable_workgroup_state_v0_t workgroup_state = {
            .workgroup_id_x = wg_id_x,
            .workgroup_id_y = wg_id_y,
            .workgroup_id_z = wg_id_z,
            .processor_id = 0,
            .local_memory = NULL,
            .local_memory_size = 0,
        };

        dispatch_func(&environment, &dispatch_state, &workgroup_state);
      }
    }
  }

  profiler_measurement_finish_and_record(profiling_header, profiling_records);

  profiler_measurement_start(profiling_header, profiling_records,
                             MEMORY_MANAGEMENT);

  // flush cache of buffers
  for (uint32_t idx_buf_ref = 0; idx_buf_ref < num_buf_refs; ++idx_buf_ref) {
    err = qurt_mem_cache_clean((qurt_addr_t)binding_ptrs[idx_buf_ref],
                               binding_lengths[idx_buf_ref],
                               QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    if (err != QURT_EOK) {
      // according to doc, the only error is QURT_EVAL - invalid cache type
      hexa_cmd_buf_unmap(mapped_fds, num_buf_refs);
      HAP_free(dispatch_arrays);
      return AEE_EFAILED;
    }
  }

  hexa_cmd_buf_unmap(mapped_fds, num_buf_refs);
  HAP_free(dispatch_arrays);

  profiler_measurement_finish_and_record(profiling_header, profiling_records);

  return AEE_SUCCESS;
}

/**
 * @brief Execute barrier command buffer entry.
 * @param[in,out] cmd_buf_data pointer to pointer to serialized barrier command
 *                data, updated by amount of processed data
 * @param[in,out] cmd_buf_size pointer to size of serialized command buffer
 *                data, updated by amount of processed data
 * @retval AEE_SUCCESS for success
 */
static int hexa_cmd_buf_exec_barrier(const uint8_t **cmd_buf_data,
                                     int *cmd_buf_size) {
  READ_SERIALIZED(*cmd_buf_data, *cmd_buf_size,
                  hexagon_rt_arm_dsp_cmd_barrier_t, cmd_barrier)
  // nothing do do here for now
  (void)cmd_barrier;
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

  // invalidate cache of input buffer
  err =
      qurt_mem_cache_clean((qurt_addr_t)src.dsp_vaddr + src.offset, src.length,
                           QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
  if (err != QURT_EOK) {
    // according to doc, the only error is QURT_EVAL - invalid cache type
    HAP_mmap_put(dest.fd);
    HAP_mmap_put(src.fd);
    return AEE_EFAILED;
  }

  // copy data
  memcpy(dest.dsp_vaddr + dest.offset, src.dsp_vaddr + src.offset,
         dest.length < src.length ? dest.length : src.length);

  // flush cache of output buffer
  err =
      qurt_mem_cache_clean((qurt_addr_t)dest.dsp_vaddr + dest.offset,
                           dest.length, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
  if (err != QURT_EOK) {
    // according to doc, the only error is QURT_EVAL - invalid cache type
    HAP_mmap_put(dest.fd);
    HAP_mmap_put(src.fd);
    return AEE_EFAILED;
  }

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

  // fill buffer with pattern
  uint8_t p = 0;
  for (uint64_t i = 0; i < cmd_fill->trgt.length; ++i) {
    dest.dsp_vaddr[dest.offset + i] = cmd_fill->pattern[p];
    ++p;
    if (p >= cmd_fill->pattern_length) {
      p = 0;
    }
  }

  // flush cache of output buffer
  err =
      qurt_mem_cache_clean((qurt_addr_t)dest.dsp_vaddr + dest.offset,
                           dest.length, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
  if (err != QURT_EOK) {
    // according to doc, the only error is QURT_EVAL - invalid cache type
    HAP_mmap_put(dest.fd);
    return AEE_EFAILED;
  }

  HAP_mmap_put(dest.fd);
  return AEE_SUCCESS;
}

/**
 * @brief Execute command buffer.
 * @param[in] cmd_buf_data pointer to serialized command buffer data
 * @param[in] cmd_buf_size size of serialized command buffer data
 * @param[in] bind_tab_data pointer to serialized binding table data
 * @param[in] bind_tab_size size of serialized binding table data
 * @retval AEE_SUCCESS for success
 */
static int
hexa_cmd_buf_exec_buf(const uint8_t *cmd_buf_data, int cmd_buf_size,
                      const uint8_t *bind_tab_data, int bind_tab_size,
                      hexagon_rt_prof_header_t *profiling_header_pointer,
                      hexagon_rt_prof_record_t *profiling_records) {
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

  // Process command buffer from serialized representation, entry by entry.
  READ_SERIALIZED(cmd_buf_data, cmd_buf_size, hexagon_rt_arm_dsp_cmd_buf_t,
                  cmd_buf)
  for (uint32_t idx_entry = 0; idx_entry < cmd_buf->num_entries; ++idx_entry) {
    PEEK_SERIALIZED(cmd_buf_data, cmd_buf_size, hexagon_rt_arm_dsp_cmd_base_t,
                    cmd_base)
    switch ((hexagon_rt_arm_dsp_cmd_type_enum_t)cmd_base->cmd_type) {

    case HEXAGON_RT_ARM_DSP_CMD_DISPATCH: {
      profiler_measurement_start(profiling_header_pointer, profiling_records,
                                 DISPATCH);
      int err = hexa_cmd_buf_exec_dispatch(
          &cmd_buf_data, &cmd_buf_size, bind_tab, bind_tab_num_ent,
          profiling_header_pointer, profiling_records);
      profiler_measurement_finish_and_record(profiling_header_pointer,
                                             profiling_records);
      if (err != AEE_SUCCESS) {
        return err;
      }
      break;
    }

    case HEXAGON_RT_ARM_DSP_CMD_BARRIER: {
      profiler_measurement_start(profiling_header_pointer, profiling_records,
                                 BARRIER);
      int err = hexa_cmd_buf_exec_barrier(&cmd_buf_data, &cmd_buf_size);
      profiler_measurement_finish_and_record(profiling_header_pointer,
                                             profiling_records);
      if (err != AEE_SUCCESS) {
        return err;
      }
      break;
    }

    case HEXAGON_RT_ARM_DSP_CMD_COPY: {
      profiler_measurement_start(profiling_header_pointer, profiling_records,
                                 COPY);
      int err = hexa_cmd_buf_exec_copy(&cmd_buf_data, &cmd_buf_size, bind_tab,
                                       bind_tab_num_ent);
      profiler_measurement_finish_and_record(profiling_header_pointer,
                                             profiling_records);
      if (err != AEE_SUCCESS) {
        return err;
      }
      break;
    }

    case HEXAGON_RT_ARM_DSP_CMD_FILL: {
      profiler_measurement_start(profiling_header_pointer, profiling_records,
                                 FILL);
      int err = hexa_cmd_buf_exec_fill(&cmd_buf_data, &cmd_buf_size, bind_tab,
                                       bind_tab_num_ent);
      profiler_measurement_finish_and_record(profiling_header_pointer,
                                             profiling_records);
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
  int err = hexa_cmd_buf_exec_buf(
      command_buffer->cmd_buf_data, command_buffer->cmd_buf_size,
      binding_table_data, binding_table_size, NULL, NULL);
  return err;
}

/**
 * @brief Execute a command buffer while logging profiling information. The PMU
 * counters are not isolated when executing multiple command buffers at once and
 * will output garbage!
 * This rpc is duplicated in order to manage how NULL is controlled without
 * relying on the RPC mechanism (crashes when passing NULL).
 * Additionally, profiling buffer sizing is manually incremented during command
 * buffer creation, while the actual number of measurement zones is defined on
 * the DSP side. Any change to the profiling must be reflected on both sides!
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
int hexagon_dsp_command_buffer_execute_profiling(
    remote_handle64 rpc_handle, int64 command_buffer_handle,
    const uint8 *binding_table_data, int binding_table_size,
    uint8 *performance_log_data, int performance_log_size) {
  hexagon_dsp_command_buffer_t *command_buffer =
      (hexagon_dsp_command_buffer_t *)command_buffer_handle;

  if (!performance_log_data) {
    FARF(
        RUNTIME_HIGH,
        "HEXAGON-RUNTIME-ERROR: Missing performance log, stopping execution\n");
    return AEE_EBADPARM;
  }

  READ_SERIALIZED_MUT(performance_log_data, performance_log_size,
                      hexagon_rt_prof_header_t, profiling_header)

  if (performance_log_size <
      profiling_header->num_records * sizeof(hexagon_rt_prof_record_t)) {
    return AEE_EINCOMPLETEITEM;
  }
  hexagon_rt_prof_record_t *profiling_records =
      (hexagon_rt_prof_record_t *)performance_log_data;

  // Setting the PMU unit, note that this assumes only one command buffer is
  // being executed at once!
  qurt_pmu_enable(0); // Reset PMU registers
  hexagon_pmu_configure(&profiling_header->pmu_event_ids);
  qurt_pmu_enable(1);

  profiler_measurement_start(profiling_header, profiling_records,
                             DSP_EXECUTION);

  int err = hexa_cmd_buf_exec_buf(command_buffer->cmd_buf_data,
                                  command_buffer->cmd_buf_size,
                                  binding_table_data, binding_table_size,
                                  profiling_header, profiling_records);

  profiler_measurement_finish_and_record(profiling_header, profiling_records);

  return err;
}
