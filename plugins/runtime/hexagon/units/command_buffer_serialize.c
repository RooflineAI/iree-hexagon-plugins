// Copyright 2025 RooflineAI GmbH

#include "command_buffer_serialize.h"

#include <stdint.h>

#include "command_buffer_types.h"
#include "hexagon/arm_dsp/cmd_buf.h"
#include "iree/base/api.h"
#include "rpc_types.h"

iree_status_t iree_hal_hexagon_command_buffer_serialize_prep(
    const iree_hal_hexagon_command_buffer_t *command_buffer,
    iree_host_size_t *out_cmd_buf_size, uint32_t *out_num_entries) {
  // start with size of command buffer header
  iree_host_size_t cmd_buf_size = sizeof(hexagon_rt_arm_dsp_cmd_buf_t);
  uint32_t num_entries = 0;

  // add size for all commands
  for (const iree_hal_hexagon_command_base_t *command =
           command_buffer->first_entry;
       command; command = command->next) {
    iree_host_size_t prev_size = cmd_buf_size;

    switch ((hexagon_rt_arm_dsp_cmd_type_enum_t)command->cmd_type) {
    case IREE_HAL_HEXAGON_COMMAND_DISPATCH: {
      const iree_hal_hexagon_command_dispatch_t *command_dispatch =
          (const iree_hal_hexagon_command_dispatch_t *)command;
      if (command_dispatch->bindings.count > UINT32_MAX) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "too many bindings in command in command buffer");
      }
      cmd_buf_size += sizeof(hexagon_rt_arm_dsp_cmd_dispatch_t) +
                      command_dispatch->bindings.count *
                          sizeof(hexagon_rt_arm_dsp_binding_t);
      break;
    }

    case IREE_HAL_HEXAGON_COMMAND_BARRIER: {
      cmd_buf_size += sizeof(hexagon_rt_arm_dsp_cmd_barrier_t);
      break;
    }

    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unknown command type in command buffer");
    }

    // count number of entries and check for overflow
    ++num_entries;
    if (cmd_buf_size < prev_size || num_entries == 0 /* overflow */) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "too many entries in command buffer");
    }
  }

  // return data size and number of entries
  *out_cmd_buf_size = cmd_buf_size;
  *out_num_entries = num_entries;
  return iree_ok_status();
}

/// Put a data structure of type T into the serialization buffer with current
/// pointer P and end pointer E.
/// Make a pointer V to the data structure for assigning values.
/// note: Because the ARM/DSP data structures are packed ones, casting uint8_t*
/// to struct pointers is okay.
#define SERIALIZE_TO(P, E, T, V)                                               \
  if (P + sizeof(T) > E) {                                                     \
    return iree_make_status(IREE_STATUS_INTERNAL,                              \
                            "size computation and actual serialization of "    \
                            "command buffer did not match");                   \
  }                                                                            \
  T *V = (T *)P;                                                               \
  P += sizeof(T);

iree_status_t iree_hal_hexagon_command_buffer_serialize_exec(
    const iree_hal_hexagon_command_buffer_t *command_buffer,
    uint32_t num_entries, uint8_t *cmd_buf_data,
    iree_host_size_t cmd_buf_size) {
  uint8_t *ptr = cmd_buf_data;
  uint8_t *endptr = cmd_buf_data + cmd_buf_size;

  // serialize command buffer header
  SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_cmd_buf_t, cmd_buf)
  cmd_buf->num_entries = num_entries;

  // serialize all commands
  for (const iree_hal_hexagon_command_base_t *command =
           command_buffer->first_entry;
       command; command = command->next) {
    switch ((hexagon_rt_arm_dsp_cmd_type_enum_t)command->cmd_type) {
    case IREE_HAL_HEXAGON_COMMAND_DISPATCH: {
      const iree_hal_hexagon_command_dispatch_t *command_dispatch =
          (const iree_hal_hexagon_command_dispatch_t *)command;
      SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_cmd_dispatch_t, cmd_dispatch)
      cmd_dispatch->base.cmd_type = HEXAGON_RT_ARM_DSP_CMD_DISPATCH;
      cmd_dispatch->executable_handle = command_dispatch->rpc_executable_handle;
      cmd_dispatch->export_ordinal = command_dispatch->export_ordinal;
      cmd_dispatch->num_bindings = command_dispatch->bindings.count;
      for (uint32_t b = 0; b < cmd_dispatch->num_bindings; ++b) {
        const iree_hal_buffer_ref_t *binding =
            &command_dispatch->bindings.values[b];
        SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_binding_t, bndg)
        bndg->slot = binding->buffer_slot;
        bndg->buffer_handle =
            0; // TODO: get buffer_handle from inside binding->buffer
        bndg->offset = binding->offset;
        bndg->length = binding->length;
      }
      break;
    }

    case IREE_HAL_HEXAGON_COMMAND_BARRIER: {
      SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_cmd_barrier_t, cmd_barrier)
      cmd_barrier->base.cmd_type = HEXAGON_RT_ARM_DSP_CMD_BARRIER;
      break;
    }

    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unknown command type in command buffer");
    }
  }

  return iree_ok_status();
}
