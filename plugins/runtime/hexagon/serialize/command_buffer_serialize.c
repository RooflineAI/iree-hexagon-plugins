// Copyright 2025 RooflineAI GmbH

#include "command_buffer_serialize.h"

#include <stdint.h>

#include "command_buffer_types.h"
#include "hexagon/arm_dsp/cmd_buf.h"
#include "hexagon/serialize/rpc_types.h"
#include "iree/base/api.h"
#include "rpc_types.h"
#include "serialize.h"

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
                          sizeof(hexagon_rt_arm_dsp_buf_ref_t);
      break;
    }

    case IREE_HAL_HEXAGON_COMMAND_BARRIER: {
      cmd_buf_size += sizeof(hexagon_rt_arm_dsp_cmd_barrier_t);
      break;
    }

    case IREE_HAL_HEXAGON_COMMAND_COPY: {
      cmd_buf_size += sizeof(hexagon_rt_arm_dsp_cmd_copy_t);
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

iree_status_t iree_hal_hexagon_command_buffer_serialize_exec(
    const iree_hal_hexagon_command_buffer_t *command_buffer,
    iree_status_t (*buffer_to_dsp_vaddr)(iree_hal_buffer_t *buffer,
                                         rpc_dsp_vaddr_t *out_dsp_vaddr),
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
        rpc_dsp_vaddr_t dsp_vaddr =
            0; // stays at 0 if no buffer, means to use slot
        if (binding->buffer) {
          IREE_RETURN_IF_ERROR(
              buffer_to_dsp_vaddr(binding->buffer, &dsp_vaddr));
        }
        SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_buf_ref_t, bndg)
        bndg->slot = binding->buffer_slot;
        bndg->buffer_dsp_vaddr = dsp_vaddr;
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

    case IREE_HAL_HEXAGON_COMMAND_COPY: {
      const iree_hal_hexagon_command_copy_t *command_copy =
          (const iree_hal_hexagon_command_copy_t *)command;
      rpc_dsp_vaddr_t src_dsp_vaddr =
          0; // stays at 0 if no buffer, means to use slot
      if (command_copy->src.buffer) {
        IREE_RETURN_IF_ERROR(
            buffer_to_dsp_vaddr(command_copy->src.buffer, &src_dsp_vaddr));
      }
      rpc_dsp_vaddr_t dest_dsp_vaddr =
          0; // stays at 0 if no buffer, means to use slot
      if (command_copy->dest.buffer) {
        IREE_RETURN_IF_ERROR(
            buffer_to_dsp_vaddr(command_copy->dest.buffer, &dest_dsp_vaddr));
      }
      SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_cmd_copy_t, cmd_copy)
      cmd_copy->base.cmd_type = HEXAGON_RT_ARM_DSP_CMD_COPY;
      cmd_copy->src.slot = command_copy->src.buffer_slot;
      cmd_copy->src.buffer_dsp_vaddr = src_dsp_vaddr;
      cmd_copy->src.offset = command_copy->src.offset;
      cmd_copy->src.length = command_copy->src.length;
      cmd_copy->dest.slot = command_copy->dest.buffer_slot;
      cmd_copy->dest.buffer_dsp_vaddr = dest_dsp_vaddr;
      cmd_copy->dest.offset = command_copy->dest.offset;
      cmd_copy->dest.length = command_copy->dest.length;
      break;
    }

    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unknown command type in command buffer");
    }
  }

  return iree_ok_status();
}
