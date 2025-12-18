// Copyright 2025 RooflineAI GmbH

#include "cmd_copy_serialize.h"

#include <stdint.h>

#include "command_buffer_types.h"
#include "hexagon/arm_dsp/cmd_buf.h"
#include "iree/base/api.h"
#include "serialize.h"

iree_status_t
iree_hal_hexagon_cmd_copy_serialize_prep(iree_host_size_t *out_cmd_size) {
  *out_cmd_size = sizeof(hexagon_rt_arm_dsp_cmd_copy_t);
  return iree_ok_status();
}

iree_status_t iree_hal_hexagon_cmd_copy_serialize_exec(
    const iree_hal_buffer_ref_t *source_ref,
    const iree_hal_buffer_ref_t *target_ref,
    iree_hal_hexagon_buffer_to_dsp_vaddr_t buffer_to_dsp_vaddr,
    uint8_t *cmd_data, iree_host_size_t cmd_size) {

  uint8_t *ptr = cmd_data;
  uint8_t *endptr = cmd_data + cmd_size;

  // serialize copy command
  SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_cmd_copy_t, copy)
  copy->base.cmd_type = HEXAGON_RT_ARM_DSP_CMD_COPY;

  // fill in source buffer
  rpc_dsp_vaddr_t src_dsp_vaddr =
      0; // stays at 0 if no buffer, means to use slot
  if (source_ref->buffer) {
    IREE_RETURN_IF_ERROR(
        buffer_to_dsp_vaddr(source_ref->buffer, &src_dsp_vaddr));
  }
  copy->src.slot = source_ref->buffer_slot;
  copy->src.buffer_dsp_vaddr = src_dsp_vaddr;
  copy->src.offset = source_ref->offset;
  copy->src.length = source_ref->length;

  // fill in target buffer
  rpc_dsp_vaddr_t trgt_dsp_vaddr =
      0; // stays at 0 if no buffer, means to use slot
  if (target_ref->buffer) {
    IREE_RETURN_IF_ERROR(
        buffer_to_dsp_vaddr(target_ref->buffer, &trgt_dsp_vaddr));
  }
  copy->trgt.slot = target_ref->buffer_slot;
  copy->trgt.buffer_dsp_vaddr = trgt_dsp_vaddr;
  copy->trgt.offset = target_ref->offset;
  copy->trgt.length = target_ref->length;

  return iree_ok_status();
}
