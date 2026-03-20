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
    iree_hal_hexagon_get_buffer_fd_t get_buffer_fd, uint8_t *cmd_data,
    iree_host_size_t cmd_size) {

  uint8_t *ptr = cmd_data;
  uint8_t *endptr = cmd_data + cmd_size;

  // serialize copy command
  SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_cmd_copy_t, copy)
  copy->base.cmd_type = HEXAGON_RT_ARM_DSP_CMD_COPY;

  // fill in source buffer
  int src_fd = -1; // stays at -1 if no buffer, means to use slot
  if (source_ref->buffer) {
    IREE_RETURN_IF_ERROR(get_buffer_fd(source_ref->buffer, &src_fd));
  }
  copy->src.slot = source_ref->buffer_slot;
  copy->src.fd = src_fd;
  copy->src.offset = source_ref->offset;
  copy->src.length = source_ref->length;

  // fill in target buffer
  int trgt_fd = -1; // stays at -1 if no buffer, means to use slot
  if (target_ref->buffer) {
    IREE_RETURN_IF_ERROR(get_buffer_fd(target_ref->buffer, &trgt_fd));
  }
  copy->trgt.slot = target_ref->buffer_slot;
  copy->trgt.fd = trgt_fd;
  copy->trgt.offset = target_ref->offset;
  copy->trgt.length = target_ref->length;

  return iree_ok_status();
}
