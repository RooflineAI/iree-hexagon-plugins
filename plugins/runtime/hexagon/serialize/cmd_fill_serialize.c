// Copyright 2025 RooflineAI GmbH

#include "cmd_fill_serialize.h"

#include <stdint.h>

#include "command_buffer_types.h"
#include "hexagon/arm_dsp/cmd_buf.h"
#include "iree/base/api.h"
#include "serialize.h"

iree_status_t
iree_hal_hexagon_cmd_fill_serialize_prep(iree_host_size_t *out_cmd_size) {
  *out_cmd_size = sizeof(hexagon_rt_arm_dsp_cmd_fill_t);
  return iree_ok_status();
}

iree_status_t iree_hal_hexagon_cmd_fill_serialize_exec(
    uint8_t pattern_length, const uint8_t *pattern,
    const iree_hal_buffer_ref_t *target_ref,
    iree_hal_hexagon_buffer_to_dsp_vaddr_t buffer_to_dsp_vaddr,
    uint8_t *cmd_data, iree_host_size_t cmd_size) {

  uint8_t *ptr = cmd_data;
  uint8_t *endptr = cmd_data + cmd_size;

  // serialize fill command
  SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_cmd_fill_t, fill)
  fill->base.cmd_type = HEXAGON_RT_ARM_DSP_CMD_FILL;

  // store pattern
  if (pattern_length > sizeof(fill->pattern)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "pattern lengths greater than %u are not supported on Hexagon",
        (unsigned int)sizeof(fill->pattern));
  }
  fill->pattern_length = pattern_length;
  memcpy(fill->pattern, pattern, pattern_length);
  memset(fill->pattern + pattern_length, 0,
         sizeof(fill->pattern) - pattern_length);

  // fill in target buffer
  rpc_dsp_vaddr_t trgt_dsp_vaddr =
      0; // stays at 0 if no buffer, means to use slot
  if (target_ref->buffer) {
    IREE_RETURN_IF_ERROR(
        buffer_to_dsp_vaddr(target_ref->buffer, &trgt_dsp_vaddr));
  }
  fill->trgt.slot = target_ref->buffer_slot;
  fill->trgt.buffer_dsp_vaddr = trgt_dsp_vaddr;
  fill->trgt.offset = target_ref->offset;
  fill->trgt.length = target_ref->length;

  return iree_ok_status();
  return iree_ok_status();
}
