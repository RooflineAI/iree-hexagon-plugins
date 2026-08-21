// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
    iree_hal_hexagon_get_buffer_fd_t get_buffer_fd, uint8_t *cmd_data,
    iree_host_size_t cmd_size) {

  uint8_t *ptr = cmd_data;
  uint8_t *endptr = cmd_data + cmd_size;

  // normalize target buffer reference
  iree_hal_buffer_ref_t normalized_target_ref = *target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_ref_normalize(&normalized_target_ref));

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
  int fd = -1; // stays at -1 if no buffer, means to use slot
  if (normalized_target_ref.buffer) {
    IREE_RETURN_IF_ERROR(get_buffer_fd(normalized_target_ref.buffer, &fd));
  }
  fill->trgt.slot = normalized_target_ref.buffer_slot;
  fill->trgt.fd = fd;
  fill->trgt.offset = normalized_target_ref.offset;
  fill->trgt.length = target_ref->length;

  return iree_ok_status();
}
