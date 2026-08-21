// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "command_buffer_types.h"
#include "hexagon/arm_dsp/cmd_buf.h"
#include "iree/base/api.h"
#include "serialize.h"

/**
 * @brief get size of serialized barrier command
 * @param[out] out_cmd_size size of serialized command
 * @return IREE status
 */
iree_status_t
iree_hal_hexagon_cmd_barrier_serialize_prep(iree_host_size_t *out_cmd_size) {
  *out_cmd_size = sizeof(hexagon_rt_arm_dsp_cmd_barrier_t);
  return iree_ok_status();
}

/**
 * @brief serialize barrier command to ARM/DSP data
 * @param[out] cmd_data buffer filled with serialized data
 * @param[in] cmd_size size of buffer for serialized data
 * @return IREE status
 */
iree_status_t
iree_hal_hexagon_cmd_barrier_serialize_exec(uint8_t *cmd_data,
                                            iree_host_size_t cmd_size) {

  uint8_t *ptr = cmd_data;
  uint8_t *endptr = cmd_data + cmd_size;

  // serialize barrier command
  SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_cmd_barrier_t, barrier)
  barrier->base.cmd_type = HEXAGON_RT_ARM_DSP_CMD_BARRIER;

  return iree_ok_status();
}
