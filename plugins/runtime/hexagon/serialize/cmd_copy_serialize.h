// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_CMD_COPY_SERIALIZE_H_
#define IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_CMD_COPY_SERIALIZE_H_

#include <stdint.h>

#include "command_buffer_types.h"
#include "iree/base/api.h"

/**
 * @brief get size of serialized copy command
 * @param[out] out_cmd_size size of serialized command
 * @return IREE status
 */
iree_status_t
iree_hal_hexagon_cmd_copy_serialize_prep(iree_host_size_t *out_cmd_size);

/**
 * @brief serialize copy command to ARM/DSP data
 * @param[in] source_ref source buffer for the copy, either a fixed buffers or
 *                       a reference to the binding table passed to command
 *                       buffer execution
 * @param[in] target_ref target buffer for the copy, either a fixed buffers or
 *                       a reference to the binding table passed to command
 *                       buffer execution
 * @param[in] get_buffer_fd function for obtaining the file descriptor for a
 *                          mapped RPCmem (needs to be passed in here for
 *                          decoupling unit test from actual HAL)
 * @param[out] cmd_data buffer filled with serialized data
 * @param[in] cmd_size size of buffer for serialized data
 * @return IREE status
 */
iree_status_t iree_hal_hexagon_cmd_copy_serialize_exec(
    const iree_hal_buffer_ref_t *source_ref,
    const iree_hal_buffer_ref_t *target_ref,
    iree_hal_hexagon_get_buffer_fd_t get_buffer_fd, uint8_t *cmd_data,
    iree_host_size_t cmd_size);

#endif // IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_CMD_COPY_SERIALIZE_H_
