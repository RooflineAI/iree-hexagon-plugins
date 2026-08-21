// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_CMD_DISPATCH_SERIALIZE_H_
#define IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_CMD_DISPATCH_SERIALIZE_H_

#include <stdint.h>

#include "command_buffer_types.h"
#include "iree/base/api.h"
#include "rpc_types.h"

/**
 * @brief get size of serialized dispatch command
 * @param[in] constants constants to pass to the kernel, multiple of 4 bytes
 * @param[in] bindings buffer bindings for the dispatch, either fixed buffers or
 *                     references to the binding table passed to command buffer
 *                     execution
 * @param[out] out_cmd_size size of serialized command
 * @return IREE status
 */
iree_status_t iree_hal_hexagon_cmd_dispatch_serialize_prep(
    const iree_const_byte_span_t *constants,
    const iree_hal_buffer_ref_list_t *bindings, iree_host_size_t *out_cmd_size);

/**
 * @brief serialize dispatch command to ARM/DSP data
 * @param[in] executable_handle handle to executable that contains the function
 *            to call
 * @param[in] export_ordinal number of exported function to call from executable
 * @param[in] config workgroup configuration
 * @param[in] constants constants to pass to the kernel, multiple of 4 bytes
 * @param[in] bindings buffer bindings for the dispatch, either fixed buffers or
 *                     references to the binding table passed to command buffer
 *                     execution
 * @param[in] get_buffer_fd function for obtaining the file descriptor for a
 *                          mapped RPCmem (needs to be passed in here for
 *                          decoupling unit test from actual HAL)
 * @param[out] cmd_data buffer filled with serialized data
 * @param[in] cmd_size size of buffer for serialized data
 * @return IREE status
 */
iree_status_t iree_hal_hexagon_cmd_dispatch_serialize_exec(
    rpc_executable_handle_t executable_handle, uint32_t export_ordinal,
    const iree_hal_dispatch_config_t *config,
    const iree_const_byte_span_t *constants,
    const iree_hal_buffer_ref_list_t *bindings,
    iree_hal_hexagon_get_buffer_fd_t get_buffer_fd, uint8_t *cmd_data,
    iree_host_size_t cmd_size);

#endif // IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_CMD_DISPATCH_SERIALIZE_H_
