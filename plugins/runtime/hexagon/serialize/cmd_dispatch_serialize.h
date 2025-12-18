// Copyright 2025 RooflineAI GmbH

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
 * @param[in] buffer_to_dsp_vaddr function for obtaining the DSP virtual address
 *                                from a buffer
 * @param[out] cmd_data buffer filled with serialized data
 * @param[in] cmd_size size of buffer for serialized data
 * @return IREE status
 */
iree_status_t iree_hal_hexagon_cmd_dispatch_serialize_exec(
    rpc_executable_handle_t executable_handle,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t *config,
    const iree_const_byte_span_t *constants,
    const iree_hal_buffer_ref_list_t *bindings,
    iree_hal_hexagon_buffer_to_dsp_vaddr_t buffer_to_dsp_vaddr,
    uint8_t *cmd_data, iree_host_size_t cmd_size);

#endif // IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_CMD_DISPATCH_SERIALIZE_H_
