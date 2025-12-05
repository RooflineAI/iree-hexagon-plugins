// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_COMMAND_BUFFER_SERIALIZE_H_
#define IREE_HAL_DRIVERS_HEXAGON_COMMAND_BUFFER_SERIALIZE_H_

#include <stdint.h>

#include "command_buffer_types.h"
#include "hexagon/serialize/rpc_types.h"
#include "iree/base/api.h"

/**
 * @brief get size of serialized command buffer, also check and count entries
 * @param[in] command_buffer command buffer to be serialized
 * @param[out] out_cmd_buf_size size of serialized command buffer
 * @param[out] out_num_entries number of entries in (serialized) command buffer
 * @return IREE status
 */
iree_status_t iree_hal_hexagon_command_buffer_serialize_prep(
    const iree_hal_hexagon_command_buffer_t *command_buffer,
    iree_host_size_t *out_cmd_buf_size, uint32_t *out_num_entries);

/**
 * @brief serialize command buffer to ARM/DSP data
 * @param[in] command_buffer command buffer to be serialized
 * @param[in] buffer_to_dsp_vaddr function for obtaining the DSP virtual address
 *                                from a buffer (needs to be passed in here for
 *                                decoupling unit test from actual HAL)
 * @param[in] num_entries number of entries in (serialized) command buffer
 * @param[out] cmd_buf_data buffer filled with serialized data
 * @param[in] cmd_buf_size size of buffer for serialized data
 * @return IREE status
 */
iree_status_t iree_hal_hexagon_command_buffer_serialize_exec(
    const iree_hal_hexagon_command_buffer_t *command_buffer,
    iree_status_t (*buffer_to_dsp_vaddr)(iree_hal_buffer_t *buffer,
                                         rpc_dsp_vaddr_t *out_dsp_vaddr),
    uint32_t num_entries, uint8_t *cmd_buf_data, iree_host_size_t cmd_buf_size);

#endif // IREE_HAL_DRIVERS_HEXAGON_COMMAND_BUFFER_SERIALIZE_H_
