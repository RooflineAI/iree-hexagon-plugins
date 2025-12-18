// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_CMD_FILL_SERIALIZE_H_
#define IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_CMD_FILL_SERIALIZE_H_

#include <stdint.h>

#include "command_buffer_types.h"
#include "iree/base/api.h"

/**
 * @brief get size of serialized fill command
 * @param[out] out_cmd_size size of serialized command
 * @return IREE status
 */
iree_status_t
iree_hal_hexagon_cmd_fill_serialize_prep(iree_host_size_t *out_cmd_size);

/**
 * @brief serialize fill command to ARM/DSP data
 * @param[in] pattern_length pattern length, may be 1/2/4 according to comments
 *                           in IREE
 * @param[in] pattern pattern data, max length in IREE is 4
 * @param[in] target_ref target buffer for the fill, either a fixed buffers or
 *                       a reference to the binding table passed to command
 *                       buffer execution
 * @param[in] buffer_to_dsp_vaddr function for obtaining the DSP virtual address
 *                                from a buffer
 * @param[out] cmd_data buffer filled with serialized data
 * @param[in] cmd_size size of buffer for serialized data
 * @return IREE status
 */
iree_status_t iree_hal_hexagon_cmd_fill_serialize_exec(
    uint8_t pattern_length, const uint8_t *pattern,
    const iree_hal_buffer_ref_t *target_ref,
    iree_hal_hexagon_buffer_to_dsp_vaddr_t buffer_to_dsp_vaddr,
    uint8_t *cmd_data, iree_host_size_t cmd_size);

#endif // IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_CMD_FILL_SERIALIZE_H_
