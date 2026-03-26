// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_CMD_BARRIER_SERIALIZE_H_
#define IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_CMD_BARRIER_SERIALIZE_H_

#include <stdint.h>

#include "command_buffer_types.h"
#include "iree/base/api.h"

/**
 * @brief get size of serialized barrier command
 * @param[out] out_cmd_size size of serialized command
 * @return IREE status
 */
iree_status_t
iree_hal_hexagon_cmd_barrier_serialize_prep(iree_host_size_t *out_cmd_size);

/**
 * @brief serialize barrier command to ARM/DSP data
 * @param[out] cmd_data buffer filled with serialized data
 * @param[in] cmd_size size of buffer for serialized data
 * @return IREE status
 */
iree_status_t
iree_hal_hexagon_cmd_barrier_serialize_exec(uint8_t *cmd_data,
                                            iree_host_size_t cmd_size);

#endif // IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_CMD_BARRIER_SERIALIZE_H_
