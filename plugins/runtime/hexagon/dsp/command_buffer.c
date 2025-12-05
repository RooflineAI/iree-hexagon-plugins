// Copyright 2025 RooflineAI GmbH

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

#include "AEEStdErr.h"
#include "HAP_mem.h"
#include "hexagon_dsp.h"

/// data about a command buffer
typedef struct hexagon_dsp_command_buffer_s {
  /// handle of owning RPC session
  remote_handle64 rpc_handle;
  uint8_t *cmd_buf_data; ///< contains hexagon_rt_arm_dsp_cmd_buf
  size_t cmd_buf_size;
} hexagon_dsp_command_buffer_t;

/**
 * @brief Create a command buffer (called when ARM host side finalizes command
 * buffer)
 * @param[in] rpc_handle handle of DSP RPC session
 * @param[in] cmd_buf_data command buffer data of type
 * hexagon_rt_arm_dsp_cmd_buf_t
 * @param[in] cmd_buf_size size of command buffer data
 * @param[out] command_buffer_handle handle of the loaded executable
 * @retval AEE_SUCCESS for success
 */
int hexagon_dsp_command_buffer_create(remote_handle64 rpc_handle,
                                      const uint8 *cmd_buf_data,
                                      int cmd_buf_size,
                                      int64 *command_buffer_handle) {
  // allocate internal data structure used to mange command buffer
  hexagon_dsp_command_buffer_t *command_buffer = NULL;
  int err = HAP_malloc(sizeof(hexagon_dsp_command_buffer_t),
                       (void **)&command_buffer);
  if (err != AEE_SUCCESS) {
    return err;
  }
  if (!command_buffer) {
    return AEE_ENOMEMORY;
  }
  command_buffer->rpc_handle = rpc_handle;

  // Store local copy of command buffer data.
  // The buffer cmd_buf_data lives only (on DSP side) during the duration of
  // the PRC call. The data needs to be copied in order to retain it.
  command_buffer->cmd_buf_size = cmd_buf_size;
  command_buffer->cmd_buf_data = NULL;
  err = HAP_malloc(cmd_buf_size, (void **)&command_buffer->cmd_buf_data);
  if (err != AEE_SUCCESS) {
    free(command_buffer);
    return err;
  }
  if (!command_buffer->cmd_buf_data) {
    free(command_buffer);
    return AEE_ENOMEMORY;
  }
  memcpy(command_buffer->cmd_buf_data, cmd_buf_data, cmd_buf_size);

  // return pointer to internal data structure as handle
  *command_buffer_handle = (int64)command_buffer;
  return AEE_SUCCESS;
}

/**
 * @brief Destroy command buffer.
 * @param[in] rpc_handle handle of DSP RPC session
 * @param[in] command_buffer_handle handle of the command buffer
 * @retval AEE_SUCCESS for success, should always succeed
 */
int hexagon_dsp_command_buffer_destroy(remote_handle64 rpc_handle,
                                       int64 command_buffer_handle) {
  hexagon_dsp_command_buffer_t *command_buffer =
      (hexagon_dsp_command_buffer_t *)command_buffer_handle;
  // free command buffer data
  HAP_free(command_buffer->cmd_buf_data);
  // free management data structure
  HAP_free(command_buffer);
  return AEE_SUCCESS;
}
