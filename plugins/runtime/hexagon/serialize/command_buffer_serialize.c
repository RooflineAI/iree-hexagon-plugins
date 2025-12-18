// Copyright 2025 RooflineAI GmbH

#include "command_buffer_serialize.h"

#include <stdint.h>

#include "command_buffer_types.h"
#include "hexagon/arm_dsp/cmd_buf.h"
#include "hexagon/serialize/rpc_types.h"
#include "iree/base/api.h"
#include "rpc_types.h"
#include "serialize.h"

iree_status_t iree_hal_hexagon_command_buffer_serialize_prep(
    const iree_hal_hexagon_command_buffer_t *command_buffer,
    iree_host_size_t *out_cmd_buf_size, uint32_t *out_num_entries) {
  // start with size of command buffer header
  iree_host_size_t cmd_buf_size = sizeof(hexagon_rt_arm_dsp_cmd_buf_t);

  // add size of commands in all entries,
  // also count number of entries and check for overflow
  uint32_t num_entries = 0;
  for (const iree_hal_hexagon_command_buffer_entry_t *entry =
           command_buffer->first_entry;
       entry; entry = entry->next) {
    iree_host_size_t prev_size = cmd_buf_size;
    cmd_buf_size += entry->size;
    ++num_entries;
    if (cmd_buf_size < prev_size || num_entries == 0 /* overflow */) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "too many entries in command buffer");
    }
  }

  // return data size and number of entries
  *out_cmd_buf_size = cmd_buf_size;
  *out_num_entries = num_entries;
  return iree_ok_status();
}

iree_status_t iree_hal_hexagon_command_buffer_serialize_exec(
    const iree_hal_hexagon_command_buffer_t *command_buffer,
    iree_status_t (*buffer_to_dsp_vaddr)(iree_hal_buffer_t *buffer,
                                         rpc_dsp_vaddr_t *out_dsp_vaddr),
    uint32_t num_entries, uint8_t *cmd_buf_data,
    iree_host_size_t cmd_buf_size) {

  uint8_t *ptr = cmd_buf_data;
  uint8_t *endptr = cmd_buf_data + cmd_buf_size;

  // serialize command buffer header
  SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_cmd_buf_t, cmd_buf)
  cmd_buf->num_entries = num_entries;

  // concatenate pre-serialized commands from all entries
  for (const iree_hal_hexagon_command_buffer_entry_t *entry =
           command_buffer->first_entry;
       entry; entry = entry->next) {
    // command data starts after end of entry struct
    uint8_t *cmd_data = (uint8_t *)entry + sizeof(*entry);
    SERIALIZE_COPY_TO(ptr, endptr, cmd_data, entry->size);
  }

  return iree_ok_status();
}
