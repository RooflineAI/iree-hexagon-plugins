// Copyright 2025 RooflineAI GmbH

#include "cmd_dispatch_serialize.h"

#include <stdint.h>

#include "command_buffer_types.h"
#include "hexagon/arm_dsp/cmd_buf.h"
#include "iree/base/api.h"
#include "rpc_types.h"
#include "serialize.h"

static iree_status_t iree_hal_hexagon_cmd_dispatch_get_constant_count(
    const iree_const_byte_span_t *constants, uint16_t *out_constant_count) {
  if ((constants->data_length & (sizeof(uint32_t) - 1)) != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "constant data that is not a multiple of %u bytes "
                            "is not supported on Hexagon",
                            (unsigned int)sizeof(uint32_t));
  }
  iree_host_size_t constant_count = constants->data_length / sizeof(uint32_t);
  if (constant_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many constants %" PRIu64 "u for Hexagon",
                            constant_count);
  }
  *out_constant_count = constant_count;
  return iree_ok_status();
}

iree_status_t iree_hal_hexagon_cmd_dispatch_serialize_prep(
    const iree_const_byte_span_t *constants,
    const iree_hal_buffer_ref_list_t *bindings,
    iree_host_size_t *out_cmd_size) {
  uint16_t constant_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hexagon_cmd_dispatch_get_constant_count(
      constants, &constant_count));
  *out_cmd_size = sizeof(hexagon_rt_arm_dsp_cmd_dispatch_t) +
                  constant_count * sizeof(hexagon_rt_arm_dsp_con_t) +
                  bindings->count * sizeof(hexagon_rt_arm_dsp_buf_ref_t);
  return iree_ok_status();
}

iree_status_t iree_hal_hexagon_cmd_dispatch_serialize_exec(
    rpc_executable_handle_t executable_handle,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t *config,
    const iree_const_byte_span_t *constants,
    const iree_hal_buffer_ref_list_t *bindings,
    iree_hal_hexagon_get_buffer_fd_t get_buffer_fd, uint8_t *cmd_data,
    iree_host_size_t cmd_size) {

  // workgroup size Z uses a short type in IREE dispatches, check for overflow
  if (config->workgroup_size[2] > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "workgroup_size[2] (z) of %" PRIu32
                            "d is too large",
                            config->workgroup_size[2]);
  }
  if (config->workgroup_count_ref.length != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "indirect workgroup counts are not supported on Hexagon");
  }
  if (config->dynamic_workgroup_local_memory != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "dynamic workgroup local memory is not supported on Hexagon");
  }

  uint16_t constant_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hexagon_cmd_dispatch_get_constant_count(
      constants, &constant_count));

  uint8_t *ptr = cmd_data;
  uint8_t *endptr = cmd_data + cmd_size;

  // serialize dispatch command
  SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_cmd_dispatch_t, dispatch)
  dispatch->base.cmd_type = HEXAGON_RT_ARM_DSP_CMD_DISPATCH;
  dispatch->executable_handle = executable_handle;
  dispatch->export_ordinal = export_ordinal;
  // workgroup size 0 means default workgroup size, it is 1 here
  dispatch->workgroup_size_x =
      config->workgroup_size[0] ? config->workgroup_size[0] : 1;
  dispatch->workgroup_size_y =
      config->workgroup_size[1] ? config->workgroup_size[1] : 1;
  dispatch->workgroup_size_z =
      config->workgroup_size[2] ? config->workgroup_size[2] : 1;
  dispatch->workgroup_count_x = config->workgroup_count[0];
  dispatch->workgroup_count_y = config->workgroup_count[1];
  dispatch->workgroup_count_z = config->workgroup_count[2];
  dispatch->constant_count = constant_count;
  dispatch->num_bindings = bindings->count;

  // serialize (copy) constants
  SERIALIZE_COPY_TO(ptr, endptr, constants->data,
                    constant_count * sizeof(hexagon_rt_arm_dsp_con_t));

  // serialize bindings
  for (uint32_t b = 0; b < bindings->count; ++b) {
    const iree_hal_buffer_ref_t *binding = &bindings->values[b];
    int fd = -1; // stays at -1 if no buffer, means to use slot
    if (binding->buffer) {
      IREE_RETURN_IF_ERROR(get_buffer_fd(binding->buffer, &fd));
    }
    SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_buf_ref_t, bndg)
    bndg->slot = binding->buffer_slot;
    bndg->fd = fd;
    bndg->offset = binding->offset;
    bndg->length = binding->length;
  }

  return iree_ok_status();
}
