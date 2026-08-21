// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bindings_serialize.h"

#include <stdint.h>

#include "hexagon/arm_dsp/bindings.h"
#include "hexagon/serialize/command_buffer_types.h"
#include "iree/base/api.h"
#include "rpc_types.h"
#include "serialize.h"

iree_status_t iree_hal_hexagon_bindings_serialize_prep(
    const iree_hal_buffer_binding_table_t *binding_table,
    iree_host_size_t *out_bind_tab_size) {
  if (binding_table->count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many bindings in table");
  }
  *out_bind_tab_size =
      sizeof(hexagon_rt_arm_dsp_binding_tab_t) +
      binding_table->count * sizeof(hexagon_rt_arm_dsp_binding_t);
  return iree_ok_status();
}

iree_status_t iree_hal_hexagon_bindings_serialize_exec(
    const iree_hal_buffer_binding_table_t *binding_table,
    iree_hal_hexagon_get_buffer_fd_t get_buffer_fd, uint8_t *bind_tab_data,
    iree_host_size_t bind_tab_size) {
  uint8_t *ptr = bind_tab_data;
  uint8_t *endptr = bind_tab_data + bind_tab_size;

  // serialize binding table header
  SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_binding_tab_t, bind_tab)
  bind_tab->num_entries = binding_table->count;

  // serialize all binding entries
  for (iree_host_size_t i = 0; i < binding_table->count; ++i) {
    iree_hal_buffer_binding_t normalized_binding = binding_table->bindings[i];
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_binding_normalize(&normalized_binding));
    int fd = -1;
    IREE_RETURN_IF_ERROR(get_buffer_fd(normalized_binding.buffer, &fd));
    SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_binding_t, bndg)
    bndg->fd = fd;
    bndg->offset = normalized_binding.offset;
    bndg->length = normalized_binding.length;
  }

  return iree_ok_status();
}
