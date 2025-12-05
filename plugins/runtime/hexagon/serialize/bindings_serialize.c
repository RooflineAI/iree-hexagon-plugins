// Copyright 2025 RooflineAI GmbH

#include "bindings_serialize.h"

#include <stdint.h>

#include "hexagon/arm_dsp/bindings.h"
#include "hexagon/serialize/rpc_types.h"
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
    iree_status_t (*buffer_to_dsp_vaddr)(iree_hal_buffer_t *buffer,
                                         rpc_dsp_vaddr_t *out_dsp_vaddr),
    uint8_t *bind_tab_data, iree_host_size_t bind_tab_size) {
  uint8_t *ptr = bind_tab_data;
  uint8_t *endptr = bind_tab_data + bind_tab_size;

  // serialize binding table header
  SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_binding_tab_t, bind_tab)
  bind_tab->num_entries = binding_table->count;

  // serialize all binding entries
  for (iree_host_size_t i = 0; i < binding_table->count; ++i) {
    const iree_hal_buffer_binding_t *binding = &binding_table->bindings[i];
    rpc_dsp_vaddr_t dsp_vaddr = 0;
    IREE_RETURN_IF_ERROR(buffer_to_dsp_vaddr(binding->buffer, &dsp_vaddr));
    SERIALIZE_TO(ptr, endptr, hexagon_rt_arm_dsp_binding_t, bndg)
    bndg->buffer_dsp_vaddr = dsp_vaddr;
    bndg->offset = binding->offset;
    bndg->length = binding->length;
  }

  return iree_ok_status();
}
