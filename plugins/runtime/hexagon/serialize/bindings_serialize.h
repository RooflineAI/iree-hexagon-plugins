// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_BINDINGS_SERIALIZE_H_
#define IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_BINDINGS_SERIALIZE_H_

#include <stdint.h>

#include "hexagon/serialize/rpc_types.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

/**
 * @brief get size of serialized binding table, also some checking of input
 * @param[in] binding_table binding table to be serialized
 * @param[out] out_bind_tab_size size of serialized binding table
 * @return IREE status
 */
iree_status_t iree_hal_hexagon_bindings_serialize_prep(
    const iree_hal_buffer_binding_table_t *binding_table,
    iree_host_size_t *out_bind_tab_size);

/**
 * @brief serialize binding table to ARM/DSP data
 * @param[in] binding_table binding table to be serialized
 * @param[in] buffer_to_dsp_vaddr function for obtaining the DSP virtual address
 *                                from a buffer (needs to be passed in here for
 *                                decoupling unit test from actual HAL)
 * @param[out] bind_tab_data buffer filled with serialized data
 * @param[in] bind_tab_size size of serialized binding table
 * @return IREE status
 */
iree_status_t iree_hal_hexagon_bindings_serialize_exec(
    const iree_hal_buffer_binding_table_t *binding_table,
    iree_status_t (*buffer_to_dsp_vaddr)(iree_hal_buffer_t *buffer,
                                         rpc_dsp_vaddr_t *out_dsp_vaddr),
    uint8_t *bind_tab_data, iree_host_size_t bind_tab_size);

#endif // IREE_HAL_DRIVERS_HEXAGON_SERIALIZE_BINDINGS_SERIALIZE_H_
