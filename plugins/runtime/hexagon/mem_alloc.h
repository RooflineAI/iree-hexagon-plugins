// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HEXAGON_MEM_ALLOC_H_
#define IREE_HAL_DRIVERS_HEXAGON_MEM_ALLOC_H_

#include <iree/base/config.h>

#include "hexagon/api.h"
#include "hexagon/serialize/rpc_types.h"
#include "iree/base/api.h"
#include "iree/base/status.h"
#include "rpc_types.h"

/** alignment (in bytes) of all Hexagon memory allocations (e.g. used for
 * buffers passed to dispatches), supposed to match size of vector register */
#define IREE_HAL_HEXAGON_MEM_ALLOC_ALIGN (128)

/// A {Qualcomm Hexagon} memory allocation.
typedef struct iree_hal_hexagon_mem_alloc_s iree_hal_hexagon_mem_alloc_t;

/// Create a {Qualcomm Hexagon} memory allocation. It is registered with the DSP
/// right away (without mapping it), the DSP maps it itself when a command uses
/// it.
iree_status_t iree_hal_hexagon_mem_alloc_create(
    iree_allocator_t host_allocator, iree_hal_hexagon_domain_id_t domain_id,
    rpc_session_handle_t rpc_session_handle, iree_device_size_t size,
    iree_hal_hexagon_mem_alloc_t **out_alloc);

/// Retain a {Qualcomm Hexagon} memory allocation - increase ref counter.
void iree_hal_hexagon_mem_alloc_retain(iree_hal_hexagon_mem_alloc_t *alloc);

/// Release a {Qualcomm Hexagon} memory allocation - decrease ref counter and
/// free it if counter reaches zero.
void iree_hal_hexagon_mem_alloc_release(iree_hal_hexagon_mem_alloc_t *alloc);

/// Return the impl_ptr from inside the memory allocation for accounting
/// purposes. It just needs to be stable (not change for the same alloc) and be
/// different for each alloc (see comment in
/// iree_hal_hexagon_allocator_allocate_buffer() in allocator.c).
void *iree_hal_hexagon_mem_alloc_impl_ptr(iree_hal_hexagon_mem_alloc_t *alloc);

/// Return the host span to the memory allocation if possible.
iree_status_t
iree_hal_hexagon_mem_alloc_get_host_span(iree_hal_hexagon_mem_alloc_t *alloc,
                                         iree_byte_span_t *out_host_span);

/// Return the file descriptor of the memory allocation and the size of the
/// whole RPC memory allocation.
iree_status_t
iree_hal_hexagon_mem_alloc_get_fd(iree_hal_hexagon_mem_alloc_t *alloc,
                                  int *out_fd, uint64_t *out_alloc_size);

#endif // IREE_HAL_DRIVERS_HEXAGON_MEM_ALLOC_H_
