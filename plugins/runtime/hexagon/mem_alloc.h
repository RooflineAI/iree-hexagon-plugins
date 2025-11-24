// Copyright 2025 RooflineAI GmbH

#ifndef IREE_HAL_DRIVERS_HEXAGON_MEM_ALLOC_H_
#define IREE_HAL_DRIVERS_HEXAGON_MEM_ALLOC_H_

#include <iree/base/config.h>

#include "hexagon/api.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "rpc_types.h"

typedef enum iree_hal_hexagon_mem_kind_e {
  IREE_HAL_HEXAGON_MEM_KIND_HOST,
  IREE_HAL_HEXAGON_MEM_KIND_RPCMEM,
  IREE_HAL_HEXAGON_MEM_KIND_DEVICE_HAP,
} iree_hal_hexagon_mem_kind_t;

/// A {Qualcomm Hexagon} memory allocation.
typedef struct iree_hal_hexagon_mem_alloc_s iree_hal_hexagon_mem_alloc_t;

/// Create a {Qualcomm Hexagon} memory allocation.
iree_status_t iree_hal_hexagon_mem_alloc_create(
    iree_allocator_t host_allocator, iree_hal_hexagon_domain_id_t domain_id,
    rpc_session_handle_t rpc_session_handle, iree_hal_hexagon_mem_kind_t kind,
    iree_device_size_t size, iree_hal_hexagon_mem_alloc_t **out_alloc);

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

#endif // IREE_HAL_DRIVERS_HEXAGON_MEM_ALLOC_H_
