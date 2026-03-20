// Copyright 2025 RooflineAI GmbH

#include "hexagon/mem_alloc.h"

#include <limits.h>

#include <iree/base/status.h>

#include "AEEStdErr.h"
#include "hexagon/serialize/rpc_types.h"
#include "hexagon/utils.h"
#include "hexagon_dsp.h"
#include "remote.h"
#include "rpcmem.h"

struct iree_hal_hexagon_mem_alloc_s {
  iree_allocator_t host_allocator;
  iree_hal_hexagon_domain_id_t domain_id;
  rpc_session_handle_t rpc_session_handle;
  iree_device_size_t size;
  unsigned int ref_count; // reference count for allocation, for "host"
  unsigned int map_count; // reference count for mapping, for "fd"
  void *ptr_host;         // host pointer as returned by rpcmem_alloc()
  int fd; // file descriptor returned by rpcmem_to_fd(), valid if map_count > 0
};

iree_status_t iree_hal_hexagon_mem_alloc_create(
    iree_allocator_t host_allocator, iree_hal_hexagon_domain_id_t domain_id,
    rpc_session_handle_t rpc_session_handle, iree_device_size_t size,
    iree_hal_hexagon_mem_alloc_t **out_alloc) {
  IREE_ASSERT_ARGUMENT(out_alloc);
  *out_alloc = NULL;

  iree_hal_hexagon_mem_alloc_t *alloc = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*alloc), (void **)&alloc));
  alloc->host_allocator = host_allocator;
  alloc->domain_id = domain_id;
  alloc->rpc_session_handle = rpc_session_handle;
  alloc->size = size;
  alloc->ref_count = 1;
  alloc->map_count = 0;
  alloc->ptr_host = NULL;
  alloc->fd = -1;

  // According to the Hexagon SDK examples, allocating memory shared among ARM
  // host and DSP needs to execute those steps:
  // 1) host_ptr = rpcmem_alloc()
  // 2) fd = rpcmem_to_fd(host_ptr)
  // 3) fastrpc_mmap(host_ptr)
  // 4) fd ---> DSP
  // 5) on DSP: HAP_mmap_get(fd)
  // There is currently no complete understanding why the SDK is built this
  // way. This is following the examples in the SDK.
  // Using RPCMEM_DEFAULT_FLAGS results in cached and coherent memory on the
  // ARM side according to
  // Hexagon_SDK/6.3.0.0/docs/software/os/os_support_dsp.html#cache-management
  // During allocation, only step (1) is done. Mapping to DSP happens later,
  // see _map() and _unmap() functions.
  if (size > INT_MAX) {
    iree_allocator_free(host_allocator, alloc);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cannot allocate RPC memory bigger than INT_MAX");
  }
  void *ptr_host =
      rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)size);
  if (!ptr_host) {
    iree_allocator_free(host_allocator, alloc);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "out of RPC memory");
  }
  alloc->ptr_host = ptr_host;

  *out_alloc = alloc;
  return iree_ok_status();
}

void iree_hal_hexagon_mem_alloc_retain(iree_hal_hexagon_mem_alloc_t *alloc) {
  ++alloc->ref_count;
}

void iree_hal_hexagon_mem_alloc_release(iree_hal_hexagon_mem_alloc_t *alloc) {
  --alloc->ref_count;
  if (alloc->ref_count > 0) {
    return;
  }

  while (alloc->map_count > 0) {
    if (!iree_status_is_ok(iree_hal_hexagon_mem_alloc_unmap(alloc))) {
      break; // unmapping failed, it does not make sense to retry
    }
  }

  rpcmem_free(alloc->ptr_host);
  alloc->ptr_host = NULL;
  alloc->fd = -1;

  iree_allocator_free(alloc->host_allocator, alloc);
}

void *iree_hal_hexagon_mem_alloc_impl_ptr(iree_hal_hexagon_mem_alloc_t *alloc) {
  // We need to return any pointer that can be used for identifying the memory
  // allocation. The host pointer works fine for this.
  return alloc->ptr_host;
}

iree_status_t
iree_hal_hexagon_mem_alloc_get_host_span(iree_hal_hexagon_mem_alloc_t *alloc,
                                         iree_byte_span_t *out_host_span) {
  IREE_ASSERT_ARGUMENT(alloc);
  IREE_ASSERT_ARGUMENT(out_host_span);
  out_host_span->data = alloc->ptr_host;
  out_host_span->data_length = alloc->size;
  return iree_ok_status();
}

iree_status_t
iree_hal_hexagon_mem_alloc_map(iree_hal_hexagon_mem_alloc_t *alloc,
                               int *out_fd) {
  IREE_ASSERT_ARGUMENT(alloc);
  IREE_ASSERT_ARGUMENT(out_fd);

  if (alloc->map_count == 0) {
    int fd = rpcmem_to_fd(alloc->ptr_host);
    if (fd == -1) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "cannot get file descriptor for RPC memory");
    }
    int dsp_err = fastrpc_mmap(alloc->domain_id, fd, alloc->ptr_host, 0,
                               alloc->size, FASTRPC_MAP_FD);
    if (dsp_err != AEE_SUCCESS) {
      return IREE_HAL_HEXAGON_MAKE_STATUS_FROM_DSP_ERR(
          dsp_err, "cannot map RPC memory to DSP");
    }
    alloc->fd = fd;
  }
  alloc->map_count++;

  *out_fd = alloc->fd;
  return iree_ok_status();
}

iree_status_t
iree_hal_hexagon_mem_alloc_get_fd(iree_hal_hexagon_mem_alloc_t *alloc,
                                  int *out_fd) {
  IREE_ASSERT_ARGUMENT(alloc);
  IREE_ASSERT_ARGUMENT(out_fd);

  if (alloc->map_count == 0) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "cannot get file descriptor for RPC memory not mapped to DSP");
  }

  *out_fd = alloc->fd;
  return iree_ok_status();
}

iree_status_t
iree_hal_hexagon_mem_alloc_unmap(iree_hal_hexagon_mem_alloc_t *alloc) {
  IREE_ASSERT_ARGUMENT(alloc);

  if (alloc->map_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "cannot unmap RPC memory that is not mapped to DSP");
  }

  if (alloc->map_count == 1) {
    int dsp_err = fastrpc_munmap(alloc->domain_id, alloc->fd, NULL, 0);
    if (dsp_err != AEE_SUCCESS) {
      return IREE_HAL_HEXAGON_MAKE_STATUS_FROM_DSP_ERR(
          dsp_err, "cannot unmap RPC memory from DSP");
    }
    alloc->fd = -1;
  }
  alloc->map_count--;

  return iree_ok_status();
}
