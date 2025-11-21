// Copyright 2025 RooflineAI GmbH

#include "hexagon/mem_alloc.h"

#include <limits.h>

#include <iree/base/status.h>

#include "AEEStdErr.h"
#include "hexagon/utils.h"
#include "hexagon_dsp.h"
#include "remote.h"
#include "rpcmem.h"

struct iree_hal_hexagon_mem_alloc_s {
  iree_allocator_t host_allocator;
  iree_hal_hexagon_domain_id_t domain_id;
  rpc_session_handle_t rpc_session_handle;
  iree_hal_hexagon_mem_kind_t kind;
  iree_device_size_t size;
  unsigned int ref_count;
  union {
    void *host; ///< for _KIND_HOST
    struct {
      void *host;
      int fd;
      int64_t dsp_vaddr;
    } rpcmem;           ///< for _KIND_RPCMEM
    int64_t device_hap; ///< for _KIND_DEVICE_HAP
    void *impl_ptr; ///< take whatever is stored in this union for "accounting"
                    ///< (see comment in _allocate_buffer() in allocator.c)
  } ptr;
};

iree_status_t iree_hal_hexagon_mem_alloc_create(
    iree_allocator_t host_allocator, iree_hal_hexagon_domain_id_t domain_id,
    rpc_session_handle_t rpc_session_handle, iree_hal_hexagon_mem_kind_t kind,
    iree_device_size_t size, iree_hal_hexagon_mem_alloc_t **out_alloc) {
  IREE_ASSERT_ARGUMENT(out_alloc);
  *out_alloc = NULL;

  iree_hal_hexagon_mem_alloc_t *alloc = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*alloc), (void **)&alloc));
  alloc->host_allocator = host_allocator;
  alloc->domain_id = domain_id;
  alloc->rpc_session_handle = rpc_session_handle;
  alloc->kind = kind;
  alloc->size = size;
  alloc->ref_count = 1;

  iree_status_t status = iree_make_status(
      IREE_STATUS_UNIMPLEMENTED, "memory kind %d not implemented", kind);
  switch (kind) {
  case IREE_HAL_HEXAGON_MEM_KIND_HOST: {
    void *buf = NULL;
    status = iree_allocator_malloc(host_allocator, size, &buf);
    if (!iree_status_is_ok(status)) {
      break;
    }
    alloc->ptr.host = buf;
    status = iree_ok_status();
    break;
  }

  case IREE_HAL_HEXAGON_MEM_KIND_RPCMEM: {
    // According to the Hexagon SDK examples, allocating memory shared among ARM
    // and DSP needs to execute those steps:
    // 1) host_ptr = rpcmem_alloc()
    // 2) fd = rpcmem_to_fd(host_ptr)
    // 3) fastrpc_mmap(host_ptr)
    // 4) fd ---> DSP
    // 5) on DSP: HAP_mmap_get(fd) (see hexagon_dsp_buffer_rpcmem_mmap())
    // There is currently no complete understanding why the SDK is built this
    // way. This is following the examples in the SDK.
    if (size > INT_MAX) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "cannot allocate RPC memory bigger than INT_MAX");
      break;
    }
    void *ptr_host =
        rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)size);
    if (!ptr_host) {
      status =
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "out of RPC memory");
      break;
    }
    int fd = rpcmem_to_fd(ptr_host);
    if (fd == -1) {
      rpcmem_free(ptr_host);
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "cannot get file descriptor for RPC memory");
      break;
    }
    int dsp_err =
        fastrpc_mmap(domain_id, fd, ptr_host, 0, size, FASTRPC_MAP_FD);
    if (dsp_err != AEE_SUCCESS) {
      rpcmem_free(ptr_host);
      status = IREE_HAL_HEXAGON_MAKE_STATUS_FROM_DSP_ERR(
          dsp_err, "cannot map RPC memory on host side");
      break;
    }
    long long dsp_vaddr = 0;
    dsp_err =
        hexagon_dsp_buffer_rpcmem_mmap(rpc_session_handle, fd, &dsp_vaddr);
    if (dsp_err != AEE_SUCCESS) {
      fastrpc_munmap(domain_id, fd, NULL, 0);
      rpcmem_free(ptr_host);
      status = IREE_HAL_HEXAGON_MAKE_STATUS_FROM_DSP_ERR(
          dsp_err, "cannot map RPC memory on DSP side");
      break;
    }
    alloc->ptr.rpcmem.host = ptr_host;
    alloc->ptr.rpcmem.fd = fd;
    alloc->ptr.rpcmem.dsp_vaddr = dsp_vaddr;
    status = iree_ok_status();
    break;
  }

  case IREE_HAL_HEXAGON_MEM_KIND_DEVICE_HAP: {
    if (size > UINT_MAX) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "cannot allocate DSP memory bigger than UINT_MAX");
      break;
    }
    long long dsp_vaddr = 0;
    int dsp_err =
        hexagon_dsp_buffer_hap_malloc(rpc_session_handle, size, &dsp_vaddr);
    if (dsp_err != AEE_SUCCESS) {
      status = IREE_HAL_HEXAGON_MAKE_STATUS_FROM_DSP_ERR(dsp_err,
                                                         "out of DSP memory");
      break;
    }
    alloc->ptr.device_hap = dsp_vaddr;
    status = iree_ok_status();
    break;
  }
  }

  if (iree_status_is_ok(status)) {
    *out_alloc = alloc;
  } else {
    iree_allocator_free(host_allocator, alloc);
  }
  return status;
}

void iree_hal_hexagon_mem_alloc_retain(iree_hal_hexagon_mem_alloc_t *alloc) {
  ++alloc->ref_count;
}

void iree_hal_hexagon_mem_alloc_release(iree_hal_hexagon_mem_alloc_t *alloc) {
  --alloc->ref_count;
  if (alloc->ref_count > 0) {
    return;
  }

  switch (alloc->kind) {
  case IREE_HAL_HEXAGON_MEM_KIND_HOST:
    iree_allocator_free(alloc->host_allocator, alloc->ptr.host);
    alloc->ptr.host = NULL;
    break;

  case IREE_HAL_HEXAGON_MEM_KIND_RPCMEM:
    hexagon_dsp_buffer_rpcmem_munmap(alloc->rpc_session_handle,
                                     alloc->ptr.rpcmem.fd);
    fastrpc_munmap(alloc->domain_id, alloc->ptr.rpcmem.fd, NULL, 0);
    rpcmem_free(alloc->ptr.rpcmem.host);
    alloc->ptr.rpcmem.host = NULL;
    alloc->ptr.rpcmem.fd = -1;
    alloc->ptr.rpcmem.dsp_vaddr = 0;
    break;

  case IREE_HAL_HEXAGON_MEM_KIND_DEVICE_HAP:
    hexagon_dsp_buffer_hap_free(alloc->rpc_session_handle,
                                alloc->ptr.device_hap);
    alloc->ptr.device_hap = 0;
    break;
  }

  iree_allocator_free(alloc->host_allocator, alloc);
}

iree_status_t
iree_hal_hexagon_mem_alloc_get_host_span(iree_hal_hexagon_mem_alloc_t *alloc,
                                         iree_byte_span_t *out_host_span) {
  IREE_ASSERT_ARGUMENT(alloc);
  IREE_ASSERT_ARGUMENT(out_host_span);
  out_host_span->data = NULL;
  out_host_span->data_length = 0;

  switch (alloc->kind) {
  case IREE_HAL_HEXAGON_MEM_KIND_HOST:
    out_host_span->data = alloc->ptr.host;
    out_host_span->data_length = alloc->size;
    return iree_ok_status();
  case IREE_HAL_HEXAGON_MEM_KIND_RPCMEM:
    out_host_span->data = alloc->ptr.rpcmem.host;
    out_host_span->data_length = alloc->size;
    return iree_ok_status();
  case IREE_HAL_HEXAGON_MEM_KIND_DEVICE_HAP:
    // no host pointer available
    break;
  }

  return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                          "Hexagon memory kind %d does not have a host pointer",
                          alloc->kind);
}

void *iree_hal_hexagon_mem_alloc_impl_ptr(iree_hal_hexagon_mem_alloc_t *alloc) {
  return alloc->ptr.impl_ptr;
}
