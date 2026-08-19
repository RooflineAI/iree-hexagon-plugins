// Copyright 2025 RooflineAI GmbH

#include "hexagon/dsp/rt/vtcm_pool.h"

#include <stdlib.h>
#include <unordered_map>

#include "AEEStdErr.h"

#include "HexagonAPI.h"
#include "VTCMPool.h"

/* hexagon-mlir's VtcmPool needs the size of the memory block in the free
 * call. Store sizes from malloc call in a map to be able to look those up
 * in the free call. FIXME: rework VtcmPool ROO-1441
 */
static std::unordered_map<void *, size_t> vtcm_pool_sizes;

void *hexagon_dsp_vtcm_pool_allocate(size_t nbytes) {
  VtcmPool *vtcm_pool = HexagonAPI::Global()->getVtcmPool();
  if (!vtcm_pool) {
    return NULL;
  }
  // minimum VTCM allocation is 128 bytes
  if (nbytes < 128) {
    nbytes = 128;
  }
  void *ptr = vtcm_pool->Allocate(nbytes);
  if (ptr) {
    vtcm_pool_sizes[ptr] = nbytes;
  }
  return ptr;
}

void hexagon_dsp_vtcm_pool_free(void *ptr) {
  VtcmPool *vtcm_pool = HexagonAPI::Global()->getVtcmPool();
  if (!vtcm_pool) {
    return;
  }
  const auto sizes_iter = vtcm_pool_sizes.find(ptr);
  size_t nbytes = 0;
  if (sizes_iter != vtcm_pool_sizes.end()) {
    nbytes = sizes_iter->second;
    vtcm_pool_sizes.erase(sizes_iter);
  }
  vtcm_pool->Free(ptr, nbytes);
}
