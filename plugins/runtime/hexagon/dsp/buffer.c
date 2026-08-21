// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>
#include <stdlib.h>

#include "AEEStdErr.h"
#include "HAP_mem.h"
#include "hexagon_dsp.h"

/**
 * @brief Map shared RPC memory to DSP virtual address.
 * @param[in] rpc_handle handle of DSP RPC session
 * @param[in] fd file descriptor of shared RPC memory
 * @param[out] vaddr virtual address on DSP
 * @retval AEE_SUCCESS for success
 */
int hexagon_dsp_buffer_rpcmem_mmap(remote_handle64 rpc_handle, int fd,
                                   int64 *vaddr) {
  (void)rpc_handle;
  void *virt_addr = NULL;
  uint64 phys_addr = 0;
  int err = HAP_mmap_get(fd, &virt_addr, &phys_addr);
  if (err == AEE_SUCCESS) {
    *vaddr = (int64)virt_addr;
  }
  return err;
}

/**
 * @brief Unmap shared DSP memory.
 * @param[in] rpc_handle handle of DSP RPC session
 * @param[in] fd file descriptor of shared RPC memory
 * @retval AEE_SUCCESS for success
 */
int hexagon_dsp_buffer_rpcmem_munmap(remote_handle64 rpc_handle, int fd) {
  (void)rpc_handle;
  return HAP_mmap_put(fd);
}
