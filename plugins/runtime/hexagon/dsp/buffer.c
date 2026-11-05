// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>
#include <stdlib.h>

#include "AEEStdErr.h"
#include "hexagon/dsp/device.h"
#include "hexagon/dsp/residency.h"
#include "hexagon_dsp.h"

/**
 * @brief Unmap shared RPC memory the host is about to free, if the residency
 *        cache has mapped it.
 * @param[in] rpc_handle handle of DSP RPC session
 * @param[in] fd file descriptor of shared RPC memory
 * @param[out] in_use set to 1 if a command still uses the memory, it is not
 *             unmapped then and the host must not free it
 * @retval AEE_SUCCESS for success
 */
int hexagon_dsp_buffer_forget(remote_handle64 rpc_handle, int fd, int *in_use) {
  bool is_in_use = false;
  hexagon_dsp_residency_forget(hexagon_dsp_get_residency(rpc_handle), fd,
                               &is_in_use);
  *in_use = is_in_use ? 1 : 0;
  return AEE_SUCCESS;
}
