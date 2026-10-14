// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_DEVICE_H
#define HEXAGON_DSP_DEVICE_H

#include <stdint.h>

#include "remote.h"

/**
 * @brief Max. number of dispatches executing at the same time in this RPC
 *        session, as set by the host; 1 means sequential execution.
 * @param[in] rpc_handle RPC handle, the value returned by hexagon_dsp_open
 */
uint32_t hexagon_dsp_get_max_concurrent_dispatches(remote_handle64 rpc_handle);

#endif // #ifndef HEXAGON_DSP_DEVICE_H
