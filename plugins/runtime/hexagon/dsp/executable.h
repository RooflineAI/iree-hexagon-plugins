// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_EXECUTABLE_H
#define HEXAGON_DSP_EXECUTABLE_H

#include <stdint.h>
#include <stdlib.h>

#include "hexagon/dsp/rt/executable_library.h"

#include "hexagon_dsp.h"

/***** DSP remote procedures,
 ***** running on DSP,
 ***** called by ARM host side via RPC mechanism */

/**
 * @brief Load executable (shared library).
 * @param[in] rpc_handle handle of DSP RPC session
 * @param[in] name name of executable (shared object) to load
 * @param[out] executable_handle handle of the loaded executable
 * @retval, AEE_SUCCESS for success
 */
int hexagon_dsp_executable_load(remote_handle64 rpc_handle, const char *name,
                                int64 *executable_handle);

/**
 * @brief Close executable.
 * @param[in] rpc_handle handle of DSP RPC session
 * @param[in] executable_handle handle of the executable
 * @retval AEE_SUCCESS for success, should always succeed
 */
int hexagon_dsp_executable_close(remote_handle64 rpc_handle,
                                 int64 executable_handle);

/**
 * @brief Look up an exported function by name.
 * @param[in] rpc_handle handle of DSP RPC session
 * @param[in] executable_handle handle of the executable
 * @param[in] name name of the exported function
 * @param[out] function_handle handle of the function, a pointer to a
 *             hexagon_dsp_function_t owned by the executable
 * @retval AEE_SUCCESS for success, AEE_ERESOURCENOTFOUND if there is no export
 *         with that name
 */
int hexagon_dsp_executable_lookup_function(remote_handle64 rpc_handle,
                                           int64 executable_handle,
                                           const char *name,
                                           int64 *function_handle);

/***** DSP internal functions *****/

/// exported function of a loaded executable, the RPC function handle points
/// to this, valid as long as the executable is loaded
typedef struct hexagon_dsp_function_s hexagon_dsp_function_t;

/**
 * @brief get dispatch function pointer of a function
 * @param[in] function function
 * @return dispatch function pointer
 */
hexagon_executable_dispatch_fn_t
hexagon_dsp_function_get_dispatch_func(const hexagon_dsp_function_t *function);

/**
 * @brief get initialized environment of the executable containing a function
 * @param[in] function function
 * @return initialized HAL executable environment
 */
const hexagon_executable_environment_t *
hexagon_dsp_function_get_environment(const hexagon_dsp_function_t *function);

/**
 * @brief get export name of a function
 * @param[in] function function
 * @return export name, for profiling and diagnostics
 */
const char *
hexagon_dsp_function_get_name(const hexagon_dsp_function_t *function);

/**
 * @brief get dispatch attributes of a function
 * @param[in] function function
 * @return dispatch attributes, NULL if the executable has no attribute table
 */
const hexagon_executable_dispatch_attrs_t *
hexagon_dsp_function_get_attrs(const hexagon_dsp_function_t *function);

#endif // #ifndef HEXAGON_DSP_EXECUTABLE_H
