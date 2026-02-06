// Copyright 2025 RooflineAI GmbH

#ifndef HEXAGON_DSP_EXECUTABLE_H
#define HEXAGON_DSP_EXECUTABLE_H

#include <stdint.h>
#include <stdlib.h>

// This define is needed by executable_library.h to compile for Hexagon.
#define static_assert _Static_assert

#include "iree/hal/local/executable_library.h"

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

/***** internal functions,
 ***** running on DSP,
 ***** called by other DSP compilation units */

/**
 * @brief get dispatch function pointer for an entry point
 * @param[in] executable_handle handle of the executable
 * @param[in] export_ordinal number of exported function
 * @param[out] out_dispatch_func pointer to dispatch function
 * @retval AEE_SUCCESS for success
 */
int hexagon_dsp_executable_get_dispatch_func(
    int64 executable_handle, uint32_t export_ordinal,
    iree_hal_executable_dispatch_v0_t *out_dispatch_func);

/**
 * @brief Get the dispatch function.
 *
 * The export table may omit names to save binary size. When unavailable,
 * the corresponding output pointers are set to NULL.
 *
 * @param[in] executable_handle handle of the executable
 * @param[in] export_ordinal number of exported function
 * @param[out] out_name optional export name string
 * @retval AEE_SUCCESS for success
 */
int hexagon_dsp_executable_get_dispatch_func_name(int64 executable_handle,
                                                  uint32_t export_ordinal,
                                                  const char **out_name);

#endif // #ifndef HEXAGON_DSP_EXECUTABLE_H
