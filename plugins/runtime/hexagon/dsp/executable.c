// Copyright 2025 RooflineAI GmbH

#include <dlfcn.h>
#include <stdlib.h>

#include "AEEStdErr.h"
#include "hexagon_dsp.h"

/// data about a loaded executable
typedef struct hexagon_dsp_executable_s {
  /// handle of owning RPC session
  remote_handle64 rpc_handle;
  /// name of executable
  char *name;
  /// pointer to shared library (as returned by dlopen)
  void *shlib;
} hexagon_dsp_executable_t;

/**
 * @brief Load executable (shared library).
 * @param[in] rpc_handle handle of DSP RPC session
 * @param[in] name name of executable (shared object) to load
 * @param[out] executable_handle handle of the loaded executable
 * @retval, AEE_SUCCESS for success
 */
int hexagon_dsp_executable_load(remote_handle64 rpc_handle, const char *name,
                                int64 *executable_handle) {
  // allocate internal data structure used to mange executable
  hexagon_dsp_executable_t *executable =
      (hexagon_dsp_executable_t *)calloc(1, sizeof(hexagon_dsp_executable_t));
  if (!executable) {
    return AEE_ENOMEMORY;
  }
  executable->rpc_handle = rpc_handle;

  // store name of executable
  executable->name = strdup(name);
  if (!executable->name) {
    hexagon_dsp_executable_close(rpc_handle, (int64)executable);
    return AEE_ENOMEMORY;
  }

  // open shared library
  executable->shlib = dlopen(executable->name, RTLD_NOW);
  if (!executable->shlib) {
    hexagon_dsp_executable_close(rpc_handle, (int64)executable);
    return AEE_EUNABLETOLOAD;
  }

  // return pointer to internal data structure as handle
  *executable_handle = (int64)executable;
  return AEE_SUCCESS;
}

/**
 * @brief Close executable.
 * @param[in] rpc_handle handle of DSP RPC session
 * @param[in] executable_handle handle of the executable
 * @retval AEE_SUCCESS for success, should always succeed
 */
int hexagon_dsp_executable_close(remote_handle64 rpc_handle,
                                 int64 executable_handle) {
  hexagon_dsp_executable_t *executable =
      (hexagon_dsp_executable_t *)executable_handle;
  // close shared library, free resources
  if (executable->shlib) {
    dlclose(executable->shlib);
  }
  free(executable->name);
  // free management data structure
  free(executable);
  return AEE_SUCCESS;
}
