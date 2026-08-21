// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable.h"
#include "import_provider.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

#include "AEEStdErr.h"
#include "HAP_farf.h"

/// data about a loaded executable
typedef struct hexagon_dsp_executable_s {
  /// handle of owning RPC session
  remote_handle64 rpc_handle;
  /// name of executable
  char *name;
  /// pointer to shared library (as returned by dlopen)
  void *shlib;
  /// query function pointer
  iree_hal_executable_library_query_fn_t query_func;
  /// library information
  /// The approach is taken from
  /// iree/hal/local/loaders/static_library_loader.c:30. The executable_library
  /// API works with "..._header_t **", but the specific implementation needs
  /// more information. So the "..._library_v0_t" structure contains a
  /// "..._header_t*" as the first member. The union is then used to "cast"
  /// the "..._header_t**" to a "..._library_v0_t*" by assigning the incoming
  /// "..._header_t**" to "header" and using "v0" to access the "derived class"
  /// information.
  union {
    const iree_hal_executable_library_header_t **header;
    const iree_hal_executable_library_v0_t *v0;
  } library;
  /// executable environment passed to dispatches
  iree_hal_executable_environment_v0_t environment;
} hexagon_dsp_executable_t;

/// set up executable data structure, used inside hexagon_dsp_executable_load()
int hexagon_dsp_executable_set_up(hexagon_dsp_executable_t *executable,
                                  const char *name) {
  // store name of executable
  executable->name = strdup(name);
  if (!executable->name) {
    return AEE_ENOMEMORY;
  }

  // open shared library
  executable->shlib = dlopen(executable->name, RTLD_NOW);
  if (!executable->shlib) {
    const char *loadError = dlerror();
    FARF(ERROR,
         "HEXAGON-RUNTIME-ERROR: failed to dlopen kernel '%s': %s. "
         "Ensure libhexagon_dsp_skel.so exports the required "
         "DMA/HexKL/HexagonMem symbols.",
         executable->name ? executable->name : "(null)",
         loadError ? loadError : "(no loader error message)");
    return AEE_EUNABLETOLOAD;
  }

  // get query function
  executable->query_func = (iree_hal_executable_library_query_fn_t)dlsym(
      executable->shlib, IREE_HAL_EXECUTABLE_LIBRARY_EXPORT_NAME);
  if (!executable->query_func) {
    return AEE_ERESOURCENOTFOUND;
  }

  // query library version
  executable->library.header =
      (const iree_hal_executable_library_header_t **)executable->query_func(
          IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST, &executable->environment);
  if (!executable->library.header) {
    return AEE_EUNSUPPORTED;
  }

  // check library version and that no sanitizers are compiled into it
  const iree_hal_executable_library_header_t *header =
      *executable->library.header;
  if (header->version != IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST) {

    return AEE_EVERSIONNOTSUPPORT;
  }
  if (header->sanitizer != IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE) {
    return AEE_ECLASSNOTSUPPORT;
  }

  int err = hexagon_dsp_import_provider_initialize(
      &executable->library.v0->imports, &executable->environment);
  if (err != AEE_SUCCESS) {
    FARF(ERROR,
         "HEXAGON-RUNTIME-ERROR: failed to initialize import provider for '%s' "
         "(error=0x%x)",
         executable->name ? executable->name : "(null)", err);
    return err;
  }

  return AEE_SUCCESS;
}

int hexagon_dsp_executable_load(remote_handle64 rpc_handle, const char *name,
                                int64 *executable_handle) {
  // allocate internal data structure used to mange executable
  hexagon_dsp_executable_t *executable =
      (hexagon_dsp_executable_t *)calloc(1, sizeof(hexagon_dsp_executable_t));
  if (!executable) {
    return AEE_ENOMEMORY;
  }
  executable->rpc_handle = rpc_handle;

  // set up executable data structure
  int err = hexagon_dsp_executable_set_up(executable, name);
  if (err != AEE_SUCCESS) {
    hexagon_dsp_executable_close(rpc_handle, (int64)executable);
    return err;
  }

  // return pointer to internal data structure as handle
  *executable_handle = (int64)executable;
  return AEE_SUCCESS;
}

int hexagon_dsp_executable_close(remote_handle64 rpc_handle,
                                 int64 executable_handle) {
  hexagon_dsp_executable_t *executable =
      (hexagon_dsp_executable_t *)executable_handle;
  hexagon_dsp_import_provider_deinitialize(&executable->environment);
  // close shared library, free resources
  if (executable->shlib) {
    dlclose(executable->shlib);
  }
  free(executable->name);
  // free management data structure
  free(executable);
  return AEE_SUCCESS;
}

int hexagon_dsp_executable_get_dispatch_func(
    int64 executable_handle, uint32_t export_ordinal,
    iree_hal_executable_dispatch_v0_t *out_dispatch_func) {
  hexagon_dsp_executable_t *executable =
      (hexagon_dsp_executable_t *)executable_handle;
  const iree_hal_executable_export_table_v0_t *exports =
      &executable->library.v0->exports;

  if (export_ordinal >= exports->count) {
    return AEE_EBADITEM;
  }

  *out_dispatch_func = exports->ptrs[export_ordinal];
  return AEE_SUCCESS;
}

int hexagon_dsp_executable_get_environment(
    int64 executable_handle,
    const iree_hal_executable_environment_v0_t **out_environment) {
  if (!out_environment) {
    return AEE_EBADPARM;
  }
  hexagon_dsp_executable_t *executable =
      (hexagon_dsp_executable_t *)executable_handle;
  *out_environment = &executable->environment;
  return AEE_SUCCESS;
}

int hexagon_dsp_executable_get_dispatch_func_name(int64 executable_handle,
                                                  uint32_t export_ordinal,
                                                  const char **out_name) {
  hexagon_dsp_executable_t *executable =
      (hexagon_dsp_executable_t *)executable_handle;
  const iree_hal_executable_export_table_v0_t *exports =
      &executable->library.v0->exports;

  if (export_ordinal >= exports->count || !out_name) {
    FARF(RUNTIME_HIGH, "HEXAGON-RUNTIME-WARNING: Failed to retrieve dispatch "
                       "function name.\n");
    return AEE_EBADPARM;
  }

  *out_name = exports->names ? exports->names[export_ordinal] : NULL;

  return AEE_SUCCESS;
}
