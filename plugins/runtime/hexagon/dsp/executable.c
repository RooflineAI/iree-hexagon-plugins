// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable.h"

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
  hexagon_executable_library_query_fn_t query_func;
  /// library information
  /// The approach is taken from
  /// iree/hal/local/loaders/static_library_loader.c. The executable_library
  /// API works with "..._header_t **", but the specific implementation needs
  /// more information. So the "..._library_t" structure contains a
  /// "..._header_t*" as the first member. The union is then used to "cast"
  /// the "..._header_t**" to a "..._library_t*" by assigning the incoming
  /// "..._header_t**" to "header" and using "v0" to access the "derived class"
  /// information.
  union {
    const hexagon_executable_library_header_t **header;
    const hexagon_executable_library_t *v0;
  } library;
  /// executable environment passed to dispatches
  /// FIXME: zero-initialized; executable constants are not populated.
  hexagon_executable_environment_t environment;
  /// exported functions, one per export, RPC function handles point here
  hexagon_dsp_function_t *functions;
  /// number of entries in functions
  uint32_t function_count;
} hexagon_dsp_executable_t;

struct hexagon_dsp_function_s {
  /// executable that contains the function
  const hexagon_dsp_executable_t *executable;
  /// dispatch function pointer
  hexagon_executable_dispatch_fn_t dispatch_func;
  /// export name for profiling and diagnostics
  const char *name;
  /// dispatch attributes, NULL if the executable has no attribute table
  const hexagon_executable_dispatch_attrs_t *attrs;
};

/// set up function table of executable, inside hexagon_dsp_executable_set_up()
static int
hexagon_dsp_executable_set_up_functions(hexagon_dsp_executable_t *executable) {
  const hexagon_executable_export_table_t *exports =
      &executable->library.v0->exports;
  if (exports->count == 0) {
    return AEE_SUCCESS;
  }
  // functions are looked up by name, so names are required
  if (!exports->names) {
    FARF(ERROR,
         "HEXAGON-RUNTIME-ERROR: executable '%s' has no export names, "
         "cannot look up functions by name",
         executable->name ? executable->name : "(null)");
    return AEE_EUNSUPPORTED;
  }
  executable->functions = (hexagon_dsp_function_t *)calloc(
      exports->count, sizeof(hexagon_dsp_function_t));
  if (!executable->functions) {
    return AEE_ENOMEMORY;
  }
  executable->function_count = exports->count;
  for (uint32_t i = 0; i < exports->count; ++i) {
    hexagon_dsp_function_t *function = &executable->functions[i];
    function->executable = executable;
    function->dispatch_func = exports->ptrs[i];
    function->name = exports->names[i];
    function->attrs = exports->attrs ? &exports->attrs[i] : NULL;
  }
  return AEE_SUCCESS;
}

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
  executable->query_func = (hexagon_executable_library_query_fn_t)dlsym(
      executable->shlib, HEXAGON_EXECUTABLE_LIBRARY_EXPORT_NAME);
  if (!executable->query_func) {
    return AEE_ERESOURCENOTFOUND;
  }

  // query library version
  executable->library.header =
      (const hexagon_executable_library_header_t **)executable->query_func(
          HEXAGON_EXECUTABLE_LIBRARY_VERSION_LATEST, &executable->environment);
  if (!executable->library.header) {
    return AEE_EUNSUPPORTED;
  }

  // check library version
  const hexagon_executable_library_header_t *header =
      *executable->library.header;
  if (header->version != HEXAGON_EXECUTABLE_LIBRARY_VERSION_LATEST) {
    return AEE_EVERSIONNOTSUPPORT;
  }

  // FIXME: dynamic HAL imports are not supported.
  if (executable->library.v0->imports.count != 0) {
    FARF(ERROR,
         "HEXAGON-RUNTIME-ERROR: executable '%s' declares %u HAL imports, "
         "but the Hexagon plugins currently support only static/native DSP "
         "runtime linking",
         executable->name ? executable->name : "(null)",
         (unsigned)executable->library.v0->imports.count);
    return AEE_EUNSUPPORTED;
  }

  return hexagon_dsp_executable_set_up_functions(executable);
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
  // close shared library, free resources
  if (executable->shlib) {
    dlclose(executable->shlib);
  }
  free(executable->functions);
  free(executable->name);
  // free management data structure
  free(executable);
  return AEE_SUCCESS;
}

int hexagon_dsp_executable_lookup_function(remote_handle64 rpc_handle,
                                           int64 executable_handle,
                                           const char *name,
                                           int64 *function_handle) {
  if (!executable_handle || !name || !function_handle) {
    return AEE_EBADPARM;
  }
  hexagon_dsp_executable_t *executable =
      (hexagon_dsp_executable_t *)executable_handle;
  for (uint32_t i = 0; i < executable->function_count; ++i) {
    hexagon_dsp_function_t *function = &executable->functions[i];
    if (function->name && strcmp(function->name, name) == 0) {
      *function_handle = (int64)function;
      return AEE_SUCCESS;
    }
  }
  return AEE_ERESOURCENOTFOUND;
}

hexagon_executable_dispatch_fn_t
hexagon_dsp_function_get_dispatch_func(const hexagon_dsp_function_t *function) {
  return function->dispatch_func;
}

const hexagon_executable_environment_t *
hexagon_dsp_function_get_environment(const hexagon_dsp_function_t *function) {
  return &function->executable->environment;
}

const char *
hexagon_dsp_function_get_name(const hexagon_dsp_function_t *function) {
  return function->name;
}

const hexagon_executable_dispatch_attrs_t *
hexagon_dsp_function_get_attrs(const hexagon_dsp_function_t *function) {
  return function->attrs;
}
