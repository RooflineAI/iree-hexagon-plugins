// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "import_provider.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AEEStdErr.h"
#include "HAP_farf.h"

// Hexagon runtime/helper symbols are now resolved natively by the DSP dynamic
// loader from libhexagon_dsp_skel.so. This import table remains only as a
// safety net for any unexpected HAL imports emitted elsewhere.

typedef struct {
  const char *name;
  iree_hal_executable_import_v0_t adapter;
  const void *targetFn;
} hexagon_dsp_import_entry_t;

static const hexagon_dsp_import_entry_t *kImportEntries = NULL;
static const size_t kImportEntryCount = 0;

static int hexagon_dsp_import_thunk_v0(iree_hal_executable_import_v0_t fn_ptr,
                                       void *params, void *context,
                                       void *reserved) {
  return fn_ptr(params, context, reserved);
}

static const char *hexagon_dsp_strip_optional_import(const char *symbol,
                                                     bool *isOptional) {
  *isOptional = symbol && symbol[0] == '?';
  return *isOptional ? symbol + 1 : symbol;
}

static void hexagon_dsp_log_available_imports(void) {
  FARF(ERROR, "HEXAGON-RUNTIME-ERROR: supported DSP import symbols:");
  for (size_t i = 0; i < kImportEntryCount; ++i) {
    FARF(ERROR, "  - %s", kImportEntries[i].name);
  }
}

static int
hexagon_dsp_resolve_import(const char *symbol,
                           iree_hal_executable_import_v0_t *outImportFn,
                           const void **outImportContext) {
  if (!outImportFn || !outImportContext) {
    FARF(ERROR,
         "HEXAGON-RUNTIME-ERROR: invalid output pointers while resolving "
         "import '%s'",
         symbol ? symbol : "(null)");
    return AEE_EBADPARM;
  }

  bool isOptional = false;
  const char *baseName = hexagon_dsp_strip_optional_import(symbol, &isOptional);
  if (!baseName || !baseName[0]) {
    FARF(ERROR,
         "HEXAGON-RUNTIME-ERROR: malformed import symbol (null or empty)");
    return AEE_EBADPARM;
  }

  const hexagon_dsp_import_entry_t *entry = NULL;
  for (size_t i = 0; i < kImportEntryCount; ++i) {
    if (strcmp(kImportEntries[i].name, baseName) == 0) {
      entry = &kImportEntries[i];
      break;
    }
  }
  if (!entry) {
    if (isOptional) {
      *outImportFn = NULL;
      *outImportContext = NULL;
      return AEE_SUCCESS;
    }
    FARF(ERROR, "HEXAGON-RUNTIME-ERROR: unresolved required import: %s",
         baseName);
    hexagon_dsp_log_available_imports();
    return AEE_ERESOURCENOTFOUND;
  }

  *outImportFn = entry->adapter;
  *outImportContext = entry->targetFn;
  return AEE_SUCCESS;
}

void hexagon_dsp_import_provider_deinitialize(
    iree_hal_executable_environment_v0_t *environment) {
  if (!environment) {
    return;
  }
  free((void *)environment->import_funcs);
  free((void *)environment->import_contexts);
  environment->import_thunk = NULL;
  environment->import_funcs = NULL;
  environment->import_contexts = NULL;
}

int hexagon_dsp_import_provider_initialize(
    const iree_hal_executable_import_table_v0_t *imports,
    iree_hal_executable_environment_v0_t *environment) {
  if (!environment) {
    return AEE_EBADPARM;
  }
  environment->import_funcs = NULL;
  environment->import_contexts = NULL;

  if (!imports || imports->count == 0) {
    return AEE_SUCCESS;
  }
  if (!imports->symbols) {
    FARF(ERROR,
         "HEXAGON-RUNTIME-ERROR: import table has count=%u but symbols array "
         "is null",
         (unsigned)imports->count);
    return AEE_EBADPARM;
  }

  iree_hal_executable_import_v0_t *import_funcs =
      (iree_hal_executable_import_v0_t *)calloc(imports->count,
                                                sizeof(*import_funcs));
  const void **import_contexts =
      (const void **)calloc(imports->count, sizeof(*import_contexts));
  if (!import_funcs || !import_contexts) {
    free(import_funcs);
    free((void *)import_contexts);
    return AEE_ENOMEMORY;
  }

  for (uint32_t i = 0; i < imports->count; ++i) {
    const char *symbol = imports->symbols[i];
    if (!symbol) {
      FARF(ERROR, "HEXAGON-RUNTIME-ERROR: import[%u/%u] symbol pointer is null",
           (unsigned)(i + 1), (unsigned)imports->count);
      free(import_funcs);
      free((void *)import_contexts);
      return AEE_EBADPARM;
    }
    int err = hexagon_dsp_resolve_import(symbol, &import_funcs[i],
                                         &import_contexts[i]);
    if (err != AEE_SUCCESS) {
      FARF(ERROR,
           "HEXAGON-RUNTIME-ERROR: failed to resolve import[%u/%u] '%s' "
           "(error=0x%x)",
           (unsigned)(i + 1), (unsigned)imports->count, symbol, err);
      free(import_funcs);
      free((void *)import_contexts);
      return err;
    }
  }

  environment->import_thunk = hexagon_dsp_import_thunk_v0;
  environment->import_funcs = import_funcs;
  environment->import_contexts = import_contexts;
  return AEE_SUCCESS;
}
