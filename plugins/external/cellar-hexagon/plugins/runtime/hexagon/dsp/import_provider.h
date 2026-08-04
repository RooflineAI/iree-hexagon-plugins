// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_IMPORT_PROVIDER_H
#define HEXAGON_DSP_IMPORT_PROVIDER_H

#include <stdint.h>

#include "hexagon/dsp/rt/executable_library.h"

// Resolves helper executable imports from `imports` and initializes
// `environment`. Any allocated import storage is owned through the environment
// pointer and must be released with `hexagon_dsp_import_provider_deinitialize`.
int hexagon_dsp_import_provider_initialize(
    const iree_hal_executable_import_table_v0_t *imports,
    iree_hal_executable_environment_v0_t *environment);

// Deinitializes import-related environment state and frees owned storage.
void hexagon_dsp_import_provider_deinitialize(
    iree_hal_executable_environment_v0_t *environment);

#endif // HEXAGON_DSP_IMPORT_PROVIDER_H
