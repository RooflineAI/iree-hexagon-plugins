// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_RT_DISPATCH_STATE_H
#define HEXAGON_DSP_RT_DISPATCH_STATE_H

#include "hexagon/dsp/rt/executable_library.h"
#include "hexagon/dsp/rt/runtime_state_fwd_decl.h"

#include <stddef.h>

// Extended dispatch state for Hexagon. It's the common dispatch state followed
// by a pointer to the runtime state structure. It enables (ab)using the second
// arg passed to dispatches to get access to the runtime state.
typedef struct hexagon_dsp_extended_dispatch_state_s {
  /// original IREE dispatch state !!! keep this member the first one !!!
  iree_hal_executable_dispatch_state_v0_t dispatch_state;
  /// pointer to the runtime state !!! keep this member right after the base
  /// dispatch state; codegen assumes this offset !!!
  const hexagon_rt_state_t *runtime_state;
} hexagon_dsp_extended_dispatch_state_t;

#endif // #ifndef HEXAGON_DSP_RT_DISPATCH_STATE_H
