// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_DSP_RT_EXECUTABLE_LIBRARY_H
#define HEXAGON_DSP_RT_EXECUTABLE_LIBRARY_H

// This define is needed by iree/hal/local/executable_library.h to compile for
// Hexagon.
#define static_assert _Static_assert

#include "iree/hal/local/executable_library.h"

#undef static_assert

#endif // #ifndef HEXAGON_DSP_RT_EXECUTABLE_LIBRARY_H
