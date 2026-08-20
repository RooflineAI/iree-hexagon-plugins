// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_HAL_HEXAGONSESSION_H_
#define ROOF_HEXAGON_HAL_HEXAGONSESSION_H_

#include "iree/compiler/PluginAPI/Client.h"

namespace mlir::iree_compiler::hexagon::target {

bool registerHexagonPlugin(
    mlir::iree_compiler::PluginRegistrar *registrar);

} // namespace mlir::iree_compiler::hexagon::target

#endif // ROOF_HEXAGON_HAL_HEXAGONSESSION_H_
