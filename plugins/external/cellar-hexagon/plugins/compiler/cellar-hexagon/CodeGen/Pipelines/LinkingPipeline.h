// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This header declares the Hexagon executable linking pipeline builder.

#ifndef ROOF_HEXAGON_CODEGEN_LINKINGPIPELINE_H_
#define ROOF_HEXAGON_CODEGEN_LINKINGPIPELINE_H_

#include <optional>
#include <string>

#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {

void buildHexagonLinkingPassPipeline(
    mlir::OpPassManager &passManager,
    std::optional<std::string> target = std::nullopt);

} // namespace mlir::iree_compiler::cellar_hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_LINKINGPIPELINE_H_
