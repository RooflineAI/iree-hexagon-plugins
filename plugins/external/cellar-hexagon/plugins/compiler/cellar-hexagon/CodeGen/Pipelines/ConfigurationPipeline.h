// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This header declares the configuration-stage Hexagon pipeline builder.

#ifndef ROOF_HEXAGON_CODEGEN_CONFIGURATIONPIPELINE_H_
#define ROOF_HEXAGON_CODEGEN_CONFIGURATIONPIPELINE_H_

#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {

void buildHexagonConfigurationPassPipeline(mlir::OpPassManager &passManager);

} // namespace mlir::iree_compiler::cellar_hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_CONFIGURATIONPIPELINE_H_
