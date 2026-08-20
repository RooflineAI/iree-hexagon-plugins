// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This header declares Hexagon-specific comprehensive bufferization helpers.

#ifndef ROOF_HEXAGON_CODEGEN_BUFFERIZATION_H_
#define ROOF_HEXAGON_CODEGEN_BUFFERIZATION_H_

#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::hexagon::codegen {

void addHexagonBufferizePasses(mlir::OpPassManager &funcPassManager);

void addHexagonBufferizePassesForHexagonMlir(
    mlir::OpPassManager &funcPassManager);

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_BUFFERIZATION_H_
