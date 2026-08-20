// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_HEXAGONENCODINGEXTERNALMODELS_H
#define ROOF_HEXAGON_CODEGEN_HEXAGONENCODINGEXTERNALMODELS_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir::iree_compiler::hexagon::codegen {

/// Registers the Hexagon encoding resolver interfaces.
void registerHexagonEncodingExternalModels(mlir::DialectRegistry &registry);

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_HEXAGONENCODINGEXTERNALMODELS_H
