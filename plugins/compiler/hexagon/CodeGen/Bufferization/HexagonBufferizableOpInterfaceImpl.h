// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_BUFFERIZATION_HEXAGONBUFFERIZABLEOPINTERFACEIMPL_H_
#define ROOF_HEXAGON_CODEGEN_BUFFERIZATION_HEXAGONBUFFERIZABLEOPINTERFACEIMPL_H_

#include "mlir/IR/Dialect.h"

namespace mlir::iree_compiler::IREE::Hexagon {

// Attaches BufferizableOpInterface external models for iree_hexagon ops so they
// survive One-Shot Bufferize (e.g. `iree_hexagon.hmx.matmul`).
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

} // namespace mlir::iree_compiler::IREE::Hexagon

#endif // ROOF_HEXAGON_CODEGEN_BUFFERIZATION_HEXAGONBUFFERIZABLEOPINTERFACEIMPL_H_
