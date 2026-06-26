// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGTYPES_H
#define PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGTYPES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

namespace mlir::iree_compiler::IREE::Hexagon {

class IREEHexagonEncodingDialect;

} // namespace mlir::iree_compiler::IREE::Hexagon

#define GET_TYPEDEF_CLASSES
#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingTypes.h.inc"

#endif // PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGTYPES_H
