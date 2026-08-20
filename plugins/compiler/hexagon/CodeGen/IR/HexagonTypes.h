// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_IR_HEXAGONTYPES_H
#define ROOF_HEXAGON_CODEGEN_IR_HEXAGONTYPES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

namespace mlir::iree_compiler::IREE::Hexagon {

class IREEHexagonDialect;

} // namespace mlir::iree_compiler::IREE::Hexagon

#define GET_TYPEDEF_CLASSES
#include "hexagon/CodeGen/IR/HexagonTypes.h.inc"

#endif // ROOF_HEXAGON_CODEGEN_IR_HEXAGONTYPES_H
