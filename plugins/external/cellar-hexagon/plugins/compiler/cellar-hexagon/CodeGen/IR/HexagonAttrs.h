// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONATTRS_H
#define PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONATTRS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

namespace mlir::iree_compiler::IREE::Hexagon {

class IREEHexagonDialect;

} // namespace mlir::iree_compiler::IREE::Hexagon

#define GET_ATTRDEF_CLASSES
#include "cellar-hexagon/CodeGen/IR/HexagonAttrs.h.inc"

#endif // PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONATTRS_H
