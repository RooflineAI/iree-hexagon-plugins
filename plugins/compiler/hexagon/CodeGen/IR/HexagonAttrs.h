// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_IR_HEXAGONATTRS_H
#define ROOF_HEXAGON_CODEGEN_IR_HEXAGONATTRS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

namespace mlir::iree_compiler::IREE::Hexagon {

class IREEHexagonDialect;

} // namespace mlir::iree_compiler::IREE::Hexagon

// clang-format off: the enum declarations must come before the attributes
// wrapping them.
#include "hexagon/CodeGen/IR/HexagonEnums.h.inc" // IWYU pragma: export

#define GET_ATTRDEF_CLASSES
#include "hexagon/CodeGen/IR/HexagonAttrs.h.inc"
// clang-format on

#endif // ROOF_HEXAGON_CODEGEN_IR_HEXAGONATTRS_H
