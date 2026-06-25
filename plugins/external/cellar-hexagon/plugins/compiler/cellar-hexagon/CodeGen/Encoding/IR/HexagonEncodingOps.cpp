// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingOps.h"

#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler::IREE::Hexagon {

void IREEHexagonEncodingDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingOps.cpp.inc"
      >();
}

} // namespace mlir::iree_compiler::IREE::Hexagon

#define GET_OP_CLASSES
#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingOps.cpp.inc"
