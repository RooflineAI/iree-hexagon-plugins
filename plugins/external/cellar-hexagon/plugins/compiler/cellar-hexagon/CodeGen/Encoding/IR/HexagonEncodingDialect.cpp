// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingDialect.h"

#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingOps.h"
#include "mlir/IR/DialectImplementation.h"

#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingDialect.cpp.inc"

namespace mlir::iree_compiler::IREE::Hexagon {

void IREEHexagonEncodingDialect::initialize() {
  registerAttributes();
  registerOperations();
  registerTypes();
}

} // namespace mlir::iree_compiler::IREE::Hexagon
