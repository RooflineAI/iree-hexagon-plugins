// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/IR/HexagonDialect.h"

#include "hexagon/CodeGen/IR/HexagonDialect.cpp.inc"
#include "hexagon/CodeGen/IR/HexagonOps.h"

#include "mlir/IR/DialectImplementation.h"

namespace mlir::iree_compiler::IREE::Hexagon {

void IREEHexagonDialect::initialize() {
  registerAttributes();
  registerOperations();
  registerTypes();
}

} // namespace mlir::iree_compiler::IREE::Hexagon
