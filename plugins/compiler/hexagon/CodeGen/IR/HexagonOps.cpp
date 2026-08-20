// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/IR/HexagonOps.h"

#include "hexagon/CodeGen/IR/HexagonDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "hexagon/CodeGen/IR/HexagonOps.cpp.inc"

namespace mlir::iree_compiler::IREE::Hexagon {

void IREEHexagonDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "hexagon/CodeGen/IR/HexagonOps.cpp.inc"
      >();
}

} // namespace mlir::iree_compiler::IREE::Hexagon
