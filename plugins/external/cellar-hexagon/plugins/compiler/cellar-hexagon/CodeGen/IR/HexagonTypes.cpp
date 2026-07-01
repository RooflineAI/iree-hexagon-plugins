// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/CodeGen/IR/HexagonTypes.h"

#include "cellar-hexagon/CodeGen/IR/HexagonDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "cellar-hexagon/CodeGen/IR/HexagonTypes.cpp.inc"

void mlir::iree_compiler::IREE::Hexagon::IREEHexagonDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cellar-hexagon/CodeGen/IR/HexagonTypes.cpp.inc"
      >();
}
