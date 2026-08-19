// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONDIALECT
#define PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONDIALECT

include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition.
//===----------------------------------------------------------------------===//

def IREEHexagon_Dialect : Dialect {
  let name = "iree_hexagon";
  let summary = "Roofline Hexagon encoding helpers.";
  let description = [{
    A lightweight dialect that hosts attributes and operations used by
    the custom Hexagon backend. It includes, for example, memory-staging
    information for VTCM management.
  }];
  let cppNamespace = "::mlir::iree_compiler::IREE::Hexagon";
  let extraClassDeclaration = [{
    void registerAttributes();
    void registerOperations();
    void registerTypes();
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif // PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONDIALECT
