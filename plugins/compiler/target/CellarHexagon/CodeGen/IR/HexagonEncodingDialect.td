// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGDIALECT
#define PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGDIALECT

include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition.
//===----------------------------------------------------------------------===//

def IREEHexagonEncoding_Dialect : Dialect {
  let name = "iree_hexagon";
  let summary = "Roofline Hexagon encoding helpers.";
  let description = [{
    A lightweight dialect that currently only hosts attributes used to describe
    encoding information for the custom Hexagon backend.
    This may be expanded to become a more complex dialect in the future.
    If so, please update the name!
  }];
  let cppNamespace = "::mlir::iree_compiler::IREE::Hexagon";
  let extraClassDeclaration = [{
    void registerAttributes();
  }];

  let useDefaultAttributePrinterParser = 1;
}

#endif // PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGDIALECT
