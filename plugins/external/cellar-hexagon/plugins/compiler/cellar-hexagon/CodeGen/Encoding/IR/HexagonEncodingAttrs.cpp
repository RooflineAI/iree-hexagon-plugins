// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingAttrs.h"
#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingAttrs.cpp.inc"

void mlir::iree_compiler::IREE::Hexagon::IREEHexagonEncodingDialect::
    registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingAttrs.cpp.inc"
      >();
}
