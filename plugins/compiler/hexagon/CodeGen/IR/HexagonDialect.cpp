// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/IR/HexagonDialect.h"
#include "hexagon/CodeGen/IR/HexagonAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"

#include "hexagon/CodeGen/IR/HexagonDialect.cpp.inc"
#include "hexagon/CodeGen/IR/HexagonOps.h"

#include "mlir/IR/DialectImplementation.h"

namespace mlir::iree_compiler::IREE::Hexagon {

struct HexagonOpAsmInterface : OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (isa<LoweringConfigAttr>(attr)) {
      os << "config";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

void IREEHexagonDialect::initialize() {
  getContext()->getOrLoadDialect<Codegen::IREECodegenDialect>();
  registerAttributes();
  registerOperations();
  registerTypes();
  addInterfaces<HexagonOpAsmInterface>();
}

} // namespace mlir::iree_compiler::IREE::Hexagon
