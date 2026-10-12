// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/IR/HexagonAttrs.h"
#include "hexagon/CodeGen/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONPROPAGATEDISPATCHRESOURCESPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

using IREE::Hexagon::DispatchResourcesAttr;

struct HexagonPropagateDispatchResourcesPass
    : impl::HexagonPropagateDispatchResourcesPassBase<
          HexagonPropagateDispatchResourcesPass> {
  void runOnOperation() override {
    IREE::HAL::ExecutableVariantOp variantOp = getOperation();
    ModuleOp innerModule = variantOp.getInnerModule();
    if (!innerModule)
      return;

    SymbolTable innerSymbols(innerModule);
    for (auto exportOp : variantOp.getExportOps()) {
      auto funcOp =
          innerSymbols.lookup<FunctionOpInterface>(exportOp.getSymName());
      if (!funcOp)
        continue;
      IREE::Codegen::TranslationInfoAttr translationInfo =
          getTranslationInfo(funcOp);
      if (!translationInfo)
        continue;
      DictionaryAttr configuration = translationInfo.getConfiguration();
      if (!configuration)
        continue;
      auto resources = dyn_cast_if_present<DispatchResourcesAttr>(
          configuration.get(DispatchResourcesAttr::getAttrName()));
      if (!resources)
        continue;

      assert((!exportOp->hasAttr(DispatchResourcesAttr::getAttrName()) ||
              exportOp->getAttr(DispatchResourcesAttr::getAttrName()) ==
                  resources) &&
             "conflicting dispatch resources on export");

      exportOp->setAttr(DispatchResourcesAttr::getAttrName(), resources);
#
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen
