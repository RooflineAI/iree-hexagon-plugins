// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/CodeGen/Passes.h"

#include "hexagon/Conversion/DMAToLLVM/DMAExternalFnNames.h"
#include "hexagon/Conversion/HexagonMemToLLVM/HexagonMemExternalFnNames.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {

#define GEN_PASS_DEF_MARKHEXAGONRUNTIMESTATICIMPORTSPASS
#include "cellar-hexagon/CodeGen/Passes.h.inc"

namespace {

static llvm::StringSet<> getHexagonRuntimeSymbolNames() {
  llvm::StringSet<> names;
  names.insert(mlir::hexagon::getDMAStartFnName());
  names.insert(mlir::hexagon::getDMA2DStartFnName());
  names.insert(mlir::hexagon::getDMAWaitFnName());
  names.insert(mlir::hexagonmem::getAllocFnName());
  names.insert(mlir::hexagonmem::getDeallocFnName());
  names.insert(mlir::hexagonmem::getCopyFnName());
  return names;
}

struct MarkHexagonRuntimeStaticImportsPass
    : public impl::MarkHexagonRuntimeStaticImportsPassBase<
          MarkHexagonRuntimeStaticImportsPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto symbolNames = getHexagonRuntimeSymbolNames();
    mlir::UnitAttr staticImportAttr = mlir::UnitAttr::get(&getContext());

    for (mlir::LLVM::LLVMFuncOp funcOp :
         getOperation().getOps<mlir::LLVM::LLVMFuncOp>()) {
      if (!funcOp.isExternal())
        continue;

      mlir::StringAttr symName = funcOp.getSymNameAttr();
      if (!symName || !symbolNames.contains(symName.getValue()))
        continue;

      funcOp->setAttr("hal.import.static", staticImportAttr);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::cellar_hexagon::codegen
