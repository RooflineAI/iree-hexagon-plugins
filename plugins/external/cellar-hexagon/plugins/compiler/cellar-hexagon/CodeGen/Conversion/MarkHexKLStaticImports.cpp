// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/CodeGen/Passes.h"

#include "hexagon/Conversion/HexKLToLLVM/HexKLExternalFnNames.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {

#define GEN_PASS_DEF_MARKHEXKLSTATICIMPORTSPASS
#include "cellar-hexagon/CodeGen/Passes.h.inc"

namespace {

static llvm::StringSet<> getHexKLRuntimeSymbolNames() {
  llvm::StringSet<> names;
  names.insert(mlir::hexkl::getMatmulMicroFnName());
  names.insert(mlir::hexkl::getHmxConfigSizeFnName());
  names.insert(mlir::hexkl::getHmxSetupAccReadF16FnName());
  names.insert(mlir::hexkl::getHmxAccClearF16FnName());
  names.insert(mlir::hexkl::getHmxAccReadF16FnName());
  names.insert(mlir::hexkl::getHmxCopySubmatrixToF16FnName());
  names.insert(mlir::hexkl::getHmxRmToAhF16FnName());
  names.insert(mlir::hexkl::getHmxRmToWhF16FnName());
  names.insert(mlir::hexkl::getHmxMmF16FnName());
  names.insert(mlir::hexkl::getHmxAhToRmF16FnName());
  names.insert(mlir::hexkl::getHmxCopyF16ToF32SubmatrixFnName());
  return names;
}

struct MarkHexKLStaticImportsPass
    : public impl::MarkHexKLStaticImportsPassBase<MarkHexKLStaticImportsPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto symbolNames = getHexKLRuntimeSymbolNames();
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
