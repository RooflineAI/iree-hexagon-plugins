// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/CodeGen/Conversion/HexagonRuntimeLinking.h"
#include "cellar-hexagon/CodeGen/Passes.h"

#include "cellar-hexagon/CodeGen/IR/HexagonDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {
namespace IREE = mlir::iree_compiler::IREE;

#define GEN_PASS_DEF_MARKHEXAGONNATIVERUNTIMELINKSPASS
#include "cellar-hexagon/CodeGen/Passes.h.inc"

namespace {

struct MarkHexagonNativeRuntimeLinksPass
    : public impl::MarkHexagonNativeRuntimeLinksPassBase<
          MarkHexagonNativeRuntimeLinksPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry
        .insert<mlir::LLVM::LLVMDialect, IREE::Hexagon::IREEHexagonDialect>();
  }

  void runOnOperation() override {
    bool taggedAny = false;
    ModuleOp moduleOp = getOperation();

    for (mlir::LLVM::LLVMFuncOp funcOp :
         moduleOp.getOps<mlir::LLVM::LLVMFuncOp>()) {
      if (failed(renameAndTagNativeRuntimeLinkedFunc(moduleOp, funcOp)))
        return signalPassFailure();
      taggedAny |= funcOp->hasAttr(kNativeRuntimeLinkAttrName);
    }

    if (!taggedAny)
      return;
  }
};

} // namespace

} // namespace mlir::iree_compiler::cellar_hexagon::codegen
