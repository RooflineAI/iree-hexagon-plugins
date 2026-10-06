// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Derived from Codegen/LLVMCPU/LLVMCPUSynchronizeSymbolVisibility.cpp at IREE
// revision a45adeaa6115e446c898e6eb21fb6edc0e65ddc4.

#include "hexagon/CodeGen/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONSYNCHRONIZESYMBOLVISIBILITYPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

static void setVisibilityFromLinkage(SymbolOpInterface op,
                                     LLVM::Linkage linkage) {
  SymbolTable::Visibility visibility = op.getVisibility();
  switch (linkage) {
  case LLVM::Linkage::Private:
  case LLVM::Linkage::Internal:
    visibility = SymbolTable::Visibility::Private;
    break;
  default:
    visibility = SymbolTable::Visibility::Public;
    break;
  }
  op.setVisibility(visibility);
}

struct HexagonSynchronizeSymbolVisibilityPass
    : impl::HexagonSynchronizeSymbolVisibilityPassBase<
          HexagonSynchronizeSymbolVisibilityPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    for (auto &op : moduleOp.getOps()) {
      if (auto globalOp = dyn_cast<LLVM::GlobalOp>(op)) {
        setVisibilityFromLinkage(globalOp, globalOp.getLinkage());
      } else if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        setVisibilityFromLinkage(funcOp, funcOp.getLinkage());
      }
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen
