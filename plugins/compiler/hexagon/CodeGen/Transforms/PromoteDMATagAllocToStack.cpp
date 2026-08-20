// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_PROMOTEDMATAGALLOCTOSTACKPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

// Promote:
//   %tag = memref.alloc() : memref<1xi32>
//   memref.dma_start ..., %tag[%c0] : ..., memref<1xi32>
//   memref.dma_wait %tag[%c0], ... : memref<1xi32>
//   memref.dealloc %tag : memref<1xi32>
// into:
//   %tag = memref.alloca() : memref<1xi32>
//   memref.dma_start ..., %tag[%c0] : ..., memref<1xi32>
//   memref.dma_wait %tag[%c0], ... : memref<1xi32>

// Note that we do not promote all allocations blindly, but instead this is
// exclusively targeted to this format.

static bool isEligibleDMATagAlloc(
    mlir::memref::AllocOp allocOp,
    llvm::SmallVectorImpl<mlir::memref::DeallocOp> &deallocOps) {
  mlir::MemRefType type = allocOp.getType();
  if (type.getRank() != 1 || type.getDimSize(0) != 1 ||
      !type.getElementType().isSignlessInteger(32) || type.getMemorySpace()) {
    return false;
  }

  bool hasDMAStartUse = false;
  bool hasDMAWaitUse = false;
  for (mlir::Operation *user : allocOp->getUsers()) {
    if (auto dmaStart = llvm::dyn_cast<mlir::memref::DmaStartOp>(user)) {
      if (dmaStart.getTagMemRef() != allocOp.getMemref()) {
        return false;
      }
      hasDMAStartUse = true;
      continue;
    }
    if (auto dmaWait = llvm::dyn_cast<mlir::memref::DmaWaitOp>(user)) {
      if (dmaWait.getTagMemRef() != allocOp.getMemref()) {
        return false;
      }
      hasDMAWaitUse = true;
      continue;
    }
    if (auto dealloc = llvm::dyn_cast<mlir::memref::DeallocOp>(user)) {
      if (dealloc.getMemref() != allocOp.getMemref()) {
        return false;
      }
      deallocOps.push_back(dealloc);
      continue;
    }
    return false;
  }
  return hasDMAStartUse && hasDMAWaitUse;
}

struct PromoteDMATagAllocToStackPass final
    : public impl::PromoteDMATagAllocToStackPassBase<
          PromoteDMATagAllocToStackPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    llvm::SmallVector<mlir::memref::AllocOp> allocOpsToPromote;

    funcOp.walk([&](mlir::memref::AllocOp allocOp) {
      llvm::SmallVector<mlir::memref::DeallocOp> unusedDeallocs;
      if (isEligibleDMATagAlloc(allocOp, unusedDeallocs)) {
        allocOpsToPromote.push_back(allocOp);
      }
    });

    mlir::IRRewriter rewriter(&getContext());
    for (mlir::memref::AllocOp allocOp : allocOpsToPromote) {
      llvm::SmallVector<mlir::memref::DeallocOp> deallocOps;
      if (!isEligibleDMATagAlloc(allocOp, deallocOps)) {
        continue;
      }

      rewriter.setInsertionPoint(allocOp);
      mlir::memref::AllocaOp allocaOp = mlir::memref::AllocaOp::create(
          rewriter, allocOp.getLoc(), allocOp.getType(),
          allocOp.getDynamicSizes(), allocOp.getAlignmentAttr());

      allocOp.getMemref().replaceAllUsesWith(allocaOp.getMemref());
      for (mlir::memref::DeallocOp deallocOp : deallocOps) {
        rewriter.eraseOp(deallocOp);
      }
      rewriter.eraseOp(allocOp);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::hexagon::codegen
