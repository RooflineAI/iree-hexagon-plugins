// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Passes.h"

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_FOLDDISPATCHOUTPUTSTAGINGPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

// Fold:
//   %tmp = bufferization.alloc_tensor() copy(%src) {memory_space = 0 : i64}
//   iree_tensor_ext.dispatch.tensor.store %tmp, %target, ...
// into:
//   iree_tensor_ext.dispatch.tensor.store %src, %target, ...
struct FoldOutputStagingIntoDispatchStore final
    : public mlir::OpRewritePattern<
          mlir::iree_compiler::IREE::TensorExt::DispatchTensorStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::iree_compiler::IREE::TensorExt::DispatchTensorStoreOp storeOp,
      mlir::PatternRewriter &rewriter) const override {
    auto allocTensorOp =
        storeOp.getValue().getDefiningOp<mlir::bufferization::AllocTensorOp>();
    if (!allocTensorOp || !allocTensorOp.getCopy()) {
      return mlir::failure();
    }

    auto memorySpace = allocTensorOp.getMemorySpace();
    if (!memorySpace.has_value()) {
      return mlir::failure();
    }
    auto memorySpaceAttr = llvm::dyn_cast<mlir::IntegerAttr>(*memorySpace);
    if (!memorySpaceAttr || memorySpaceAttr.getInt() != 0) {
      return mlir::failure();
    }

    mlir::Value copySource = allocTensorOp.getCopy();
    if (copySource.getType() != storeOp.getValue().getType()) {
      return mlir::failure();
    }

    rewriter.modifyOpInPlace(
        storeOp, [&] { storeOp.getValueMutable().assign(copySource); });
    if (allocTensorOp->use_empty()) {
      rewriter.eraseOp(allocTensorOp);
    }
    return mlir::success();
  }
};

struct FoldDispatchOutputStagingPass final
    : public impl::FoldDispatchOutputStagingPassBase<
          FoldDispatchOutputStagingPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry
        .insert<mlir::bufferization::BufferizationDialect,
                mlir::iree_compiler::IREE::TensorExt::IREETensorExtDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.insert<FoldOutputStagingIntoDispatchStore>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::hexagon::codegen
