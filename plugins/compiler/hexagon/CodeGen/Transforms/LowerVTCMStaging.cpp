// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Passes.h"

#include "hexagon/CodeGen/IR/HexagonDialect.h"
#include "hexagon/CodeGen/IR/HexagonOps.h"
#include "hexagon/Common/Common.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-hexagon-lower-vtcm-staging"

namespace mlir::iree_compiler::hexagon::codegen {
namespace {

namespace IREEHexagon = mlir::iree_compiler::IREE::Hexagon;

#define GEN_PASS_DEF_HEXAGONLOWERVTCMSTAGINGPASS
#include "hexagon/CodeGen/Passes.h.inc"

struct LowerStageToVTCMPattern
    : public OpRewritePattern<IREEHexagon::StageToVTCMOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREEHexagon::StageToVTCMOp copyOp,
                                PatternRewriter &rewriter) const override {
    auto tensorType = cast<RankedTensorType>(copyOp.getResult().getType());
    // alloc_tensor with a copy source does NOT take explicit dynamic-size
    // operands: the verifier rejects them, and the shape is inferred from the
    // copy source instead.  This is safe even when tensorType is dynamic.
    auto allocOp = bufferization::AllocTensorOp::create(
        rewriter, copyOp.getLoc(), tensorType,
        /*dynamicSizes=*/ValueRange{}, copyOp.getSource());
    allocOp.setMemorySpaceAttr(
        rewriter.getI64IntegerAttr(::mlir::hexagon::VTCM_ADDRESS_SPACE));
    rewriter.replaceOp(copyOp, allocOp.getResult());
    return success();
  }
};

struct LowerVTCMEmptyPattern
    : public OpRewritePattern<IREEHexagon::VTCMEmptyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREEHexagon::VTCMEmptyOp allocOp,
                                PatternRewriter &rewriter) const override {
    auto tensorType = cast<RankedTensorType>(allocOp.getResult().getType());
    // Pass the dynamic sizes carried by the op through to alloc_tensor.
    // The resulting dynamic alloc_tensor is later staticized by
    // HexagonPadDynamicVTCMAllocPass (ValueBoundsConstraintSet upper bound)
    // and hoisted outside loops by HoistStaticallyBoundAllocationsPass.
    auto loweredAllocOp = bufferization::AllocTensorOp::create(
        rewriter, allocOp.getLoc(), tensorType, allocOp.getDynamicSizes());
    loweredAllocOp.setMemorySpaceAttr(
        rewriter.getI64IntegerAttr(::mlir::hexagon::VTCM_ADDRESS_SPACE));
    rewriter.replaceOp(allocOp, loweredAllocOp.getResult());
    return success();
  }
};

struct HexagonLowerVTCMStagingPass
    : impl::HexagonLowerVTCMStagingPassBase<HexagonLowerVTCMStagingPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREEHexagon::IREEHexagonDialect,
                bufferization::BufferizationDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<LowerStageToVTCMPattern, LowerVTCMEmptyPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      getOperation().emitOpError(
          "failed to lower iree_hexagon VTCM staging ops");
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createHexagonLowerVTCMStagingPass() {
  return std::make_unique<HexagonLowerVTCMStagingPass>();
}

} // namespace mlir::iree_compiler::hexagon::codegen
