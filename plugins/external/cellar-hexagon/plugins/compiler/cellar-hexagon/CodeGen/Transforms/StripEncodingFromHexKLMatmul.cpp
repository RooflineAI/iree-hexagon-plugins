// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/CodeGen/Passes.h"

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {

#define GEN_PASS_DEF_STRIPENCODINGFROMHEXKLMATMULPASS
#include "cellar-hexagon/CodeGen/Passes.h.inc"

namespace {

static bool hasEncoding(mlir::RankedTensorType type) {
  return static_cast<bool>(type.getEncoding());
}

struct StripEncodingFromHexKLMatmulPattern final
    : public mlir::OpRewritePattern<mlir::hexkl::MatmulOp> {
  using mlir::OpRewritePattern<mlir::hexkl::MatmulOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hexkl::MatmulOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1) {
      return mlir::failure();
    }

    auto lhsType =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getLhs().getType());
    auto rhsType =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getRhs().getType());
    auto outsType =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getOuts().getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!lhsType || !rhsType || !outsType || !resultType) {
      return mlir::failure();
    }

    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
        !outsType.hasStaticShape() || !resultType.hasStaticShape()) {
      return mlir::failure();
    }

    if (!hasEncoding(lhsType) && !hasEncoding(rhsType) &&
        !hasEncoding(outsType) && !hasEncoding(resultType)) {
      return mlir::failure();
    }

    // We are going to revert the encoding normally done by IREE because data
    // layout rearrangement will be managed by the lowering of the linalg.matmul
    // operation in the hexagon-mlir pipeline. This aims taking advantage of the
    // VTCM and hexagons vector unit, which is what that lowering ends up doing.
    // Note that this is undoing the encoding, and should be optimized away. The
    // risk that they are not optimized would result in useless pack/unpack
    // operations. This has the advantage of still using iree's encodings for
    // non matmul operations if they happen in subsequent traces.

    // TODO: Nevertheless, this is more a fix than a permanent solution and
    // should be deeply discussed later down the line for a more solid solution.
    // This is currently only a proof of concept trying to hack something
    // together.
    auto stripEncoding = [&](mlir::Value value,
                             mlir::RankedTensorType type) -> mlir::Value {
      if (!hasEncoding(type)) {
        return value;
      }

      mlir::RankedTensorType unencodedType = type.dropEncoding();
      auto unsetOp = iree_compiler::IREE::Encoding::UnsetEncodingOp::create(
          rewriter, op.getLoc(), unencodedType, value, mlir::ValueRange{},
          mlir::ValueRange{});
      return unsetOp.getResult();
    };

    mlir::Value lhs = stripEncoding(op.getLhs(), lhsType);
    mlir::Value rhs = stripEncoding(op.getRhs(), rhsType);
    mlir::Value outs = stripEncoding(op.getOuts(), outsType);

    mlir::RankedTensorType unencodedResultType = resultType.dropEncoding();
    auto newMatmul = mlir::hexkl::MatmulOp::create(
        rewriter, op.getLoc(), unencodedResultType, lhs, rhs, outs);
    mlir::Value replacement = newMatmul->getResult(0);

    if (hasEncoding(resultType)) {
      auto setOp = iree_compiler::IREE::Encoding::SetEncodingOp::create(
          rewriter, op.getLoc(), resultType, replacement, mlir::ValueRange{});
      replacement = setOp.getResult();
    }

    rewriter.replaceOp(op, replacement);
    return mlir::success();
  }
};

struct StripEncodingFromHexKLMatmulPass
    : public impl::StripEncodingFromHexKLMatmulPassBase<
          StripEncodingFromHexKLMatmulPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::hexkl::HexKLDialect, tensor::TensorDialect,
                    iree_compiler::IREE::Encoding::IREEEncodingDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<StripEncodingFromHexKLMatmulPattern>(&getContext());

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::cellar_hexagon::codegen
