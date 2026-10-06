// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Derived from Codegen/LLVMCPU/LLVMCPUVectorTransposeLowering.cpp at IREE
// revision a45adeaa6115e446c898e6eb21fb6edc0e65ddc4.

#include "hexagon/CodeGen/Passes.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-hexagon-vector-transpose-lowering"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONVECTORTRANSPOSELOWERINGPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

static bool has16x16Transpose(mlir::FunctionOpInterface funcOp) {
  bool res = false;
  funcOp.walk([&](vector::TransposeOp op) {
    auto srcGtOneDims = isTranspose2DSlice(op);
    if (failed(srcGtOneDims)) {
      return WalkResult::advance();
    }
    VectorType srcType = op.getSourceVectorType();
    int64_t m = srcType.getDimSize(std::get<0>(srcGtOneDims.value()));
    int64_t n = srcType.getDimSize(std::get<1>(srcGtOneDims.value()));
    if (m == 16 && n == 16) {
      res = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return res;
}

class HexagonVectorTransposeLoweringPass
    : public impl::HexagonVectorTransposeLoweringPassBase<
          HexagonVectorTransposeLoweringPass> {
public:
  using impl::HexagonVectorTransposeLoweringPassBase<
      HexagonVectorTransposeLoweringPass>::
      HexagonVectorTransposeLoweringPassBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void HexagonVectorTransposeLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  auto vectorTransformOptions =
      vector::VectorTransformsOptions().setVectorTransposeLowering(
          vector::VectorTransposeLowering::Shuffle1D);
  if (has16x16Transpose(funcOp)) {
    vectorTransformOptions.setVectorTransposeLowering(
        vector::VectorTransposeLowering::Shuffle16x16);
  }

  constexpr unsigned kNarrowTypeEmulationBenefit = 20;

  RewritePatternSet patterns(ctx);
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  vector::populateVectorTransposeLoweringPatterns(
      patterns, vectorTransformOptions.vectorTransposeLowering);
  vector::populateVectorTransposeNarrowTypeRewritePatterns(
      patterns, kNarrowTypeEmulationBenefit);

  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}
} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen
