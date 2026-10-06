// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Derived from Codegen/LLVMCPU/LLVMCPUVectorShapeCastLowering.cpp at IREE
// revision a45adeaa6115e446c898e6eb21fb6edc0e65ddc4.

#include "hexagon/CodeGen/Passes.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-hexagon-vector-shape-cast-lowering"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONVECTORSHAPECASTLOWERINGPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {
class HexagonVectorShapeCastLoweringPass
    : public impl::HexagonVectorShapeCastLoweringPassBase<
          HexagonVectorShapeCastLoweringPass> {
public:
  using impl::HexagonVectorShapeCastLoweringPassBase<
      HexagonVectorShapeCastLoweringPass>::
      HexagonVectorShapeCastLoweringPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void HexagonVectorShapeCastLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  RewritePatternSet patterns(ctx);
  vector::populateVectorShapeCastLoweringPatterns(patterns);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}
} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen
