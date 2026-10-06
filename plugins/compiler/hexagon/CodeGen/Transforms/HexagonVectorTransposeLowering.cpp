// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Derived from Codegen/LLVMCPU/LLVMCPUVectorTransposeLowering.cpp at IREE
// revision a45adeaa6115e446c898e6eb21fb6edc0e65ddc4, without the AVX2 patterns
// and the AVX-512 16x16 shuffle network.

#include "hexagon/CodeGen/Passes.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-hexagon-vector-transpose-lowering"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONVECTORTRANSPOSELOWERINGPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

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

  constexpr unsigned kNarrowTypeEmulationBenefit = 20;

  RewritePatternSet patterns(ctx);
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  vector::populateVectorTransposeLoweringPatterns(
      patterns, vector::VectorTransposeLowering::Shuffle1D);
  vector::populateVectorTransposeNarrowTypeRewritePatterns(
      patterns, kNarrowTypeEmulationBenefit);

  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}
} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen
