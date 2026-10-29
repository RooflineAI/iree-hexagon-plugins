// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Passes.h"
#include "hexagon/CodeGen/Strategy/Planning/Strategies/Strategy.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONSPECIALIZEHMXCONTRACTIONSPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

class HexagonSpecializeHmxContractionsPass final
    : public impl::HexagonSpecializeHmxContractionsPassBase<
          HexagonSpecializeHmxContractionsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    SmallVector<linalg::GenericOp> candidates;
    getOperation().walk([&](linalg::GenericOp genericOp) {
      // An f32 body can express accumulation of f16 inputs. Specialization
      // preserves those cast semantics; HMX packing consumes the f16 tensors.
      if (planning::isHmxCandidateContraction(genericOp))
        candidates.push_back(genericOp);
    });

    IRRewriter rewriter(&getContext());
    for (linalg::GenericOp genericOp : candidates) {
      // Specialize a clone, so that a generic that does not become an
      // HMX-convertible named matmul is left untouched. Upstream
      // specialization falls back to linalg.contract, which HMX lowering
      // does not take either.
      rewriter.setInsertionPoint(genericOp);
      auto clone = cast<linalg::GenericOp>(rewriter.clone(*genericOp));
      FailureOr<linalg::LinalgOp> named =
          linalg::specializeGenericOp(rewriter, clone);
      if (failed(named)) {
        rewriter.eraseOp(clone);
        continue;
      }
      // A named op with a transposed output would be rejected outright by
      // HMX lowering, turning a slow compile into a failed one.
      if (!planning::isHmxLowerableMatmul(*named)) {
        rewriter.eraseOp(*named);
        continue;
      }
      rewriter.replaceOp(genericOp, (*named)->getResults());
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::hexagon::codegen
