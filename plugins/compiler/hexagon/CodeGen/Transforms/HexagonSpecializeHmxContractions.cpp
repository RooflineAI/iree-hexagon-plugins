// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Passes.h"
#include "hexagon/CodeGen/Strategy/Planning/Strategies/Strategy.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONSPECIALIZEHMXCONTRACTIONSPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

// Returns whether `op` is a named matmul that HMX lowering can take: a
// linalg.matmul or linalg.batch_matmul whose output is indexed by the leading
// (batch, M, N) loops in order. HexagonConvertMatmulToHmx requires a canonical
// MxN output and rejects anything else outright, so specializing to a named op
// with a transposed output would turn a slow compile into a failed one.
static bool isHmxConvertibleNamedMatmul(linalg::LinalgOp op) {
  if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op.getOperation()))
    return false;
  AffineMap outputMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  unsigned numLoops = outputMap.getNumDims();
  if (numLoops == 0)
    return false;
  // The named ops order their loops (batch..., M, N, K).
  SmallVector<unsigned> leadingLoops =
      llvm::to_vector(llvm::seq<unsigned>(0, numLoops - 1));
  return outputMap == AffineMap::getMultiDimMapWithTargets(
                          numLoops, leadingLoops, op.getContext());
}

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
      if (!isHmxConvertibleNamedMatmul(*named)) {
        rewriter.eraseOp(*named);
        continue;
      }
      rewriter.replaceOp(genericOp, (*named)->getResults());
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::hexagon::codegen
