// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONFORMASYNCTHREADSPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

/// Rewrites
///
///   scf.forall (%iv) = (%lb) to (%ub) step (%step) { body(%iv) }
///
/// into
///
///   %group = async.runtime.create_group N
///   scf.for %iv = %lb to %ub step %step {
///     %token = async.execute { body(%iv) }
///     async.add_to_group %token, %group
///   }
///   async.await_all %group
///
/// with N = ceildiv(%ub - %lb, %step) chunks. Every chunk is a task of the
/// thread pool; the dispatching thread computes its share by joining the pool
/// in the await (hexagon_runtime_thread_await_all).
LogicalResult formAsyncThreads(RewriterBase &rewriter, scf::ForallOp forallOp) {
  if (forallOp.getRank() != 1)
    return forallOp.emitOpError("expected a one-dimensional thread forall");
  if (!forallOp.getOutputs().empty())
    return forallOp.emitOpError("expected a bufferized thread forall");

  Location loc = forallOp.getLoc();
  rewriter.setInsertionPoint(forallOp);
  Value lb = getValueOrCreateConstantIndexOp(rewriter, loc,
                                             forallOp.getMixedLowerBound()[0]);
  Value ub = getValueOrCreateConstantIndexOp(rewriter, loc,
                                             forallOp.getMixedUpperBound()[0]);
  Value step = getValueOrCreateConstantIndexOp(rewriter, loc,
                                               forallOp.getMixedStep()[0]);
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value range = rewriter.createOrFold<arith::MaxSIOp>(
      loc, rewriter.createOrFold<arith::SubIOp>(loc, ub, lb), zero);
  Value numChunks = rewriter.createOrFold<arith::CeilDivSIOp>(loc, range, step);

  // async.create_group is Pure, so LICM would hoist them out of the VTCM tile
  // loop. we need to use async.runtime.create_group instead to prevent this
  Value group = async::RuntimeCreateGroupOp::create(
      rewriter, loc, rewriter.getType<async::GroupType>(), numChunks);
  auto forOp = scf::ForOp::create(rewriter, loc, lb, ub, step);
  rewriter.setInsertionPoint(forOp.getBody()->getTerminator());
  auto executeOp = async::ExecuteOp::create(rewriter, loc, TypeRange{},
                                            ValueRange{}, ValueRange{});
  async::AddToGroupOp::create(rewriter, loc, executeOp.getToken(), group);
  rewriter.setInsertionPointAfter(forOp);
  async::AwaitAllOp::create(rewriter, loc, group);

  // Move the body into the async.execute region.
  rewriter.eraseOp(forallOp.getBody()->getTerminator());
  rewriter.inlineBlockBefore(
      forallOp.getBody(), executeOp.getBody(),
      executeOp.getBody()->getTerminator()->getIterator(),
      forOp.getInductionVar());
  rewriter.eraseOp(forallOp);
  return success();
}

struct HexagonFormAsyncThreadsPass
    : impl::HexagonFormAsyncThreadsPassBase<HexagonFormAsyncThreadsPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<scf::ForallOp> forallOps;
    getOperation()->walk([&](scf::ForallOp forallOp) {
      if (isThreadForall(forallOp))
        forallOps.push_back(forallOp);
    });
    IRRewriter rewriter(&getContext());
    for (scf::ForallOp forallOp : forallOps) {
      if (failed(formAsyncThreads(rewriter, forallOp)))
        return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen
