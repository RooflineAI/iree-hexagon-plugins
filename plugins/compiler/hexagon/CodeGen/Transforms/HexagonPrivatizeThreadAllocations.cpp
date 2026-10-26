// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Passes.h"

#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONPRIVATIZETHREADALLOCATIONSPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

bool isAllocation(Operation *op) {
  return isa<memref::AllocOp, memref::AllocaOp, hexagonmem::AllocOp>(op);
}

bool isDeallocation(Operation *op) {
  return isa<memref::DeallocOp, hexagonmem::DeallocOp>(op);
}

/// Replaces `allocOp` inside the body of `forallOp` (`numChunks` chunks) by a
/// slice of one allocation with a leading dimension of `numChunks`, created
/// right before the forall and indexed by the chunk the body computes.
LogicalResult privatize(RewriterBase &rewriter, scf::ForallOp forallOp,
                        int64_t numChunks, Operation *allocOp) {
  Value buffer = allocOp->getResult(0);
  auto type = dyn_cast<MemRefType>(buffer.getType());
  if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity() ||
      allocOp->getNumOperands() != 0)
    return allocOp->emitOpError(
        "inside a thread body must have a static shape and identity layout, "
        "so that each thread can get its own slice of one allocation");

  // add num chunks to the shape
  SmallVector<int64_t> wideShape = {numChunks};
  llvm::append_range(wideShape, type.getShape());
  auto wideType =
      MemRefType::get(wideShape, type.getElementType(),
                      MemRefLayoutAttrInterface(), type.getMemorySpace());

  // The slice of the chunk, rank-reduced to the original shape.
  Location loc = allocOp->getLoc();
  SmallVector<OpFoldResult> offsets(wideShape.size(), rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1)};
  for (int64_t size : type.getShape())
    sizes.push_back(rewriter.getIndexAttr(size));
  SmallVector<OpFoldResult> strides(wideShape.size(), rewriter.getIndexAttr(1));
  SmallVector<int64_t> staticOffsets(wideShape.size(), 0);
  staticOffsets.front() = ShapedType::kDynamic;
  SmallVector<int64_t> staticSizes = {1};
  llvm::append_range(staticSizes, type.getShape());
  SmallVector<int64_t> staticStrides(wideShape.size(), 1);
  auto sliceType = memref::SubViewOp::inferRankReducedResultType(
      type.getShape(), wideType, staticOffsets, staticSizes, staticStrides);
  if (failed(canReplaceMemrefUsesAndPropagateType(buffer, sliceType)))
    return allocOp->emitOpError("inside a thread body has a use that can not "
                                "take a slice of a per-thread allocation");

  // One wide allocation for all chunks
  rewriter.setInsertionPoint(forallOp);
  Operation *wideAllocOp = rewriter.clone(*allocOp);
  wideAllocOp->getResult(0).setType(wideType);
  Value wideBuffer = wideAllocOp->getResult(0);

  // handle dealloc: replace with one large dealloc after the loop
  SmallVector<Operation *> deallocOps;
  for (Operation *user : buffer.getUsers()) {
    if (isDeallocation(user))
      deallocOps.push_back(user);
  }
  if (!deallocOps.empty()) {
    rewriter.setInsertionPointAfter(forallOp);
    Operation *wideDeallocOp = rewriter.clone(*deallocOps.front());
    wideDeallocOp->setOperand(0, wideBuffer);
  }
  for (Operation *deallocOp : deallocOps)
    rewriter.eraseOp(deallocOp);

  // rewrite users in the loop
  // Chunk index (iv - lb) / step.
  rewriter.setInsertionPoint(allocOp);
  Value lb = getValueOrCreateConstantIndexOp(rewriter, loc,
                                             forallOp.getMixedLowerBound()[0]);
  Value step = getValueOrCreateConstantIndexOp(rewriter, loc,
                                               forallOp.getMixedStep()[0]);
  Value chunk = rewriter.createOrFold<arith::DivUIOp>(
      loc,
      rewriter.createOrFold<arith::SubIOp>(loc, forallOp.getInductionVar(0),
                                           lb),
      step);
  offsets.front() = chunk;
  Value slice = memref::SubViewOp::create(rewriter, loc, sliceType, wideBuffer,
                                          offsets, sizes, strides);
  replaceMemrefUsesAndPropagateType(rewriter, loc, buffer, slice);
  rewriter.eraseOp(allocOp);
  return success();
}

struct HexagonPrivatizeThreadAllocationsPass
    : impl::HexagonPrivatizeThreadAllocationsPassBase<
          HexagonPrivatizeThreadAllocationsPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<scf::ForallOp> forallOps;
    getOperation()->walk([&](scf::ForallOp forallOp) {
      if (isThreadForall(forallOp))
        forallOps.push_back(forallOp);
    });

    IRRewriter rewriter(&getContext());
    for (scf::ForallOp forallOp : forallOps) {
      SmallVector<Operation *> allocOps;
      forallOp.getBody()->walk([&](Operation *op) {
        if (isAllocation(op))
          allocOps.push_back(op);
      });
      if (allocOps.empty())
        continue;

      std::optional<int64_t> lb =
          getConstantIntValue(forallOp.getMixedLowerBound()[0]);
      std::optional<int64_t> ub =
          getConstantIntValue(forallOp.getMixedUpperBound()[0]);
      std::optional<int64_t> step =
          getConstantIntValue(forallOp.getMixedStep()[0]);
      if (forallOp.getRank() != 1 || !lb || !ub || !step || *step <= 0) {
        forallOp.emitOpError("with allocations in its body must be "
                             "one-dimensional with static bounds");
        return signalPassFailure();
      }
      int64_t numChunks =
          llvm::divideCeil(std::max<int64_t>(*ub - *lb, 0), *step);
      for (Operation *allocOp : allocOps) {
        if (failed(privatize(rewriter, forallOp, numChunks, allocOp)))
          return signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen
