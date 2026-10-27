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
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

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

/// Returns the alignment in bytes `allocOp` requests for its buffer, 0 if none.
uint64_t getAlignment(Operation *allocOp) {
  return llvm::TypeSwitch<Operation *, uint64_t>(allocOp)
      .Case<memref::AllocOp, memref::AllocaOp>(
          [](auto op) { return op.getAlignment().value_or(0); })
      .Case([](hexagonmem::AllocOp op) { return op.getAlignment(); })
      .Default(uint64_t{0});
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

  // The slice of chunk i starts i * sizeof(T) bytes into the wide allocation,
  // so every slice keeps the alignment only if sizeof(T) is a multiple of it.
  // Otherwise we need to add padding
  uint64_t alignment = getAlignment(allocOp);
  uint64_t elementBytes =
      DataLayout::closest(allocOp).getTypeSize(type.getElementType());
  uint64_t sliceBytes = type.getNumElements() * elementBytes;
  uint64_t paddedBytes =
      alignment ? llvm::alignTo(sliceBytes, alignment) : sliceBytes;
  bool pad = paddedBytes != sliceBytes;
  if (pad && paddedBytes % elementBytes != 0)
    return allocOp->emitOpError("inside a thread body can not be padded to ")
           << alignment << " bytes per thread with elements of " << elementBytes
           << " bytes";

  // add num chunks to the shape
  int64_t chunkElements = paddedBytes / elementBytes;
  SmallVector<int64_t> wideShape = {numChunks};
  llvm::append_range(wideShape, type.getShape());
  SmallVector<int64_t> allocShape = wideShape;
  if (pad)
    allocShape = {numChunks, chunkElements};
  auto wideType =
      MemRefType::get(allocShape, type.getElementType(),
                      MemRefLayoutAttrInterface(), type.getMemorySpace());
  MemRefType viewType = wideType;
  if (pad) {
    SmallVector<int64_t> viewStrides = {chunkElements};
    llvm::append_range(viewStrides, type.getStridesAndOffset().first);
    viewType = MemRefType::get(
        wideShape, type.getElementType(),
        StridedLayoutAttr::get(type.getContext(), /*offset=*/0, viewStrides),
        type.getMemorySpace());
  }

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
      type.getShape(), viewType, staticOffsets, staticSizes, staticStrides);
  if (failed(canReplaceMemrefUsesAndPropagateType(buffer, sliceType)))
    return allocOp->emitOpError("inside a thread body has a use that can not "
                                "take a slice of a per-thread allocation");

  // One wide allocation for all chunks
  rewriter.setInsertionPoint(forallOp);
  Operation *wideAllocOp = rewriter.clone(*allocOp);
  wideAllocOp->getResult(0).setType(wideType);
  Value wideBuffer = wideAllocOp->getResult(0);
  Value view = wideBuffer;
  if (pad) {
    SmallVector<OpFoldResult> viewSizes =
        getAsIndexOpFoldResult(rewriter.getContext(), viewType.getShape());
    SmallVector<OpFoldResult> viewStrides = getAsIndexOpFoldResult(
        rewriter.getContext(), viewType.getStridesAndOffset().first);
    view = memref::ReinterpretCastOp::create(
        rewriter, loc, viewType, wideBuffer, rewriter.getIndexAttr(0),
        viewSizes, viewStrides);
  }

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
  Value slice = memref::SubViewOp::create(rewriter, loc, sliceType, view,
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
