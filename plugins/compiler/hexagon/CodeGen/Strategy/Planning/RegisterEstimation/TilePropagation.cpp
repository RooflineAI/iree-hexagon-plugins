// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Plan section 4.2: the tile coordinate system and one anchor-to-loops map
// per op, found by walking producer/consumer edges out from the anchor.

#include "TilePropagation.h"

#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

using FailureReporter = llvm::function_ref<LogicalResult(const Twine &)>;

/// The propagation's working state: one expression per loop of each op,
/// written in terms of anchor dims
class Propagator {
public:
  Propagator(ArrayRef<linalg::LinalgOp> ops, OpIdx anchorIndex,
             FailureReporter reportFailure)
      : ops(ops), anchorIndex(anchorIndex), reportFailure(reportFailure),
        loopExprs(ops.size()) {}

  FailureOr<TilePropagationResult> run();

private:
  MLIRContext *getContext() const { return ops[anchorIndex]->getContext(); }

  /// Appends an anchor dim for a loop nothing constrains, pinned to its full
  /// extent (UnconstrainedDimPolicy::FullExtent).
  AffineExpr addPinnedDim(linalg::LinalgOp op, unsigned loop);

  /// The expression a tensor `consumer` reads alongside an op already in the
  /// nest gives `loop`, or null if no such tensor addresses it.
  AffineExpr getSharedInputExpr(OpIdx consumerIndex, unsigned loop);

  /// Records `exprs` as `opIndex`'s loop mapping
  /// or fails if the op was already reached with a different one.
  LogicalResult assign(OpIdx opIndex, ArrayRef<AffineExpr> exprs);

  /// Whichever end is already mapped fixes the tensor's tile; the other is
  /// derived from it. if both are already mapped, they have to agree.
  LogicalResult linkEdge(OpIdx producerIndex, OpResult result,
                         OpIdx consumerIndex, OpOperand *operand);
  LogicalResult deriveProducer(OpIdx producerIndex, AffineMap output,
                               ArrayRef<AffineExpr> tile);
  LogicalResult deriveConsumer(OpIdx consumerIndex, AffineMap read,
                               ArrayRef<AffineExpr> tile);

  ArrayRef<linalg::LinalgOp> ops;
  OpIdx anchorIndex;
  FailureReporter reportFailure;

  SmallVector<AnchorDim> anchorDims;
  SmallVector<SmallVector<AffineExpr>> loopExprs;
  SmallVector<OpIdx> worklist;
};

AffineExpr Propagator::addPinnedDim(linalg::LinalgOp op, unsigned loop) {
  int64_t extent = op.getStaticLoopRanges()[loop];
  anchorDims.push_back(AnchorDim{op.getIteratorTypesArray()[loop], extent,
                                 op.getOperation(), loop});
  return getAffineDimExpr(anchorDims.size() - 1, getContext());
}

/// Both ops read the same tile of a tensor they share, so the loop addressing
/// one of its dims takes the other op's expression for that dim.
AffineExpr Propagator::getSharedInputExpr(OpIdx consumerIndex, unsigned loop) {
  linalg::LinalgOp consumer = ops[consumerIndex];
  for (OpOperand *operand : consumer.getDpsInputOperands()) {
    AffineMap read = consumer.getMatchingIndexingMap(operand);
    for (auto [position, result] : llvm::enumerate(read.getResults())) {
      auto dim = dyn_cast<AffineDimExpr>(result);
      if (!dim || dim.getPosition() != loop)
        continue;
      for (OpIdx otherIndex = 0, e = ops.size(); otherIndex < e; ++otherIndex) {
        if (otherIndex == consumerIndex || loopExprs[otherIndex].empty())
          continue;
        linalg::LinalgOp other = ops[otherIndex];
        for (OpOperand *otherOperand : other.getDpsInputOperands()) {
          if (otherOperand->get() != operand->get())
            continue;
          return other.getMatchingIndexingMap(otherOperand)
              .getResult(position)
              .replaceDims(loopExprs[otherIndex]);
        }
      }
    }
  }
  return AffineExpr();
}

LogicalResult Propagator::assign(OpIdx opIndex, ArrayRef<AffineExpr> exprs) {
  if (!loopExprs[opIndex].empty()) {
    // already present: check if both ways of reaching it agree
    if (!llvm::equal(loopExprs[opIndex], exprs))
      return reportFailure(
          "op '" + ops[opIndex]->getName().getStringRef() +
          "' is reached with two different tile mappings; mark the edge "
          "materialized to break the fusion");
    return success();
  }
  // record new AffineExpressions
  loopExprs[opIndex].assign(exprs.begin(), exprs.end());
  worklist.push_back(opIndex); // need to analyze everything derived from it
  return success();
}

/// The producer writes the tensor `output`
LogicalResult Propagator::deriveProducer(OpIdx producerIndex, AffineMap output,
                                         ArrayRef<AffineExpr> tile) {
  linalg::LinalgOp producer = ops[producerIndex];
  if (!output.isPermutation())
    return reportFailure("op '" + producer->getName().getStringRef() +
                         "' does not write its result through a permutation, "
                         "so its loops cannot be derived from its consumer");

  SmallVector<AffineExpr> exprs(producer.getNumLoops());
  for (auto [position, expr] : llvm::enumerate(output.getResults()))
    exprs[cast<AffineDimExpr>(expr).getPosition()] = tile[position];
  return assign(producerIndex, exprs);
}

/// The consumer reads the tensor through `read`.
LogicalResult Propagator::deriveConsumer(OpIdx consumerIndex, AffineMap read,
                                         ArrayRef<AffineExpr> tile) {
  linalg::LinalgOp consumer = ops[consumerIndex];
  SmallVector<AffineExpr> exprs(consumer.getNumLoops());
  // map dims directly visible from producer tile
  for (auto [position, expr] : llvm::enumerate(read.getResults())) {
    auto dim = dyn_cast<AffineDimExpr>(expr);
    if (!dim)
      return reportFailure("op '" + consumer->getName().getStringRef() +
                           "' reads a fused value through a non-trivial index "
                           "expression");
    if (exprs[dim.getPosition()])
      return reportFailure("op '" + consumer->getName().getStringRef() +
                           "' reads a fused value with a repeated loop");
    exprs[dim.getPosition()] = tile[position];
  }
  // look to the other operands to get the dim
  for (auto [loop, expr] : llvm::enumerate(exprs))
    if (!expr)
      expr = getSharedInputExpr(consumerIndex, loop);
  // otherwise: cannot derive from the input, independent of the anchor
  // will not tile this dim
  for (auto [loop, expr] : llvm::enumerate(exprs))
    if (!expr)
      expr = addPinnedDim(consumer, loop);
  return assign(consumerIndex, exprs);
}

LogicalResult Propagator::linkEdge(OpIdx producerIndex, OpResult result,
                                   OpIdx consumerIndex, OpOperand *operand) {
  bool haveProducer = !loopExprs[producerIndex].empty();
  bool haveConsumer = !loopExprs[consumerIndex].empty();
  // noting i known: nothing to do
  if (!haveProducer && !haveConsumer)
    return success();

  linalg::LinalgOp producer = ops[producerIndex];
  linalg::LinalgOp consumer = ops[consumerIndex];
  AffineMap output = producer.getIndexingMapMatchingResult(result);
  AffineMap read = consumer.getMatchingIndexingMap(operand);

  // this gives us the Affine Expression in "Anchor coordinates" instead of this
  // ops ones
  auto substitute = [](AffineMap map, ArrayRef<AffineExpr> loops) {
    SmallVector<AffineExpr> tile;
    for (AffineExpr expr : map.getResults())
      tile.push_back(expr.replaceDims(loops));
    return tile;
  };

  if (!haveConsumer)
    return deriveConsumer(consumerIndex, read,
                          substitute(output, loopExprs[producerIndex]));
  SmallVector<AffineExpr> wanted = substitute(read, loopExprs[consumerIndex]);
  if (!haveProducer)
    return deriveProducer(producerIndex, output, wanted);

  // Both ends are fixed: they have to describe the same tile of the tensor.
  if (!llvm::equal(substitute(output, loopExprs[producerIndex]), wanted))
    return reportFailure(
        "value produced by op '" + producer->getName().getStringRef() +
        "' is read with two different tile mappings; mark the edge "
        "materialized to break the fusion");
  return success();
}

FailureOr<TilePropagationResult> Propagator::run() {
  linalg::LinalgOp anchor = ops[anchorIndex];

  // The anchor's own loops are the tile vector's coordinate system.
  SmallVector<int64_t> anchorRanges = anchor.getStaticLoopRanges();
  SmallVector<utils::IteratorType> iterators = anchor.getIteratorTypesArray();
  SmallVector<AffineExpr> anchorExprs;
  for (auto [loop, extent, iterator] :
       llvm::enumerate(anchorRanges, iterators)) {
    if (ShapedType::isDynamic(extent)) {
      (void)reportFailure("anchor op has a dynamic loop extent");
      return failure();
    }
    anchorExprs.push_back(getAffineDimExpr(anchorDims.size(), getContext()));
    anchorDims.push_back(AnchorDim{iterator, extent, anchor.getOperation(),
                                   static_cast<unsigned>(loop)});
  }
  // initialize the worklist with the ancor
  if (failed(assign(anchorIndex, anchorExprs)))
    return failure();

  // Which op produces each tensor the dispatch's ops hand each other, and
  // which index each op has.
  llvm::DenseMap<Value, OpIdx> producerOf;
  llvm::DenseMap<Operation *, OpIdx> opIndexOf;
  for (auto [index, op] : llvm::enumerate(ops)) {
    opIndexOf[op] = index;
    for (OpResult result : op->getResults())
      producerOf[result] = index;
  }

  while (!worklist.empty()) {
    OpIdx current = worklist.pop_back_val();
    linalg::LinalgOp op = ops[current];
    // find producers
    for (OpOperand &operand : op->getOpOperands()) {
      auto producer = producerOf.find(operand.get());
      if (producer == producerOf.end())
        continue;
      if (failed(linkEdge(producer->second, cast<OpResult>(operand.get()),
                          current, &operand)))
        return failure();
    }
    // find consumers
    for (OpResult result : op->getResults()) {
      for (OpOperand &use : result.getUses()) {
        auto consumer = opIndexOf.find(use.getOwner());
        if (consumer == opIndexOf.end())
          continue;
        if (failed(linkEdge(current, result, consumer->second, &use)))
          return failure();
      }
    }
  }

  TilePropagationResult result;

  // check for validity
  for (auto [index, exprs] : llvm::enumerate(loopExprs)) {
    if (exprs.empty()) {
      (void)reportFailure("op '" + ops[index]->getName().getStringRef() +
                          "' is not reachable from the anchor, so it is not "
                          "part of the anchor's nest");
      return failure();
    }
    // has a reduction dim, that the ancor sets as a parallel dim?
    linalg::LinalgOp op = ops[index];
    for (auto [loop, iterator] : llvm::enumerate(op.getIteratorTypesArray())) {
      auto dim = dyn_cast<AffineDimExpr>(exprs[loop]);
      if (iterator == utils::IteratorType::reduction && dim &&
          anchorDims[dim.getPosition()].iteratorType !=
              utils::IteratorType::reduction) {
        (void)reportFailure("op '" + op->getName().getStringRef() +
                            "' reduces over anchor dim d" +
                            Twine(dim.getPosition()) +
                            ", which the anchor iterates in parallel; anchor "
                            "the dispatch at the reduction");
        return failure();
      }
    }
  }
  result.anchorDims = std::move(anchorDims);
  // store in result, add 0 for pinned dims that dont depend on the anchor
  for (ArrayRef<AffineExpr> exprs : loopExprs) {
    result.anchorToOpLoops.push_back(AffineMap::get(
        result.anchorDims.size(), /*symbolCount=*/0, exprs, getContext()));
  }
  return result;
}

} // namespace

FailureOr<TilePropagationResult>
propagateTiles(ArrayRef<linalg::LinalgOp> ops, OpIdx anchorIndex,
               llvm::function_ref<LogicalResult(const Twine &)> reportFailure) {
  assert(anchorIndex < ops.size() && "anchor must be one of the ops");
  return Propagator(ops, anchorIndex, reportFailure).run();
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning
