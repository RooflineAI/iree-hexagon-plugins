// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "DispatchRegisterGraph.h"
#include "EstimatorConfig.h"
#include "TilePropagation.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"

#include <optional>

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// Ops that carry no computation are simply part of the dispatch's
/// plumbing. Anything else that is not a `linalg::LinalgOp` means the
/// dispatch is outside the supported subset.
///
/// A real (post-ABI-lowering) dispatch function has no tensor arguments: its
/// linalg ops' operands trace back to `hal.interface.binding.subspan` +
/// `iree_tensor_ext.dispatch.tensor.load`, and its results are consumed by
/// `iree_tensor_ext.dispatch.tensor.store`, instead of being block arguments
/// / a `return`. None of the three carry computation, and the existing
/// SSA-producer tracking in `liftBlockArgs`/`addBoundaries` already treats
/// whatever isn't a fused linalg result as a Load, and whatever a linalg
/// result feeds outside the fused chain as a Store, regardless of which
/// concrete op does the reading or writing - so admitting these three here is
/// enough to make the whole dispatch visible, without teaching those stages
/// anything new about IREE's ABI ops.
bool isIgnorableOp(Operation *op) {
  return op->hasTrait<OpTrait::IsTerminator>() ||
         isa<arith::ConstantOp, tensor::EmptyOp,
             IREE::HAL::InterfaceBindingSubspanOp,
             IREE::TensorExt::DispatchTensorLoadOp,
             IREE::TensorExt::DispatchTensorStoreOp>(op);
}

/// Collects the dispatch's linalg ops in program order, rejecting anything
/// else the function's body block holds (plan 4.1 step 1).
LogicalResult collectOps(DispatchRegisterGraph &graph) {
  Region &body = graph.dispatch.getFunctionBody();
  if (!body.hasOneBlock())
    return graph.fail("dispatch function does not have a single body block");

  for (Operation &op : body.front()) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(&op)) {
      // only accept static shapes
      for (int64_t extent : linalgOp.getStaticLoopRanges()) {
        if (ShapedType::isDynamic(extent))
          return graph.fail("op '" + linalgOp->getName().getStringRef() +
                            "' has a dynamic loop extent");
      }
      graph.ops.push_back(linalgOp);
      continue;
    }
    if (!isIgnorableOp(&op))
      return graph.fail("dispatch contains '" + op.getName().getStringRef() +
                        "', which does not implement LinalgOp");
  }
  if (graph.ops.empty())
    return graph.fail("dispatch contains no linalg op");
  return success();
}

/// Builder for one dispatch's node list. Holds the value -> node index map
/// that splicing and the rewrites need.
class NodeBuilder {
public:
  NodeBuilder(DispatchRegisterGraph &graph) : graph(graph) {}

  LogicalResult run();

private:
  /// Appends `node` and records it as the definition of `node.value`.
  unsigned addNode(Node node);

  /// The rank-0 map every invariant uses: its shape does not depend on the
  /// tile, so it costs one splatted register.
  AffineMap getScalarMap() const;
  AffineMap getOpLoopMap(unsigned opIndex) const;

  /// Node for a value read by a body op: either one lifted earlier, or a new
  /// `Invariant` for something defined outside the body.
  FailureOr<unsigned> getOperandNode(Value value, unsigned opIndex);

  /// The invariant holding the constants of `op`'s expansion, created on
  /// first use: every op of the same kind splats the same constants.
  unsigned getExpansionConstants(Operation &op, Type elementType,
                                 int64_t registers, unsigned opIndex);

  // these helpers return false on failure, and alter the NodeBuilder's State on
  // success
  LogicalResult buildNodeForOp(unsigned opIndex);
  LogicalResult addReduceNodes(unsigned opIndex);
  LogicalResult buildBlockArgsNodes(unsigned opIndex);

  LogicalResult buildOperandNode(linalg::LinalgOp op, unsigned opIndex,
                                 ArrayRef<AffineMap> outputShapeMaps,
                                 OpOperand *operand, bool isInput);
  LogicalResult buildNodesforBody(unsigned opIndex);
  LogicalResult addResultStoreNodes(unsigned opIndex);

  /// Node holding `source` in the layout `wanted`, inserting a `Shuffle`
  /// first if `source` is laid out differently.
  unsigned addRelayoutNode(unsigned source, AffineMap wanted, unsigned opIndex);

  DispatchRegisterGraph &graph;
  llvm::DenseMap<Value, unsigned> nodeOf;
  /// Per op: the node producing each of its results, in result order.
  SmallVector<SmallVector<unsigned>> resultNodes;
  /// Tensor result of an op in the list -> the node holding it.
  llvm::DenseMap<Value, unsigned> producedTensor;
  /// A materialized tensor's `Load`, shared by every consumer of it.
  llvm::DenseMap<Value, unsigned> materializedLoad;
  /// Per expanded op name: the invariant holding its constant table, shared
  /// by every op of that kind.
  llvm::StringMap<unsigned> expansionConstants;
};

unsigned NodeBuilder::addNode(Node node) {
  unsigned index = graph.nodes.size();
  if (node.value)
    nodeOf[node.value] = index;
  graph.nodes.push_back(std::move(node));
  return index;
}

AffineMap NodeBuilder::getScalarMap() const {
  return AffineMap::get(graph.anchorDims.size(), /*symbolCount=*/0,
                        graph.dispatch->getContext());
}

AffineMap NodeBuilder::getOpLoopMap(unsigned opIndex) const {
  return graph.anchorToOpLoops[opIndex];
}

FailureOr<unsigned> NodeBuilder::getOperandNode(Value value, unsigned opIndex) {
  auto found = nodeOf.find(value);
  if (found != nodeOf.end())
    return found->second;

  // Defined outside the body: a captured scalar, splatted once.
  Node node;
  node.kind = NodeKind::Invariant;
  node.value = value;
  node.op = value.getDefiningOp();
  node.owningOp = opIndex;
  node.shapeMap = getScalarMap();
  node.elementType = getElementTypeOrSelf(value.getType());
  node.regClass = classifyResultType(node.elementType);
  if (node.regClass == RegClass::None)
    return graph.fail("captured value of unsupported type in the body of op '" +
                      graph.ops[opIndex]->getName().getStringRef() + "'");
  return addNode(std::move(node));
}

unsigned NodeBuilder::getExpansionConstants(Operation &op, Type elementType,
                                            int64_t registers,
                                            unsigned opIndex) {
  StringRef name = op.getName().getStringRef();
  auto found = expansionConstants.find(name);
  if (found != expansionConstants.end())
    return found->second;
  Node node;
  node.kind = NodeKind::Invariant;
  node.op = &op;
  node.owningOp = opIndex;
  node.shapeMap = getScalarMap();
  node.elementType = elementType;
  node.regClass = RegClass::Vector;
  node.invariantRegisters = registers;
  unsigned index = addNode(std::move(node));
  expansionConstants[name] = index;
  return index;
}

/// The anchor dims `map`'s results address, in result order, or nothing when
/// some result is not a plain dim: a window or a broadcast index says nothing
/// about lane order.
static std::optional<SmallVector<unsigned>> getDimOrder(AffineMap map) {
  SmallVector<unsigned> order;
  for (AffineExpr result : map.getResults()) {
    auto dim = dyn_cast<AffineDimExpr>(result);
    if (!dim)
      return std::nullopt;
    order.push_back(dim.getPosition());
  }
  return order;
}

/// Whether two tiles lay their shared dims out in the same order. Dims only
/// one of them has are broadcast in the other and do not move.
static bool haveSameLayout(AffineMap lhs, AffineMap rhs) {
  std::optional<SmallVector<unsigned>> lhsOrder = getDimOrder(lhs);
  std::optional<SmallVector<unsigned>> rhsOrder = getDimOrder(rhs);
  if (!lhsOrder || !rhsOrder)
    return true;
  auto shared = [](ArrayRef<unsigned> order, ArrayRef<unsigned> other) {
    SmallVector<unsigned> kept;
    for (unsigned dim : order)
      if (llvm::is_contained(other, dim))
        kept.push_back(dim);
    return kept;
  };
  return shared(*lhsOrder, *rhsOrder) == shared(*rhsOrder, *lhsOrder);
}

/// `map` with its results reordered to follow `loops`, the order an op's body
/// runs its dims in. A read map says which element each loop point takes, not
/// which lanes it arrives in: the body wants every operand along its own loop
/// order. Left as is when some result or loop is not a plain dim.
static AffineMap inLoopOrder(AffineMap map, AffineMap loops) {
  std::optional<SmallVector<unsigned>> loopOrder = getDimOrder(loops);
  std::optional<SmallVector<unsigned>> order = getDimOrder(map);
  if (!loopOrder || !order || !llvm::all_of(*order, [&](unsigned dim) {
        return llvm::is_contained(*loopOrder, dim);
      }))
    return map;
  auto loopPosition = [&](unsigned dim) {
    return llvm::find(*loopOrder, dim) - loopOrder->begin();
  };
  llvm::stable_sort(*order, [&](unsigned lhs, unsigned rhs) {
    return loopPosition(lhs) < loopPosition(rhs);
  });
  SmallVector<AffineExpr> results;
  for (unsigned dim : *order)
    results.push_back(getAffineDimExpr(dim, map.getContext()));
  return AffineMap::get(map.getNumDims(), /*symbolCount=*/0, results,
                        map.getContext());
}

/// The results of `loops` that vary only over `dims`: the shape of a body
/// value computed from operands that together span `dims`. A value derived
/// from a row's sum alone is one element per row, however many dims the op
/// iterates.
static AffineMap restrictToDims(AffineMap loops,
                                const llvm::SmallBitVector &dims) {
  SmallVector<AffineExpr> results;
  for (AffineExpr result : loops.getResults()) {
    bool inDims = true;
    for (unsigned dim = 0, e = loops.getNumDims(); dim < e; ++dim)
      if (result.isFunctionOfDim(dim) && !dims.test(dim))
        inDims = false;
    if (inDims)
      results.push_back(result);
  }
  return AffineMap::get(loops.getNumDims(), /*symbolCount=*/0, results,
                        loops.getContext());
}

/// Whether an input operand shaped `shapeMap` (anchor-dim space) needs to be
/// gathered into a different lane order before this op's vectorized body can
/// use it: it is reused across some parallel anchor dim it does not itself
/// depend on (so it has to survive a whole parallel step, not just the
/// current point), *and* its own innermost dim disagrees with at least one
/// of the op's own outputs - meaning its natural memory order does not match
/// the order the op wants to consume it in.
static bool needsFullMaterialization(AffineMap shapeMap,
                                     ArrayRef<AffineMap> outputShapeMaps,
                                     ArrayRef<AnchorDim> anchorDims) {
  unsigned numDims = anchorDims.size();
  bool reusedAcrossParallelDim = false;
  for (unsigned dim = 0; dim < numDims; ++dim)
    if (anchorDims[dim].iteratorType != utils::IteratorType::reduction &&
        !shapeMap.isFunctionOfDim(dim))
      reusedAcrossParallelDim = true;
  if (!reusedAcrossParallelDim)
    return false;

  llvm::SmallBitVector operandDims = getDimsOf(shapeMap);
  llvm::SmallBitVector operandInnermost = getLaneDims(shapeMap);
  for (AffineMap outputMap : outputShapeMaps) {
    llvm::SmallBitVector outputInnermost = getLaneDims(outputMap);
    // A rank-0 output has no lane axis to supply.
    if (outputInnermost.none())
      continue;
    if (operandDims.anyCommon(outputInnermost) &&
        operandInnermost != outputInnermost)
      return true;
  }
  return false;
}

unsigned NodeBuilder::addRelayoutNode(unsigned source, AffineMap wanted,
                                      unsigned opIndex) {
  if (haveSameLayout(graph.nodes[source].shapeMap, wanted))
    return source;

  // The shuffle produces the tile the consumer wants, and the source stays
  // live across it because the permute network reads all of the one while
  // writing all of the other. Together that is the plan's source + dest.
  Node shuffle;
  shuffle.kind = NodeKind::Shuffle;
  shuffle.owningOp = opIndex;
  shuffle.shapeMap = wanted;
  shuffle.elementType = graph.nodes[source].elementType;
  shuffle.regClass = graph.nodes[source].regClass;
  shuffle.operands.push_back(source);
  return addNode(std::move(shuffle));
}

LogicalResult NodeBuilder::buildOperandNode(linalg::LinalgOp op,
                                            unsigned opIndex,
                                            ArrayRef<AffineMap> outputShapeMaps,
                                            OpOperand *operand, bool isInput) {
  BlockArgument arg = op.getMatchingBlockArgument(operand);
  Value tensor = operand->get();
  AffineMap shapeMap =
      op.getMatchingIndexingMap(operand).compose(getOpLoopMap(opIndex));
  Type elementType = getElementTypeOrSelf(tensor.getType());
  RegClass regClass = classifyResultType(elementType);
  if (regClass == RegClass::None)
    return graph.fail("operand of op '" + op->getName().getStringRef() +
                      "' has a type that is not held in an HVX register");

  // check if this is already in registers (from a previous producer)
  // in which case, there is nothing to do
  auto produced = producedTensor.find(tensor);
  bool materialized = graph.options.materializedValues.contains(tensor);
  if (produced != producedTensor.end() && !materialized) {
    // check if this needs a re-layout node
    if (isInput && // output is written in it layout by construction
        needsFullMaterialization(shapeMap, outputShapeMaps, graph.anchorDims))
      graph.nodes[produced->second].fullyMaterialized = true;
    nodeOf[arg] =
        addRelayoutNode(produced->second,
                        inLoopOrder(shapeMap, getOpLoopMap(opIndex)), opIndex);
    return success();
  }

  // Everything else comes from memory
  if (materialized) {
    auto reload = materializedLoad.find(tensor);
    if (reload != materializedLoad.end()) {
      nodeOf[arg] = reload->second;
      return success();
    }
  }

  Node node;
  node.kind = NodeKind::Load;
  node.value = arg;
  node.owningOp = opIndex;
  node.shapeMap = shapeMap;
  node.elementType = elementType;
  node.regClass = regClass;
  node.fullyMaterialized =
      isInput && // output is written in it layout by construction
      needsFullMaterialization(shapeMap, outputShapeMaps, graph.anchorDims);
  unsigned index = addNode(std::move(node));
  if (materialized)
    materializedLoad[tensor] = index;
  return success();
}

LogicalResult NodeBuilder::buildBlockArgsNodes(unsigned opIndex) {
  linalg::LinalgOp op = graph.ops[opIndex];
  bool hasReduction = op.getNumReductionLoops() > 0;

  // Every output's own shape, in anchor-dim space - what an input operand's
  // lane order is measured against to decide whether it needs a relayout.
  SmallVector<AffineMap> outputShapeMaps;
  for (int64_t i = 0, e = op.getNumDpsInits(); i < e; ++i)
    outputShapeMaps.push_back(op.getMatchingIndexingMap(op.getDpsInitOperand(i))
                                  .compose(getOpLoopMap(opIndex)));

  for (OpOperand *operand : op.getDpsInputOperands())
    if (failed(buildOperandNode(op, opIndex, outputShapeMaps, operand,
                                /*isInput=*/true)))
      return failure();

  for (int64_t i = 0, e = op.getNumDpsInits(); i < e; ++i) {
    OpOperand *init = op.getDpsInitOperand(i);
    BlockArgument arg = op.getMatchingBlockArgument(init);
    // An unused init block argument names no register: the yielded value is
    // what the output tile costs.
    if (arg.use_empty())
      continue;
    if (!hasReduction) {
      // Without a reduction, reading the init tile is an ordinary tile read.
      if (failed(buildOperandNode(op, opIndex, outputShapeMaps, init,
                                  /*isInput=*/false)))
        return failure();
      continue;
    }
    // for a reduction, this is the loop-carried accumulator.
    Node node;
    node.kind = NodeKind::Accumulator;
    node.value = arg;
    node.owningOp = opIndex;
    node.shapeMap =
        op.getMatchingIndexingMap(init).compose(getOpLoopMap(opIndex));
    node.elementType = getElementTypeOrSelf(init->get().getType());
    node.regClass = classifyResultType(node.elementType);
    if (node.regClass == RegClass::None)
      return graph.fail("accumulator of op '" + op->getName().getStringRef() +
                        "' has a type that is not held in an HVX register");
    addNode(std::move(node));
  }
  return success();
}

LogicalResult NodeBuilder::buildNodesforBody(unsigned opIndex) {
  linalg::LinalgOp op = graph.ops[opIndex];
  Block &body = op->getRegion(0).front();

  // analyze this linalgs body
  for (Operation &nested : body) {
    // yield doesnt have its own node, its modeled by the other nodes
    if (isa<linalg::YieldOp>(&nested)) {
      SmallVector<unsigned> producers;
      for (Value yielded : nested.getOperands()) {
        FailureOr<unsigned> producer = getOperandNode(yielded, opIndex);
        if (failed(producer))
          return failure();
        producers.push_back(*producer);
      }
      resultNodes[opIndex] = std::move(producers);
      continue;
    }
    if (nested.getNumResults() != 1)
      return graph.fail("body op '" + nested.getName().getStringRef() +
                        "' does not produce exactly one value");

    Node node;
    node.value = nested.getResult(0);
    node.op = &nested;
    node.owningOp = opIndex;
    node.elementType = getElementTypeOrSelf(node.value.getType());

    if (nested.hasTrait<OpTrait::ConstantLike>()) {
      node.kind = NodeKind::Invariant;
      node.shapeMap = getScalarMap();
      node.regClass = classifyResultType(node.elementType);
    } else {
      node.kind = NodeKind::BodyOp;
      node.regClass = classifyResultType(node.value.getType());
      llvm::SmallBitVector dims(graph.anchorDims.size());
      for (Value operand : nested.getOperands()) {
        FailureOr<unsigned> source = getOperandNode(operand, opIndex);
        if (failed(source))
          return failure();
        node.operands.push_back(*source);
        dims |= getDimsOf(graph.nodes[*source].shapeMap);
      }
      // The op runs once per point of the dims its operands vary over
      node.shapeMap = restrictToDims(getOpLoopMap(opIndex), dims);

      // if this op is costly and needs more than its tile live
      auto expansion =
          graph.getConfig().OpCosts.find(nested.getName().getStringRef());
      if (expansion != graph.getConfig().OpCosts.end()) {
        node.expansionTiles = expansion->second.extraTiles;
        if (expansion->second.invariantRegisters > 0)
          node.operands.push_back(getExpansionConstants(
              nested, node.elementType, expansion->second.invariantRegisters,
              opIndex));
      }
    }
    if (node.regClass == RegClass::None)
      return graph.fail("body op '" + nested.getName().getStringRef() +
                        "' produces a value that is not held in an HVX "
                        "register");
    addNode(std::move(node));
  }
  return success();
}

LogicalResult NodeBuilder::addReduceNodes(unsigned opIndex) {
  linalg::LinalgOp op = graph.ops[opIndex];
  if (op.getNumReductionLoops() == 0)
    return success();

  for (int64_t i = 0, e = op.getNumDpsInits(); i < e; ++i) {
    BlockArgument arg = op.getMatchingBlockArgument(op.getDpsInitOperand(i));
    auto accumulator = nodeOf.find(arg);
    if (accumulator == nodeOf.end())
      return graph.fail("op '" + op->getName().getStringRef() +
                        "' has reduction iterators but never reads its init "
                        "operand, so it has no accumulator");
    if (graph.reduction)
      return graph.fail("dispatches with more than one reduction are not "
                        "supported yet: phase ordering across reductions is "
                        "unimplemented");
    const Node &source = graph.nodes[accumulator->second];
    Node reduced;
    reduced.kind = NodeKind::Reduced;
    reduced.owningOp = opIndex;
    reduced.shapeMap = source.shapeMap;
    reduced.elementType = source.elementType;
    reduced.regClass = source.regClass;
    reduced.operands.push_back(accumulator->second);
    unsigned index = addNode(std::move(reduced));
    graph.reduction =
        Reduction{accumulator->second, resultNodes[opIndex][i], index};
    resultNodes[opIndex][i] = index;
  }
  return success();
}

LogicalResult NodeBuilder::addResultStoreNodes(unsigned opIndex) {
  linalg::LinalgOp op = graph.ops[opIndex];

  // Plan 4.4 step 2: a result leaves the dispatch unless every use of it is
  // another op of this dispatch
  // A materialized result is stored as well - that is what materializing it
  // means.
  for (auto [resultIndex, producer] : llvm::enumerate(resultNodes[opIndex])) {
    unsigned stored = producer;
    if (resultIndex < op->getNumResults()) {
      auto tensor = cast<OpResult>(op->getResult(resultIndex));
      producedTensor[tensor] = producer;
      bool fusedAway =
          !graph.options.materializedValues.contains(tensor) &&
          !tensor.use_empty() &&
          llvm::all_of(tensor.getUsers(), [&](Operation *user) {
            return isa<linalg::LinalgOp>(user) && user != op.getOperation();
          });
      if (fusedAway)
        continue;

      stored = addRelayoutNode(producer,
                               op.getIndexingMapMatchingResult(tensor).compose(
                                   getOpLoopMap(opIndex)),
                               opIndex);
    }

    // The store holds no register of its own - it consumes the value that
    // does - so it is classed `None` and only keeps that value live here.
    Node store;
    store.kind = NodeKind::Store;
    store.owningOp = opIndex;
    store.shapeMap = graph.nodes[stored].shapeMap;
    store.elementType = graph.nodes[stored].elementType;
    store.regClass = RegClass::None;
    store.operands.push_back(stored);
    addNode(std::move(store));
  }
  return success();
}

LogicalResult NodeBuilder::buildNodeForOp(unsigned opIndex) {
  if (failed(buildBlockArgsNodes(opIndex)))
    return failure();
  if (failed(buildNodesforBody(opIndex)))
    return failure();
  if (resultNodes[opIndex].empty() && graph.ops[opIndex]->getNumResults() != 0)
    return graph.fail("op '" + graph.ops[opIndex]->getName().getStringRef() +
                      "' yields no value");
  if (failed(addReduceNodes(opIndex)))
    return failure();
  return addResultStoreNodes(opIndex);
}

LogicalResult NodeBuilder::run() {
  // build empty nodes first
  resultNodes.assign(graph.ops.size(), {});
  // and create the node for each op
  for (unsigned index = 0, e = graph.ops.size(); index < e; ++index)
    if (failed(buildNodeForOp(index)))
      return failure();
  return success();
}

} // namespace

LogicalResult buildNodeGraph(DispatchRegisterGraph &graph) {
  if (failed(collectOps(graph)))
    return failure();

  auto anchor = llvm::find(graph.ops, graph.anchor);
  if (anchor == graph.ops.end())
    return graph.fail("the anchor is not a linalg op of the dispatch");

  // get tile sizes of all other ops expressed in "anchor coordinates" (symbolic
  // expressions)
  FailureOr<TilePropagationResult> tiles =
      propagateTiles(graph.ops, anchor - graph.ops.begin(),
                     [&](const Twine &reason) { return graph.fail(reason); });
  if (failed(tiles))
    return failure();
  graph.anchorDims = std::move(tiles->anchorDims);
  graph.anchorToOpLoops = std::move(tiles->anchorToOpLoops);
  graph.reductionDims.resize(graph.anchorDims.size());
  for (auto [index, dim] : llvm::enumerate(graph.anchorDims))
    if (dim.iteratorType == utils::IteratorType::reduction)
      graph.reductionDims.set(index);

  // build the graph nodes
  NodeBuilder node_builder(graph);
  if (failed(node_builder.run()))
    return failure();
  lowerReduction(graph);
  return success();
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning
