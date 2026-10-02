// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "DispatchRegisterGraph.h"
#include "EstimatorConfig.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// Node indices that read `node`.
SmallVector<SmallVector<unsigned>>
buildConsumers(const DispatchRegisterGraph &graph) {
  SmallVector<SmallVector<unsigned>> consumers(graph.nodes.size());
  for (auto [index, node] : llvm::enumerate(graph.nodes)) {
    if (node.dead)
      continue;
    for (unsigned operand : node.operands)
      consumers[operand].push_back(index);
  }
  return consumers;
}

/// Everything `root` transitively reads, following operand edges backwards
/// and stopping at `stop`.
llvm::SmallDenseSet<unsigned>
backwardClosure(const DispatchRegisterGraph &graph, unsigned root,
                unsigned stop) {
  llvm::SmallDenseSet<unsigned> visited;
  SmallVector<unsigned> worklist = {root};
  while (!worklist.empty()) {
    unsigned index = worklist.pop_back_val();
    if (index == stop || !visited.insert(index).second)
      continue;
    for (unsigned operand : graph.nodes[index].operands)
      worklist.push_back(operand);
  }
  return visited;
}

bool isExtension(Operation *op) {
  return op && isa<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp>(op);
}

bool isMultiply(Operation *op) {
  return op && isa<arith::MulIOp, arith::MulFOp>(op);
}

bool isAdd(Operation *op) {
  return op && isa<arith::AddIOp, arith::AddFOp>(op);
}

/// Marks every node whose work exists only to feed an accumulator. Such a
/// node is produced and consumed one reduction chunk at a time.
// Unless its also read from outside the reduction
void markReductionChunkValues(DispatchRegisterGraph &graph) {
  if (!graph.reduction)
    return;
  const Reduction &reduction = *graph.reduction;
  SmallVector<SmallVector<unsigned>> consumers = buildConsumers(graph);
  llvm::SmallDenseSet<unsigned> feeding =
      backwardClosure(graph, reduction.update, reduction.accumulator);
  for (unsigned index : feeding) {
    if (llvm::all_of(consumers[index], [&](unsigned consumer) {
          return feeding.contains(consumer);
        }))
      graph.nodes[index].perReductionChunk = true;
  }
}

/// Collapses `mul` (or `ext` + `mul`) feeding the `add` that updates an
/// accumulator into one fused multiply-add node that writes the accumulator in
/// place.
void recognizeMultiplyAccumulates(DispatchRegisterGraph &graph) {
  if (!graph.reduction)
    return;
  Reduction &reduction = *graph.reduction;
  SmallVector<SmallVector<unsigned>> consumers = buildConsumers(graph);
  unsigned accumulator = reduction.accumulator;
  unsigned update = reduction.update;
  const Node &add = graph.nodes[update];
  if (!isAdd(add.op) || add.operands.size() != 2)
    return;

  // The update has to be `accumulator + something`.
  unsigned product = add.operands[0] == accumulator   ? add.operands[1]
                     : add.operands[1] == accumulator ? add.operands[0]
                                                      : update;
  if (product == update || !isMultiply(graph.nodes[product].op))
    return;
  if (graph.nodes[product].operands.size() != 2)
    return;

  // Peel an extension off each factor when it exists only to widen it: the
  // instruction natively reads the narrow operand.
  SmallVector<unsigned> sources;
  for (unsigned factor : graph.nodes[product].operands) {
    Node &node = graph.nodes[factor];
    if (isExtension(node.op) && node.operands.size() == 1 &&
        consumers[factor].size() == 1) {
      sources.push_back(node.operands.front());
      node.dead = true;
      continue;
    }
    sources.push_back(factor);
  }

  Node mac;
  mac.kind = NodeKind::MultiplyAccumulate;
  mac.owningOp = add.owningOp;
  mac.shapeMap = add.shapeMap;
  mac.elementType = add.elementType;
  mac.operands.assign(sources.begin(), sources.end());
  mac.operands.push_back(accumulator);
  // The fused node writes the accumulator it already reads, so it names no
  // register of its own.
  mac.regClass = RegClass::None;
  mac.perReductionChunk = add.perReductionChunk;

  graph.nodes[product].dead = true;
  graph.nodes[update].dead = true;
  reduction.update = graph.nodes.size();
  graph.nodes.push_back(std::move(mac));
}

/// When the reduction runs along the innermost dim of the value being
/// reduced, the accumulator ends up spread across the lanes of a vector and
/// has to be folded across them.
void markHorizontalReduction(DispatchRegisterGraph &graph) {
  if (!graph.reduction)
    return;
  Reduction &reduction = *graph.reduction;
  std::optional<unsigned> outputLane =
      getLaneDim(graph.nodes[reduction.accumulator].shapeMap);
  bool laneAligned = false;
  bool alongReduction = false;
  bool anyFullyMaterialized = false;
  for (const Node &node : graph.nodes) {
    if (node.dead || node.kind == NodeKind::Accumulator ||
        !node.perReductionChunk)
      continue;
    if (node.fullyMaterialized)
      anyFullyMaterialized = true;
    std::optional<unsigned> lane = getLaneDim(node.shapeMap);
    if (!lane)
      continue;
    if (lane == outputLane)
      laneAligned = true;
    else if (graph.reductionDims.test(*lane))
      alongReduction = true;
  }
  reduction.horizontal =
      !laneAligned && alongReduction && !anyFullyMaterialized;
}

/// A horizontal reduction keeps one lane-spread partial result per register of
/// its operands until the phase ends and they are folded into a single value.
/// So the accumulator is as wide as the widest per-chunk operand that runs
/// along a reduction lane. Records those operands.
/// e.g. an update fed by a 2-register operand needs 2 accumulator registers,
/// folded together only after the last step.
void findTiledReductionWidening(DispatchRegisterGraph &graph) {
  if (!graph.reduction || !graph.reduction->horizontal)
    return;
  Reduction &reduction = *graph.reduction;
  for (unsigned operand : graph.nodes[reduction.update].operands) {
    if (operand == reduction.accumulator)
      continue;
    const Node &source = graph.nodes[operand];
    if (source.perReductionChunk &&
        getLaneDims(source.shapeMap).anyCommon(graph.reductionDims))
      reduction.wideningOperands.push_back(operand);
  }
}

} // namespace

void lowerReduction(DispatchRegisterGraph &graph) {
  // Order matters: chunk marking reads the body as written, the MAC fold then
  // replaces that body, and the horizontal decision reads what is left.
  markReductionChunkValues(graph);
  recognizeMultiplyAccumulates(graph);
  markHorizontalReduction(graph);
  findTiledReductionWidening(graph);
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning
