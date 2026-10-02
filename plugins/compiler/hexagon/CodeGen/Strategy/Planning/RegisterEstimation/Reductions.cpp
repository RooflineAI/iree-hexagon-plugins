// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "DispatchRegisterGraph.h"
#include "EstimatorConfig.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

/// Node indices that read `node`.
SmallVector<SmallVector<NodeIdx>>
buildConsumers(const DispatchRegisterGraph &graph) {
  SmallVector<SmallVector<NodeIdx>> consumers(graph.nodes.size());
  for (auto [index, node] : llvm::enumerate(graph.nodes)) {
    if (node.dead)
      continue;
    for (NodeIdx operand : node.operands)
      consumers[operand].push_back(index);
  }
  return consumers;
}

/// Everything `root` transitively reads, following operand edges backwards
/// and stopping at `stop`.
llvm::SmallDenseSet<NodeIdx> backwardClosure(const DispatchRegisterGraph &graph,
                                             NodeIdx root, NodeIdx stop) {
  llvm::SmallDenseSet<NodeIdx> visited;
  SmallVector<NodeIdx> worklist = {root};
  while (!worklist.empty()) {
    NodeIdx index = worklist.pop_back_val();
    if (index == stop || !visited.insert(index).second)
      continue;
    for (NodeIdx operand : graph.nodes[index].operands)
      worklist.push_back(operand);
  }
  return visited;
}

/// Whether `op` is one of the ops `names` lists.
bool matches(const llvm::StringSet<> &names, Operation *op) {
  return op && names.contains(op->getName().getStringRef());
}

/// Marks every node whose work exists only to feed an accumulator. Such a
/// node is produced and consumed one reduction chunk at a time.
// Unless its also read from outside the reduction
void markReductionChunkValues(DispatchRegisterGraph &graph) {
  if (!graph.reduction)
    return;
  const Reduction &reduction = *graph.reduction;
  SmallVector<SmallVector<NodeIdx>> consumers = buildConsumers(graph);
  llvm::SmallDenseSet<NodeIdx> feeding =
      backwardClosure(graph, reduction.update, reduction.accumulator);
  for (NodeIdx index : feeding) {
    if (llvm::all_of(consumers[index], [&](NodeIdx consumer) {
          return feeding.contains(consumer);
        }))
      graph.nodes[index].perReductionChunk = true;
  }
}

/// Collapses the update of the accumulator into one fused node that writes the
/// accumulator in place, when it matches one of the config's
/// `accumulateFusions`
void recognizeFusedAccumulates(DispatchRegisterGraph &graph) {
  if (!graph.reduction)
    return;
  Reduction &reduction = *graph.reduction;
  SmallVector<SmallVector<NodeIdx>> consumers = buildConsumers(graph);
  NodeIdx accumulator = reduction.accumulator;
  NodeIdx update = reduction.update;
  const Node &combine = graph.nodes[update];
  if (combine.operands.size() != 2)
    return;
  // The update has to be `accumulator <combine> something`.
  // find the producer
  NodeIdx produced = combine.operands[0] == accumulator   ? combine.operands[1]
                     : combine.operands[1] == accumulator ? combine.operands[0]
                                                          : update;
  if (produced == update)
    return;

  const AccumulateFusion *fusion = llvm::find_if(
      graph.getConfig().accumulateFusions, [&](const AccumulateFusion &f) {
        return matches(f.accumulateOps, combine.op) &&
               matches(f.producerOps, graph.nodes[produced].op);
      });
  if (fusion == graph.getConfig().accumulateFusions.end())
    return;

  // Absorb an operand op (e.g. an extension) when it exists only to feed the
  // producer: the instruction natively reads its operand.
  SmallVector<NodeIdx> sources;
  for (NodeIdx operand : graph.nodes[produced].operands) {
    Node &node = graph.nodes[operand];
    if (matches(fusion->absorbedOperandOps, node.op) &&
        node.operands.size() == 1 && consumers[operand].size() == 1) {
      sources.push_back(node.operands.front());
      node.dead = true;
      continue;
    }
    sources.push_back(operand);
  }

  Node fused;
  fused.kind = NodeKind::FusedAccumulate;
  fused.owningOp = combine.owningOp;
  fused.shapeMap = combine.shapeMap;
  fused.elementType = combine.elementType;
  fused.operands.assign(sources.begin(), sources.end());
  fused.operands.push_back(accumulator);
  // The fused node writes the accumulator it already reads, so it names no
  // register of its own.
  fused.regClass = RegClass::None;
  fused.perReductionChunk = combine.perReductionChunk;

  graph.nodes[produced].dead = true;
  graph.nodes[update].dead = true;
  reduction.update = graph.nodes.size();
  graph.nodes.push_back(std::move(fused));
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
  for (NodeIdx operand : graph.nodes[reduction.update].operands) {
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
  // Order matters: chunk marking reads the body as written, the accumulate
  // fusion then replaces that body, and the horizontal decision reads what is
  // left.
  markReductionChunkValues(graph);
  recognizeFusedAccumulates(graph);
  markHorizontalReduction(graph);
  findTiledReductionWidening(graph);
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning
