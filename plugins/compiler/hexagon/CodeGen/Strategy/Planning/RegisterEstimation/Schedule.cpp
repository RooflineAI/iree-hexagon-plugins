// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Plan sections 4.5 and 4.6: phases, node placement, the schedule, its live
// ranges, and the footprint every node is weighed by. Everything here is
// tile-independent, so it runs once in `build`.

#include "DispatchRegisterGraph.h"
#include "EstimatorConfig.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <limits>

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

constexpr unsigned kNotScheduled = std::numeric_limits<unsigned>::max();

/// Which nodes carry a reduction anchor dim in their tile: those are
/// the ones that only exist inside the reduction loop.
llvm::SmallBitVector carriesReduction(const DispatchRegisterGraph &graph) {
  llvm::SmallBitVector carries(graph.nodes.size());
  for (auto [index, node] : llvm::enumerate(graph.nodes))
    if (node.shapeMap &&
        getDimsOf(node.shapeMap).anyCommon(graph.reductionDims))
      carries.set(index);
  return carries;
}

/// Everything that transitively feeds the accumulator, plus the accumulator
/// itself.
llvm::SmallBitVector reachesAccumulator(const DispatchRegisterGraph &graph) {
  llvm::SmallBitVector inPhase(graph.nodes.size());
  SmallVector<unsigned> worklist;
  if (graph.reduction) {
    inPhase.set(graph.reduction->accumulator);
    worklist.push_back(graph.reduction->update);
  }
  while (!worklist.empty()) {
    unsigned index = worklist.pop_back_val();
    if (inPhase.test(index))
      continue;
    inPhase.set(index);
    for (unsigned operand : graph.nodes[index].operands)
      worklist.push_back(operand);
  }
  return inPhase;
}

/// collects the indices of the nodes that use each operand
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

/// Everything that transitively reads the value the reduction left behind.
llvm::SmallBitVector readsReduced(const DispatchRegisterGraph &graph,
                                  ArrayRef<SmallVector<unsigned>> consumers) {
  llvm::SmallBitVector reads(graph.nodes.size());
  SmallVector<unsigned> worklist;
  if (graph.reduction)
    worklist.append(consumers[graph.reduction->reduced].begin(),
                    consumers[graph.reduction->reduced].end());
  while (!worklist.empty()) {
    unsigned index = worklist.pop_back_val();
    if (reads.test(index))
      continue;
    reads.set(index);
    worklist.append(consumers[index].begin(), consumers[index].end());
  }
  return reads;
}

/// Where a placement falls in the skeleton, for ordering placements.
unsigned getPlacementRank(const Placement &placement, unsigned numPhases) {
  switch (placement.kind) {
  case Placement::Prologue:
    return 0;
  case Placement::InPhase:
    return 1 + placement.phase;
  case Placement::Epilogue:
    return 1 + numPhases;
  }
  llvm_unreachable("unknown placement kind");
}

/// Assign each value a phase, where it is produced
void assignPlacements(DispatchRegisterGraph &graph) {
  bool hoistInvariants =
      graph.getConfig().invariantPlacement == InvariantPlacement::AlwaysLive;

  if (!graph.hasReductionDims()) {
    // Without a reduction, there are no different phases
    for (Node &node : graph.nodes) {
      node.placement = node.kind == NodeKind::Invariant && hoistInvariants
                           ? Placement{Placement::Prologue, 0}
                           : Placement{Placement::InPhase, 0};
    }
    graph.numPhases = 1;
    return;
  }

  // classify where the values live
  SmallVector<SmallVector<unsigned>> consumers = buildConsumers(graph);
  llvm::SmallBitVector carries = carriesReduction(graph);
  llvm::SmallBitVector inReduction = reachesAccumulator(graph);
  llvm::SmallBitVector afterReduction = readsReduced(graph, consumers);

  // A trailing Map phase exists if something still runs over the
  // reduction dims after the reduced value is known (like RMSNorm's scale,
  // softmax's divide)
  bool needsMapPhase = false;
  for (auto [index, node] : llvm::enumerate(graph.nodes))
    if (!node.dead && afterReduction.test(index) && carries.test(index))
      needsMapPhase = true;
  graph.numPhases = needsMapPhase ? 2 : 1;
  unsigned lastPhase = graph.numPhases - 1;

  // place nodes in prologue or in reduction
  llvm::SmallBitVector placed(graph.nodes.size());
  for (auto [index, node] : llvm::enumerate(graph.nodes)) {
    if (node.kind == NodeKind::Invariant && hoistInvariants) {
      node.placement = Placement{Placement::Prologue, 0};
      placed.set(index);
    } else if (inReduction.test(index) || node.kind == NodeKind::Reduced) {
      node.placement = Placement{Placement::InPhase, 0};
      placed.set(index);
    }
  }

  // reverse iteration through the nodes,
  // decides the placement of consumers first
  for (int index = graph.nodes.size() - 1; index >= 0; --index) {
    Node &node = graph.nodes[index];
    if (node.dead || placed.test(index) || node.kind == NodeKind::Store)
      continue;
    std::optional<Placement> earliest;
    for (unsigned consumer : consumers[index]) {
      const Node &reader = graph.nodes[consumer];
      if (reader.kind == NodeKind::Store)
        continue;
      if (!earliest || getPlacementRank(reader.placement, graph.numPhases) <
                           getPlacementRank(*earliest, graph.numPhases))
        earliest = reader.placement;
    }

    bool after = afterReduction.test(index);
    if (carries.test(index)) {
      // It only exists inside a loop over the reduction dims.
      unsigned phase = after ? lastPhase : 0;
      if (!after && earliest && earliest->kind == Placement::InPhase)
        phase = earliest->phase;
      node.placement = Placement{Placement::InPhase, phase};
    } else if (!earliest) {
      node.placement = after ? Placement{Placement::Epilogue, 0}
                             : Placement{Placement::Prologue, 0};
    } else if (earliest->kind == Placement::InPhase && earliest->phase == 0 &&
               !after) {
      // Invariant over the reduction loop: computed once ahead of it.
      node.placement = Placement{Placement::Prologue, 0};
    } else {
      node.placement = *earliest;
    }
    placed.set(index);
  }

  for (Node &node : graph.nodes) {
    if (node.dead || node.kind != NodeKind::Store)
      continue;
    // A reduced value exists once its phase is over
    if (llvm::any_of(node.operands, [&](unsigned operand) {
          return graph.nodes[operand].kind == NodeKind::Reduced;
        })) {
      node.placement = Placement{Placement::Epilogue, 0};
      continue;
    }
    node.placement = graph.nodes[node.operands.front()].placement;
    for (unsigned operand : node.operands)
      if (getPlacementRank(graph.nodes[operand].placement, graph.numPhases) >
          getPlacementRank(node.placement, graph.numPhases))
        node.placement = graph.nodes[operand].placement;
  }
}

/// orders the nodes: prologue, each phase, then the epilogue. Within
/// a phase, nodes are ordered in program order. The exception is a
/// value, the reduction produces, which is placed at the end of the reduction
SmallVector<unsigned> buildExecutionOrder(const DispatchRegisterGraph &graph) {
  SmallVector<unsigned> order;
  auto append = [&](llvm::function_ref<bool(const Node &)> accept) {
    for (auto [index, node] : llvm::enumerate(graph.nodes))
      if (!node.dead && accept(node))
        order.push_back(index);
  };

  append([](const Node &node) {
    return node.placement.kind == Placement::Prologue;
  });
  for (unsigned phase = 0; phase < graph.numPhases; ++phase) {
    auto inThisPhase = [&](const Node &node) {
      return node.placement.kind == Placement::InPhase &&
             node.placement.phase == phase;
    };
    append([&](const Node &node) {
      return inThisPhase(node) && node.kind != NodeKind::Reduced;
    });
    append([&](const Node &node) {
      return inThisPhase(node) && node.kind == NodeKind::Reduced;
    });
  }
  append([](const Node &node) {
    return node.placement.kind == Placement::Epilogue;
  });
  return order;
}

/// every scheduled nodes live range, from its definition
/// to its last use, both inclusive.
void computeLiveIntervals(DispatchRegisterGraph &graph,
                          ArrayRef<unsigned> order,
                          ArrayRef<unsigned> positionOf) {
  graph.live.assign(graph.nodes.size(), LiveInterval{});
  if (order.empty())
    return;
  unsigned lastPosition = order.size() - 1;
  auto isScheduled = [&](unsigned node) {
    return positionOf[node] != kNotScheduled;
  };

  // convert phase number into position in the schedule
  SmallVector<unsigned> phaseStart(graph.numPhases, lastPosition);
  SmallVector<unsigned> phaseEnd(graph.numPhases, 0);
  for (auto [position, index] : llvm::enumerate(order)) {
    const Placement &placement = graph.nodes[index].placement;
    if (placement.kind != Placement::InPhase)
      continue;
    unsigned at = position;
    phaseStart[placement.phase] = std::min(phaseStart[placement.phase], at);
    phaseEnd[placement.phase] = at;
  }

  llvm::SmallBitVector carries = carriesReduction(graph);
  // start with the phase, where the value is created
  for (auto [position, index] : llvm::enumerate(order))
    graph.live[index] = LiveInterval{static_cast<unsigned>(position),
                                     static_cast<unsigned>(position)};

  // extend the live interval of each value, up to its read
  for (auto [position, index] : llvm::enumerate(order)) {
    const Node &node = graph.nodes[index];
    for (unsigned operand : node.operands) {
      if (!isScheduled(operand))
        continue;
      LiveInterval &interval = graph.live[operand];
      interval.end = std::max<unsigned>(interval.end, position);

      // if a value is created before a phase and read in every iteration, of
      // the next phase, it needs to be live the entire phase (loop invariant
      // for this phase, not consumed on first usage)
      const Placement &use = node.placement;
      const Placement &def = graph.nodes[operand].placement;
      if (use.kind != Placement::InPhase || !carries.test(index))
        continue;
      bool definedEarlier =
          def.kind == Placement::Prologue ||
          (def.kind == Placement::InPhase && def.phase < use.phase);
      if (definedEarlier)
        interval.end = std::max(interval.end, phaseEnd[use.phase]);
    }
  }

  // A value defined in the readers phase, but loop invariant in the phase, is
  // hoisted, so its live until the end
  for (auto [index, node] : llvm::enumerate(graph.nodes)) {
    if (!isScheduled(index) || node.placement.kind != Placement::InPhase ||
        node.kind == NodeKind::Accumulator || carries.test(index))
      continue;
    unsigned phase = node.placement.phase;
    bool readEveryTrip = llvm::any_of(order, [&](unsigned reader) {
      const Node &other = graph.nodes[reader];
      return carries.test(reader) &&
             other.placement.kind == Placement::InPhase &&
             other.placement.phase == phase &&
             llvm::is_contained(other.operands, index);
    });
    if (readEveryTrip)
      graph.live[index].end = std::max(graph.live[index].end, phaseEnd[phase]);
  }

  // The accumulator is is live over the whole phase.
  if (graph.reduction && isScheduled(graph.reduction->accumulator)) {
    unsigned accumulator = graph.reduction->accumulator;
    unsigned phase = graph.nodes[accumulator].placement.phase;
    graph.live[accumulator] = LiveInterval{phaseStart[phase], phaseEnd[phase]};
  }

  // A hoisted invariant is materialized once and never recomputed.
  if (graph.getConfig().invariantPlacement == InvariantPlacement::AlwaysLive) {
    for (auto [index, node] : llvm::enumerate(graph.nodes))
      if (node.kind == NodeKind::Invariant && isScheduled(index))
        graph.live[index] = LiveInterval{0, lastPosition};
  }
}

/// Turns the order and live ranges into steps: what starts, what may be
/// reused and what ends at each position.
void buildSteps(DispatchRegisterGraph &graph, ArrayRef<unsigned> order,
                ArrayRef<unsigned> positionOf) {
  graph.steps.assign(order.size(), Step{});
  for (auto [position, index] : llvm::enumerate(order)) {
    Step &step = graph.steps[position];
    step.node = index;
    const LiveInterval &interval = graph.live[index];
    graph.steps[interval.start].starts.push_back(index);
    graph.steps[interval.end].ends.push_back(index);

    // check which operands end here
    if (graph.nodes[index].kind == NodeKind::Shuffle)
      continue;
    for (unsigned operand : graph.nodes[index].operands) {
      if (positionOf[operand] == kNotScheduled ||
          graph.live[operand].end != position ||
          llvm::is_contained(step.reusable, operand))
        continue;
      step.reusable.push_back(operand);
    }
  }
}

/// A per-chunk value is produced and consumed one reduction step at a time:
/// only the dims of the class its own lane dim belongs to (reduction or
/// parallel) are live. Among those, the lane dims pack into a register and
/// every other one is an unrolled row.
Footprint getSliceFootprint(const DispatchRegisterGraph &graph,
                            const Node &node, unsigned bits) {
  llvm::SmallBitVector laneDims = getLaneDims(node.shapeMap);
  bool keepReduction = laneDims.anyCommon(graph.reductionDims);
  Footprint footprint;
  footprint.bits = bits;
  for (unsigned dim = 0, e = graph.anchorDims.size(); dim < e; ++dim) {
    if (graph.reductionDims.test(dim) != keepReduction ||
        !node.shapeMap.isFunctionOfDim(dim))
      continue;
    (laneDims.test(dim) ? footprint.lanes : footprint.rows)
        .push_back(Extent::ofDim(dim));
  }
  return footprint;
}

/// The reader that makes `index` a phase-invariant value broadcast along
/// lanes it lacks: a node running over the reduction dims, whose own lanes
/// `index` does not have. Not running over the reduction dims itself, `index`
/// is the same on every trip of that reader's loop - whether it was computed
/// in the same phase, left behind by an earlier one or hoisted.
std::optional<unsigned>
findBroadcastReader(const DispatchRegisterGraph &graph, unsigned index,
                    ArrayRef<SmallVector<unsigned>> consumers,
                    const llvm::SmallBitVector &carries) {
  const Node &node = graph.nodes[index];
  if (node.kind == NodeKind::Invariant || node.kind == NodeKind::Accumulator ||
      node.perReductionChunk || node.fullyMaterialized || carries.test(index) ||
      node.shapeMap.getNumResults() == 0)
    return std::nullopt;
  llvm::SmallBitVector dims = getDimsOf(node.shapeMap);
  for (unsigned consumer : consumers[index]) {
    const Node &reader = graph.nodes[consumer];
    if (!carries.test(consumer) || reader.placement.kind != Placement::InPhase)
      continue;
    llvm::SmallBitVector lanes = getLaneDims(reader.shapeMap);
    if (lanes.any() && !dims.anyCommon(lanes))
      return consumer;
  }
  return std::nullopt;
}

/// compute every scheduled node's footprint and temporaries.
LogicalResult assignFootprints(DispatchRegisterGraph &graph) {
  SmallVector<SmallVector<unsigned>> consumers = buildConsumers(graph);
  llvm::SmallBitVector carries = carriesReduction(graph);
  for (auto [index, node] : llvm::enumerate(graph.nodes)) {
    if (node.dead)
      continue;
    unsigned bits = getElementBitWidth(node.elementType);
    FailureOr<Footprint> resident = Footprint::resident(node.shapeMap, bits);
    if (failed(resident))
      return graph.fail("cannot resolve the tile shape of a " +
                        stringifyNodeKind(node.kind) + " node");

    if (node.kind == NodeKind::Reduced && graph.reduction &&
        graph.reduction->reduced == index && graph.reduction->horizontal) {
      // The cross-lane fold works through a tree of shuffles and adds,
      // holding copies of the output tile on the way.
      node.temporaries = *resident;
      node.temporaries.copies = graph.getConfig().horizontalReductionCopies;
    }

    // A mask that the following node consumes stays in the predicate, so it
    // costs no vector register. One that outlives that node is materialized as
    // a vector, which is what its footprint counts.
    const LiveInterval &live = graph.live[index];
    bool shortLivedPredicate =
        node.regClass == RegClass::Predicate && live.end <= live.start + 1;
    if (node.regClass == RegClass::None || shortLivedPredicate) {
      node.footprint = Footprint{};
      continue;
    }

    if (node.kind == NodeKind::Shuffle) {
      // The destination tile; the source is charged by staying live across.

      // THis would be the point to extend, iw we want to model shuffle cost
      node.footprint = *resident;
    } else if (node.fullyMaterialized) {
      // First the Node is read in and then transposed into the order the
      // consumer needs
      node.footprint = *Footprint::flat(node.shapeMap, bits);
      node.temporaries = node.footprint;
    } else if (node.perReductionChunk) {
      node.footprint = getSliceFootprint(graph, node, bits);
    } else {
      node.footprint = *resident;
    }

    if (node.kind == NodeKind::Invariant)
      node.footprint.copies = node.invariantRegisters;
    if (node.expansionTiles > 0) {
      node.temporaries = node.footprint;
      node.temporaries.copies = node.expansionTiles;
    }

    // a value that needs to be broadcasted to its consumers shape
    if (std::optional<unsigned> reader =
            findBroadcastReader(graph, index, consumers, carries))
      node.broadcast =
          *Footprint::resident(graph.nodes[*reader].shapeMap, bits);
  }
  return success();
}

} // namespace

LogicalResult buildSchedule(DispatchRegisterGraph &graph) {
  assignPlacements(graph);
  SmallVector<unsigned> order = buildExecutionOrder(graph);
  SmallVector<unsigned> positionOf(graph.nodes.size(), kNotScheduled);
  for (auto [position, index] : llvm::enumerate(order))
    positionOf[index] = position;

  // double-check that no consumer is placed after their producers
  for (auto [position, index] : llvm::enumerate(order))
    for (unsigned operand : graph.nodes[index].operands)
      if (positionOf[operand] != kNotScheduled &&
          positionOf[operand] >= position)
        return graph.fail("the schedule places a " +
                          stringifyNodeKind(graph.nodes[index].kind) +
                          " node before the value it reads");

  computeLiveIntervals(graph, order, positionOf);
  buildSteps(graph, order, positionOf);
  return assignFootprints(graph);
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning
