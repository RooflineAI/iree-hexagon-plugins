// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_DISPATCHREGISTERGRAPH_INTERNALS_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_DISPATCHREGISTERGRAPH_INTERNALS_H_

/// Defines the internal Datastructures for RegisterGraph

#include "Footprint.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <optional>

namespace mlir::iree_compiler::hexagon::codegen::planning {

/// Index of a node in `DispatchRegisterGraph::nodes`.
using NodeIdx = unsigned;
/// Position in the schedule, i.e. an index into `DispatchRegisterGraph::steps`.
using StepIdx = unsigned;
/// Index of a linalg op in `DispatchRegisterGraph::ops`.
using OpIdx = unsigned;

/// What a node stands for. Every node holds at most one value; the kinds
/// differ in where that value comes from and how its live range is derived.
enum class NodeKind {
  BodyOp,    // an arith/math op from a linalg op's body
  Load,      // tile loaded from memory (dispatch input or materialized edge)
  Store,     // tile stored to memory (dispatch output or materialized edge)
  Invariant, // splatted constant / captured scalar
  Shuffle,   // inserted layout change between producer and consumer
  // Note: the shuffle node could be the pooint to investigate, if we want to
  // model shuffle cost
  Accumulator,        // loop-carried value of a reduction
  MultiplyAccumulate, // a contraction's fused multiply-accumulate, in place
  Reduced,            // what a reduction leaves behind once its phase has run
};

StringRef stringifyNodeKind(NodeKind kind);

/// Position of a node in the schedule skeleton:
/// Prologue, Phase[0], ..., Phase[n-1], Epilogue.
struct Placement {
  enum Kind { Prologue, InPhase, Epilogue };
  Kind kind = Prologue;
  unsigned phase = 0; // valid for InPhase
};

struct Node {
  NodeKind kind = NodeKind::BodyOp;
  /// The IR value this node stands for. Null for synthetic nodes (stores,
  /// shuffles, MACs, reduced values).
  Value value;
  /// The body op this node came from, if any.
  Operation *op = nullptr;
  /// The linalg op this node belongs to.
  OpIdx owningOp = 0;
  /// anchor dims -> this value's tile dims. Its footprint is derived from it.
  AffineMap shapeMap;
  /// Element type of the value.
  Type elementType;
  RegClass regClass = RegClass::None;
  Placement placement;
  /// This value is produced and consumed one reduction step at a time, so it
  /// is weighed as a slice rather than as a resident tile.
  bool perReductionChunk = false;
  /// This operand's own layout disagrees with the lane order its op's output
  /// wants, so the whole tile is gathered into that order before use and
  /// every element of it is live at once.
  bool fullyMaterialized = false;
  /// Registers a hoisted invariant holds: one for a splat, more for the
  /// constant table of an expanded op (OpExpansion::invariantRegisters).
  int64_t invariantRegisters = 1;
  /// Tiles the size of this node's result that an expanded op holds while it
  /// runs.
  int64_t expansionTiles = 0;
  /// Removed while the graph was built (its work was folded into another
  /// node).
  bool dead = false;
  /// Nodes this node reads.
  SmallVector<NodeIdx> operands;

  // --- decided at the end of build, read by evaluate -----------------------
  /// Registers holding this node's value.
  Footprint footprint;
  /// Registers live only while this node executes, beyond its value.
  Footprint temporaries;
  /// Set for a value that needs to be broadcasted to its readers shape
  std::optional<Footprint> broadcast;
};

/// The reduction of the dispatch: its loop-carried accumulator, the node that
/// writes the accumulator's next value, and the value left once the phase has
/// run.
struct Reduction {
  NodeIdx accumulator = 0;
  NodeIdx update = 0;
  NodeIdx reduced = 0;
  /// The accumulator ends up spread across the lanes of a vector and has to
  /// be folded across them when the phase ends.
  bool horizontal = false;
  /// When the reduction runs over several tiles, a horizontal reduction's
  /// accumulator keeps every step's lane-spread partial result live, so it is
  /// at least as wide as the widest reduction-lane operand feeding it.
  SmallVector<NodeIdx> wideningOperands;
};

/// A closed position interval [start, end] in the schedule.
struct LiveInterval {
  StepIdx start = 0;
  StepIdx end = 0;
};

/// One position of the schedule: the node that executes there and how the
/// live set changes around it. Everything here is tile-independent.
struct Step {
  NodeIdx node = 0;
  /// Values that become live at this step.
  SmallVector<NodeIdx, 2> starts;
  /// Operands read here for the last time.
  SmallVector<NodeIdx, 2> reusable;
  /// Values dead once this step has run.
  SmallVector<NodeIdx, 2> ends;
};

/// What every node's value holds at one tile, indexed by node.
struct NodeWeights {
  SmallVector<int64_t> registers;
  /// Bits of those registers that hold data (Footprint::usefulBits).
  SmallVector<int64_t> usefulBits;
};

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_DISPATCHREGISTERGRAPH_INTERNALS_H_
