// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_DISPATCHREGISTERGRAPH_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_DISPATCHREGISTERGRAPH_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include "DispatchRegisterGraph-internals.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {

struct EstimatorConfig;

/// Peak simultaneously-live registers.
///
/// Only the vector registers are tracked.
// Other, like Loop control, addressing, tile bookkeeping, ...
// are left out
struct RegisterPressure {
  int64_t vector;      // HVX registers
  int64_t usefulBytes; /// Bytes of data the values live at the peak hold
  /// Total Bytes in one register
  // its an extra field, so `utilization` needs no config.
  int64_t registerBytes;

  /// Fraction of the peak's registers that holds data
  /// Does not count additional overhead due to shuffles or packing
  /// a scalarized kernel will see a very low utilization, since this is not
  /// modeled
  double utilization() const {
    if (vector == 0)
      return 1.0;
    return static_cast<double>(usefulBytes) /
           static_cast<double>(vector * registerBytes);
  }
};

/// One entry of the caller's tile vector. `op` and `loop` say which op loop's
/// tile the entry carries. anchor is usually the root op in a dispatch
struct AnchorDim {
  utils::IteratorType iteratorType = utils::IteratorType::parallel;
  /// Static extent of the loop. The analysis rejects dynamic shapes, so this
  /// is always a concrete positive number.
  int64_t extent = 0;
  /// The op loop whose tile this entry carries: the anchor's own loop, or for
  /// a pinned dim the fused-op loop no anchor loop reaches. The estimator
  /// infers the tile of every other op loop tied to it.
  Operation *op = nullptr;
  /// position of that loop in `op`'s own loop order
  unsigned loop = 0;
};

// Configuration of the Graph to be specified by the callers
struct DispatchGraphOptions {
  /// Values that are written to memory and reloaded by their consumers,
  ///  instead of staying in registers.
  ///  Default: every fused edge lives in registers.
  /// Note: This could be the point to add accounting for the HMX part
  llvm::SmallPtrSet<Value, 4> materializedValues;
  /// All modeling knobs.
  /// This includes the Vector register width
  std::shared_ptr<const EstimatorConfig> config;
  /// Optional: Receives the reason whenever `build` or `evaluate` fails. The
  /// same reason is also emitted through LLVM_DEBUG. Optional.
  std::function<void(StringRef)> onFailure;
};

/// A tile-independent model of one fused dispatch
/// One First builds the graph and can in turn query the vector register usage
/// of the dispatch with different tile sizes. Does not model overhead due to
/// shuffling or unaligned loads
class DispatchRegisterGraph {
public:
  /// Tile-independent analysis. `anchor` is the dispatch's root op, whose
  /// loop order the tile vectors passed to `evaluate` are expressed in
  /// Fails if the dispatch is outside the supported subset.
  static FailureOr<DispatchRegisterGraph>
  build(FunctionOpInterface dispatch, linalg::LinalgOp anchor,
        DispatchGraphOptions options = DispatchGraphOptions());

  /// Per candidate. `tileSizes` has one entry per anchor dim, in
  /// getAnchorDims() order. whether the tile is legal for codegen is the
  /// caller's responsibility.
  FailureOr<RegisterPressure> evaluate(ArrayRef<int64_t> tileSizes) const;

  /// The meaning of each entry of `evaluate`'s tile vector
  ArrayRef<AnchorDim> getAnchorDims() const;
  /// The dispatch's linalg ops, in program order.
  ArrayRef<linalg::LinalgOp> getOps() const;

  // Everything below is what `build` precomputes and `evaluate` reads. It is
  // public so the build stages in their own translation units can fill it.
  // TODO evaluate if i need to keep it public

  // --- inputs ------------------------------------------------------------
  FunctionOpInterface dispatch;
  /// The caller's root op; the tile vector is in its loop order.
  linalg::LinalgOp anchor;
  std::shared_ptr<const EstimatorConfig> config;
  DispatchGraphOptions options;

  // --- structure ---------------------------------------------------------
  /// The dispatch's linalg ops, in program order.
  SmallVector<linalg::LinalgOp> ops;
  SmallVector<AnchorDim> anchorDims;
  /// Per op: anchor dims -> that op's loops.
  SmallVector<AffineMap> anchorToOpLoops;
  /// Anchor dims that are reductions.
  llvm::SmallBitVector reductionDims;

  SmallVector<Node, 0> nodes;
  /// The dispatch's single reduction, if it has one.
  std::optional<Reduction> reduction;
  /// Loops over the shared reduction dims inside the parallel tile: one for
  /// the reduction, plus one for a map that still runs over them afterwards.
  unsigned numPhases = 0;

  // --- schedule ----------------------------------------------------------
  SmallVector<Step> steps;
  /// Node index -> its live range, valid only for scheduled nodes.
  SmallVector<LiveInterval> live;

  // --- helpers -----------------------------------------------------------
  /// Reports `reason` through the caller's callback and LLVM_DEBUG, and
  /// returns failure so callers can `return graph.fail(...)`.
  LogicalResult fail(const Twine &reason) const;

  const EstimatorConfig &getConfig() const { return *config; }
  bool hasReductionDims() const { return reductionDims.any(); }

  /// Whether a resolved tile vector leaves some reduction anchor dim short of
  /// its full extent, i.e. whether the reduction runs over several tiles.
  bool isReductionTiled(ArrayRef<int64_t> tile) const;

  /// Whether `resolvedTile` chunks the reduction dim of a value that is both
  /// `fullyMaterialized` and a fused producer's result.
  ///  In This case, re-layout cost is relevant, but not modeled
  bool chunksAFusedRelayout(ArrayRef<int64_t> resolvedTile) const;

  /// What every node's value holds at `resolvedTile`.
  NodeWeights weighNodes(ArrayRef<int64_t> tile) const;

private:
  /// Debug only: prints nodes, phases, schedule, weights and the live set at
  /// the peak for the given tile.
  void dump(llvm::raw_ostream &os, const ArrayRef<int64_t> tileSizes,
            const NodeWeights &weights, StepIdx peak_position,
            const RegisterPressure &pressure) const;

  DispatchRegisterGraph() = default;
};

// --- stages, each in its own translation unit ----------------------------

/// collect the ops, propagate tile maps, lift the bodies into
/// nodes, splice fused edges and insert dispatch boundaries.
LogicalResult buildNodeGraph(DispatchRegisterGraph &graph);

/// marks per-chunk values, fuses the accumulator update and decides whether the
/// reduction folds horizontally.
void lowerReduction(DispatchRegisterGraph &graph);

/// phases, placement, the schedule, its live ranges, and every
/// node's footprint.
LogicalResult buildSchedule(DispatchRegisterGraph &graph);

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_DISPATCHREGISTERGRAPH_H_
