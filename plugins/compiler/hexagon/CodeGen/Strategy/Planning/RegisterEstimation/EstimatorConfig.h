// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_ESTIMATORCONFIG_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_ESTIMATORCONFIG_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <cstdint>
#include <memory>

namespace mlir::iree_compiler::hexagon::codegen::planning {

/// Where a loop-invariant value (a splatted constant or a captured scalar)
/// is charged.
enum class InvariantPlacement {
  /// Hoisted once and resident for the whole schedule. Default: it is what
  /// the backend does with a splat, and it is the higher count.
  AlwaysLive,
  /// Rematerialized at each use, so it is live only across its uses.
  LiveAtUsesOnly,
};

/// What a body op that is expanded into a longer instruction sequence holds
/// beyond its operands and result.
struct OpExpansion {
  /// Registers of constants the expansion splats. They are hoisted like any
  /// other invariant, and shared by every op of the same kind in the dispatch.
  int64_t invariantRegisters = 0;
  /// Intermediate tiles live while the op runs, each the size of its result.
  int64_t extraTiles = 0;
};

/// An accumulator update `acc <accumulate> <producer>(operands...)` that the
/// backend emits as one instruction writing the accumulator in place.
/// This means, intermediary results never materialize as registers
struct AccumulateFusion {
  /// Ops combining the accumulator with the produced value, e.g. "arith.addf".
  llvm::StringSet<> accumulateOps;
  /// Ops producing the value folded into the accumulator, e.g. "arith.mulf".
  llvm::StringSet<> producerOps;
  /// Ops on a producer operand the instruction absorbs because it reads the
  /// narrow operand natively, e.g. "arith.extsi". Absorbed only when the
  /// producer is their sole reader.
  llvm::StringSet<> absorbedOperandOps;
};

/// A candidate that chunks the reduction dim of a fused producer's result
/// needing a relayout will incur high re-layout cost
enum class FusedRelayoutChunkPolicy {
  /// `evaluate` fails: "could not estimate"
  Fail,
  /// `evaluate` estimates anyway, while still missing the cost for re-layout
  EstimateAnyway,
};

/// Bits in one HVX vector register.
/// unittests have a static assert to check this against TargetInfo
inline constexpr int64_t HexagonVectorBits = 1024;

/// Every modeling decision the estimator ues, default values are for hexagon
struct EstimatorConfig {
  /// Bits in one vector register.
  int64_t vectorBits = HexagonVectorBits;
  InvariantPlacement invariantPlacement = InvariantPlacement::AlwaysLive;
  /// Copies of the output tile a horizontal reduction holds while it folds
  /// across lanes: the running value and the shuffled copy added into it.
  int64_t horizontalReductionCopies = 2;
  /// Body ops, by name, that the backend expands into a sequence holding more
  /// than one tile.
  llvm::StringMap<OpExpansion> OpCosts = {
      // FIXME: Re-Evaluate these values, to match actual produced lowerings,
      // especially for the trigonometric functions
      {"math.exp", OpExpansion{/*invariantRegisters=*/12,
                               /*extraTiles=*/1}},
      {"math.rsqrt", OpExpansion{/*invariantRegisters=*/2,
                                 /*extraTiles=*/0}},
      {"math.tanh", OpExpansion{/*invariantRegisters=*/6,
                                /*extraTiles=*/1}},
      {"math.sin", OpExpansion{/*invariantRegisters=*/19,
                               /*extraTiles=*/6}},
      {"math.cos", OpExpansion{/*invariantRegisters=*/21,
                               /*extraTiles=*/6}},
  };

  /// Accumulator updates fused into one in-place node
  llvm::SmallVector<AccumulateFusion> accumulateFusions = {
      // Multiply-accumulate.
      {/*accumulateOps=*/{"arith.addf", "arith.addi"},
       /*producerOps=*/{"arith.mulf", "arith.muli"},
       /*absorbedOperandOps=*/{"arith.extsi", "arith.extui", "arith.extf"}},
      // Widening add-accumulate, e.g. Vxx.w += vadd(Vu.h, Vv.h).
      {/*accumulateOps=*/{"arith.addi"},
       /*producerOps=*/{"arith.addi"},
       /*absorbedOperandOps=*/{"arith.extsi", "arith.extui"}},
      // Shift-accumulate by a scalar amount, e.g. Vx.w += vasl(Vu.w, Rt),
      // Vx.w += vasr(Vu.w, Rt).
      {/*accumulateOps=*/{"arith.addi"},
       /*producerOps=*/{"arith.shli", "arith.shrsi"},
       /*absorbedOperandOps=*/{}},
      // Compare-accumulate into a predicate, e.g. Qx &= vcmp.gt(Vu.w, Vv.w),
      // also |= and ^=, for integer and hf/sf/bf compares.
      {/*accumulateOps=*/{"arith.andi", "arith.ori", "arith.xori"},
       /*producerOps=*/{"arith.cmpi", "arith.cmpf"},
       /*absorbedOperandOps=*/{}},
  };

  FusedRelayoutChunkPolicy fusedRelayoutChunkPolicy =
      FusedRelayoutChunkPolicy::Fail;
};

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_ESTIMATORCONFIG_H_
