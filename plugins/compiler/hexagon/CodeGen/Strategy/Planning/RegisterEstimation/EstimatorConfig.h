// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_ESTIMATORCONFIG_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_ESTIMATORCONFIG_H_

#include "llvm/ADT/StringMap.h"

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

/// A candidate that chunks the reduction dim of a fused producer's result
/// needing a relayout will encuur high re-layout cost
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
      {"math.exp", OpExpansion{/*invariantRegisters=*/12,
                               /*extraTiles=*/1}},
      {"math.tanh", OpExpansion{/*invariantRegisters=*/6,
                                /*extraTiles=*/1}},
      {"math.rsqrt", OpExpansion{/*invariantRegisters=*/2,
                                 /*extraTiles=*/0}},
  };
  FusedRelayoutChunkPolicy fusedRelayoutChunkPolicy =
      FusedRelayoutChunkPolicy::Fail;
};

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_REGISTERESTIMATION_ESTIMATORCONFIG_H_
