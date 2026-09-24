// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_DECISIONTRACE_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_DECISIONTRACE_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>

namespace mlir::iree_compiler::hexagon::codegen::planning {

struct TileDecision;

enum class DecisionStage {
  Analysis,
  Strategy,
  Resource,
  ComputeTile,
  Verification,
  Encoding,
  Fallback,
};

enum class DecisionKind {
  Selected,
  Adjusted,
  Derived,
  Rejected,
  Fallback,
};

/// Structured debug trace for heuristic decisions.
///
/// Implementations should prefer these methods over direct `llvm::errs()` or
/// ad hoc debug logging. Operation ordinals come from DispatchShape, keeping
/// output stable without dumping full operations. The trace is available in
/// debug-capable builds under `iree-hexagon-dispatch-planning`.
class DecisionTrace {
public:
  void record(DecisionStage stage, DecisionKind kind,
              const llvm::Twine &message);
  void recordForOp(DecisionStage stage, DecisionKind kind, uint32_t opOrdinal,
                   llvm::StringRef opName, const llvm::Twine &message);
  void recordTilePlan(DecisionStage stage, DecisionKind kind,
                      uint32_t opOrdinal, llvm::StringRef opName,
                      llvm::StringRef label,
                      llvm::ArrayRef<TileDecision> tiles);
  void recordAdjustment(DecisionStage stage, uint32_t opOrdinal,
                        llvm::StringRef opName, llvm::StringRef label,
                        uint32_t localDim, int64_t before, int64_t after,
                        llvm::StringRef reason);
};

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_DECISIONTRACE_H_
