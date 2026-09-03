// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "DecisionTrace.h"

#include "DispatchPlanTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "iree-hexagon-dispatch-planning"

namespace mlir::iree_compiler::hexagon::codegen::planning {
#ifndef NDEBUG
namespace {

llvm::raw_ostream &operator<<(llvm::raw_ostream &output, DecisionStage stage) {
  switch (stage) {
  case DecisionStage::Analysis:
    return output << "analysis";
  case DecisionStage::Strategy:
    return output << "strategy";
  case DecisionStage::Resource:
    return output << "resource";
  case DecisionStage::ComputeTile:
    return output << "compute-tile";
  case DecisionStage::Verification:
    return output << "verify";
  case DecisionStage::Encoding:
    return output << "encode";
  case DecisionStage::Fallback:
    return output << "fallback";
  }
  llvm_unreachable("unknown decision stage");
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &output, DecisionKind kind) {
  switch (kind) {
  case DecisionKind::Selected:
    return output << "selected";
  case DecisionKind::Adjusted:
    return output << "adjusted";
  case DecisionKind::Derived:
    return output << "derived";
  case DecisionKind::Rejected:
    return output << "rejected";
  case DecisionKind::Fallback:
    return output << "fallback";
  }
  llvm_unreachable("unknown decision kind");
}

} // namespace
#endif // NDEBUG

void DecisionTrace::record(DecisionStage stage, DecisionKind kind,
                           const llvm::Twine &message) {
  LLVM_DEBUG(llvm::dbgs() << "[hexagon-tiling][" << stage << "][" << kind
                          << "] " << message << '\n');
}

void DecisionTrace::recordForOp(DecisionStage stage, DecisionKind kind,
                                uint32_t opOrdinal, llvm::StringRef opName,
                                const llvm::Twine &message) {
  LLVM_DEBUG(llvm::dbgs() << "[hexagon-tiling][" << stage << "][" << kind
                          << "] op#" << opOrdinal << ' ' << opName << ": "
                          << message << '\n');
}

void DecisionTrace::recordTilePlan(DecisionStage stage, DecisionKind kind,
                                   uint32_t opOrdinal, llvm::StringRef opName,
                                   llvm::StringRef label,
                                   llvm::ArrayRef<TileDecision> tiles) {
  LLVM_DEBUG({
    llvm::raw_ostream &output = llvm::dbgs();
    output << "[hexagon-tiling][" << stage << "][" << kind << "] op#"
           << opOrdinal << ' ' << opName << ": " << label << "=[";
    llvm::interleaveComma(
        tiles, output, [&](const TileDecision &tile) { output << tile.size; });
    output << "]\n";
  });
}

void DecisionTrace::recordAdjustment(DecisionStage stage, uint32_t opOrdinal,
                                     llvm::StringRef opName,
                                     llvm::StringRef label, uint32_t localDim,
                                     int64_t before, int64_t after,
                                     llvm::StringRef reason) {
  LLVM_DEBUG(llvm::dbgs() << "[hexagon-tiling][" << stage << "]["
                          << DecisionKind::Adjusted << "] op#" << opOrdinal
                          << ' ' << opName << ": " << label << " dim#"
                          << localDim << ": " << before << " -> " << after
                          << " (" << reason << ")\n");
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning
