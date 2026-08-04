// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Adds profiler markers around the Hexagon memory operations
// hexagonmem.alloc/dealloc/copy and around the outermost compute loops.

#include "cellar-hexagon/CodeGen/Passes.h"

#include "cellar-hexagon/CodeGen/IR/HexagonDialect.h"
#include "cellar-hexagon/CodeGen/IR/HexagonOps.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include <optional>

namespace mlir::iree_compiler::cellar_hexagon::codegen {
namespace IREE = mlir::iree_compiler::IREE;

#define GEN_PASS_DEF_INSERTPROFILERMARKERSPASS
#include "cellar-hexagon/CodeGen/Passes.h.inc"

namespace {

// The `extra_info` strings identifying the marked operation in the profile.
constexpr llvm::StringLiteral kAllocMarker = "kernel_allocation";
constexpr llvm::StringLiteral kFreeMarker = "kernel_free";
constexpr llvm::StringLiteral kCopyMarker = "memref_copy";
constexpr llvm::StringLiteral kHexagonMemAllocMarker = "hexagonmem.alloc";
constexpr llvm::StringLiteral kHexagonMemDeallocMarker = "hexagonmem.dealloc";
constexpr llvm::StringLiteral kHexagonMemCopyMarker = "hexagonmem.copy";
constexpr llvm::StringLiteral kComputeInnerLoopMarker = "compute.inner_loop";

Value insertGetRuntimeState(IRRewriter &rewriter,
                            mlir::FunctionOpInterface funcOp) {
  rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
  auto stateType = IREE::Hexagon::RuntimeStateType::get(rewriter.getContext());
  return IREE::Hexagon::GetRuntimeStateOp::create(rewriter, funcOp.getLoc(),
                                                  stateType)
      .getState();
}

Value insertMarkerBegin(IRRewriter &rewriter, Value state, Location loc,
                        IREE::Hexagon::ProfilerZone zoneType,
                        StringRef extraInfo) {
  auto recordType =
      IREE::Hexagon::ProfilerRecordType::get(rewriter.getContext());
  return IREE::Hexagon::ProfilerBeginOp::create(
             rewriter, loc, recordType, state,
             IREE::Hexagon::ProfilerZoneAttr::get(rewriter.getContext(),
                                                  zoneType),
             rewriter.getStringAttr(extraInfo))
      .getRecord();
}

void insertMarkerEnd(IRRewriter &rewriter, Location loc, Value record) {
  IREE::Hexagon::ProfilerEndOp::create(rewriter, loc, record);
}

// An operation to wrap with a profiler marker zone, together with the zone type
// it is attributed to and the `extra_info` string identifying it in the
// profile.
struct MarkerTarget {
  Operation *op;
  IREE::Hexagon::ProfilerZone zone;
  StringRef extraInfo;
};

void wrapOpWithMarker(IRRewriter &rewriter, Value state,
                      const MarkerTarget &markerTarget) {
  Operation *op = markerTarget.op;
  rewriter.setInsertionPoint(op);
  Value record = insertMarkerBegin(rewriter, state, op->getLoc(),
                                   markerTarget.zone, markerTarget.extraInfo);

  rewriter.setInsertionPointAfter(op);
  insertMarkerEnd(rewriter, op->getLoc(), record);
}

bool isHexagonMemOp(Operation *op) {
  return op->getName().getDialectNamespace() ==
         mlir::hexagonmem::HexagonMemDialect::getDialectNamespace();
}

bool containsHexagonMemOp(Operation *op) {
  bool containsHexagonMem = false;
  op->walk([&](Operation *nestedOp) {
    if (nestedOp == op) {
      return WalkResult::advance();
    }
    if (isHexagonMemOp(nestedOp)) {
      containsHexagonMem = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return containsHexagonMem;
}

bool isOuterMostOpWithoutHexagonMemOps(Operation *op) {
  Operation *parent = op->getParentOp();
  // The ops being targeted are expected to be in a function
  if (!parent)
    return false;

  return containsHexagonMemOp(parent) && !containsHexagonMemOp(op);
}

std::optional<MarkerTarget> getMarkerTargetForOp(Operation *op) {
  if (isa<mlir::memref::AllocOp>(op)) {
    return MarkerTarget{op, IREE::Hexagon::ProfilerZone::MemoryManagement,
                        kAllocMarker};
  }
  if (isa<mlir::memref::DeallocOp>(op)) {
    return MarkerTarget{op, IREE::Hexagon::ProfilerZone::MemoryManagement,
                        kFreeMarker};
  }
  if (isa<mlir::memref::CopyOp>(op)) {
    return MarkerTarget{op, IREE::Hexagon::ProfilerZone::MemoryManagement,
                        kCopyMarker};
  }
  if (isa<mlir::hexagonmem::AllocOp>(op)) {
    return MarkerTarget{op, IREE::Hexagon::ProfilerZone::MemoryManagement,
                        kHexagonMemAllocMarker};
  }
  if (isa<mlir::hexagonmem::DeallocOp>(op)) {
    return MarkerTarget{op, IREE::Hexagon::ProfilerZone::MemoryManagement,
                        kHexagonMemDeallocMarker};
  }
  if (isa<mlir::hexagonmem::CopyOp>(op)) {
    return MarkerTarget{op, IREE::Hexagon::ProfilerZone::MemoryManagement,
                        kHexagonMemCopyMarker};
  }
  if (isa<mlir::scf::ForOp>(op)) {
    if (isOuterMostOpWithoutHexagonMemOps(op)) {
      return MarkerTarget{op, IREE::Hexagon::ProfilerZone::Marker,
                          kComputeInnerLoopMarker};
    }
  }
  return std::nullopt;
}

struct InsertProfilerMarkersPass final
    : public impl::InsertProfilerMarkersPassBase<InsertProfilerMarkersPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<IREE::Hexagon::IREEHexagonDialect,
                    mlir::hexagonmem::HexagonMemDialect>();
  }

  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();

    // Collect the targets first so the walk is not perturbed by the markers
    // inserted around each operation.
    llvm::SmallVector<MarkerTarget> markerTargets;
    funcOp.walk([&](Operation *op) {
      if (std::optional<MarkerTarget> markerTarget = getMarkerTargetForOp(op)) {
        markerTargets.push_back(*markerTarget);
      }
    });
    if (markerTargets.empty()) {
      return;
    }

    mlir::IRRewriter rewriter(funcOp->getContext());
    Value state = insertGetRuntimeState(rewriter, funcOp);
    for (const MarkerTarget &markerTarget : markerTargets) {
      wrapOpWithMarker(rewriter, state, markerTarget);
    }
  };
};

} // namespace

} // namespace mlir::iree_compiler::cellar_hexagon::codegen
