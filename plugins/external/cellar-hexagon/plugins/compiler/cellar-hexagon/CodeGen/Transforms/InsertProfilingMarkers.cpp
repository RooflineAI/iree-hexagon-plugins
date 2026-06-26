// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/CodeGen/Passes.h"

#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingDialect.h"
#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingOps.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {
namespace IREE = mlir::iree_compiler::IREE;

#define GEN_PASS_DEF_INSERTPROFILINGMARKERSPASS
#include "cellar-hexagon/CodeGen/Passes.h.inc"

namespace {

// This marker is hardcoded here and the reference enum is in the runtime
constexpr int32_t kMarkerZoneType = 7;
constexpr llvm::StringLiteral kComputeInnerLoopMarker = "compute.inner_loop";
constexpr llvm::StringLiteral kHexagonMemAllocMarker = "hexagonmem.alloc";
constexpr llvm::StringLiteral kHexagonMemCopyMarker = "hexagonmem.copy";

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

Value insertMarkerBegin(IRRewriter &rewriter, Location loc,
                        StringRef extraInfo) {
  auto recordType =
      IREE::Hexagon::ProfilingRecordType::get(rewriter.getContext());
  return IREE::Hexagon::ProfilingBeginOp::create(
             rewriter, loc, recordType,
             rewriter.getI32IntegerAttr(kMarkerZoneType),
             rewriter.getStringAttr(extraInfo))
      .getRecord();
}

void insertMarkerEnd(IRRewriter &rewriter, Location loc, Value record) {
  IREE::Hexagon::ProfilingEndOp::create(rewriter, loc, record);
}

void wrapOpWithMarker(IRRewriter &rewriter, Operation *op,
                      StringRef extraInfo) {
  rewriter.setInsertionPoint(op);
  Value record = insertMarkerBegin(rewriter, op->getLoc(), extraInfo);

  rewriter.setInsertionPointAfter(op);
  insertMarkerEnd(rewriter, op->getLoc(), record);
}

struct InsertProfilingMarkersPass final
    : public impl::InsertProfilingMarkersPassBase<InsertProfilingMarkersPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<IREE::Hexagon::IREEHexagonEncodingDialect,
                    mlir::hexagonmem::HexagonMemDialect>();
  }

  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    llvm::SmallVector<mlir::hexagonmem::AllocOp> allocOps;
    llvm::SmallVector<mlir::hexagonmem::CopyOp> copyOps;
    llvm::SmallVector<mlir::scf::ForOp> computeLoops;
    funcOp.walk([&](Operation *op) {
      if (auto allocOp = dyn_cast<mlir::hexagonmem::AllocOp>(op)) {
        allocOps.push_back(allocOp);
        return;
      }
      if (auto copyOp = dyn_cast<mlir::hexagonmem::CopyOp>(op)) {
        copyOps.push_back(copyOp);
        return;
      }
      if (auto forOp = dyn_cast<mlir::scf::ForOp>(op)) {
        if (isOuterMostOpWithoutHexagonMemOps(forOp)) {
          computeLoops.push_back(forOp);
        }
      }
    });

    mlir::IRRewriter rewriter(&getContext());
    for (mlir::hexagonmem::AllocOp allocOp : allocOps) {
      wrapOpWithMarker(rewriter, allocOp, kHexagonMemAllocMarker);
    }
    for (mlir::hexagonmem::CopyOp copyOp : copyOps) {
      wrapOpWithMarker(rewriter, copyOp, kHexagonMemCopyMarker);
    }
    for (mlir::scf::ForOp forOp : computeLoops) {
      wrapOpWithMarker(rewriter, forOp, kComputeInnerLoopMarker);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::cellar_hexagon::codegen
