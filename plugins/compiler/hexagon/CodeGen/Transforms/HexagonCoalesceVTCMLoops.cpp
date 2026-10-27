// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/IR/HexagonDialect.h"
#include "hexagon/CodeGen/IR/HexagonOps.h"
#include "hexagon/CodeGen/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-hexagon-coalesce-vtcm-loops"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONCOALESCEVTCMLOOPSPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

namespace IREEHexagon = mlir::iree_compiler::IREE::Hexagon;

bool isVTCMStagingOp(Operation *op) {
  return isa<IREEHexagon::StageToVTCMOp, IREEHexagon::VTCMEmptyOp>(op);
}

/// Returns the band of `stagingLoop`: the loops from the outermost loop that
/// perfectly nests it down to `stagingLoop`. A parent is part of the band if
/// its body holds nothing but the nested loop, it forwards its iter_args to
/// the nested loop and yields its results, and the bounds of the band do not
/// depend on it.
SmallVector<scf::ForOp> getBand(scf::ForOp stagingLoop) {
  SmallVector<scf::ForOp> band = {stagingLoop};
  while (auto parent = dyn_cast<scf::ForOp>(band.front()->getParentOp())) {
    scf::ForOp inner = band.front();
    Block *body = parent.getBody();
    if (!llvm::hasNItems(*body, 2) ||
        !llvm::equal(parent.getRegionIterArgs(), inner.getInitArgs()) ||
        !llvm::equal(body->getTerminator()->getOperands(),
                     inner.getResults())) {
      break;
    }
    bool boundsAbove = llvm::all_of(band, [&](scf::ForOp loop) {
      return areValuesDefinedAbove(ValueRange{loop.getLowerBound(),
                                              loop.getUpperBound(),
                                              loop.getStep()},
                                   parent.getRegion());
    });
    if (!boundsAbove)
      break;
    band.insert(band.begin(), parent);
  }
  return band;
}

/// Whether `iterArg` only carries the DDR result: it is sliced, written with
/// insert_slice, or forwarded to a nested loop that does the same with it.
bool carriesOnlyDDRResult(BlockArgument iterArg) {
  for (OpOperand &use : iterArg.getUses()) {
    Operation *user = use.getOwner();
    if (isa<tensor::ExtractSliceOp>(user))
      continue;
    if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(user);
        insertOp && &use == &insertOp.getDestMutable())
      continue;
    if (auto nestedLoop = dyn_cast<scf::ForOp>(user)) {
      BlockArgument nestedIterArg = nestedLoop.getTiedLoopRegionIterArg(&use);
      if (nestedIterArg && carriesOnlyDDRResult(nestedIterArg))
        continue;
    }
    return false;
  }
  return true;
}

/// Whether `loop` carries nothing but the DDR result from one iteration to
/// the next: its inits are not VTCM staging values and its iter_args only
/// carry the DDR result. A VTCM buffer of the tile then never holds data of a
/// previous tile.
bool carriesOnlyDDRResults(scf::ForOp loop) {
  for (auto [init, iterArg] :
       llvm::zip_equal(loop.getInitArgs(), loop.getRegionIterArgs())) {
    if (Operation *def = init.getDefiningOp(); def && isVTCMStagingOp(def))
      return false;
    if (!carriesOnlyDDRResult(iterArg))
      return false;
  }
  return true;
}

struct HexagonCoalesceVTCMLoopsPass
    : impl::HexagonCoalesceVTCMLoopsPassBase<HexagonCoalesceVTCMLoopsPass> {
  using Base::Base;

  void runOnOperation() override {
    // The loops that stage tiles into VTCM. Staging ops hoisted out of all
    // loops (e.g. a loop-invariant HMX operand) need no double buffering.
    llvm::SetVector<scf::ForOp> stagingLoops;
    getOperation()->walk([&](IREEHexagon::StageToVTCMOp stageOp) {
      if (auto forOp = dyn_cast<scf::ForOp>(stageOp->getParentOp()))
        stagingLoops.insert(forOp);
    });

    IRRewriter rewriter(&getContext());
    for (scf::ForOp stagingLoop : stagingLoops) {
      SmallVector<scf::ForOp> band = getBand(stagingLoop);
      // The outer loops of the band forward their iter_args to the nested
      // ones, so checking the outermost loop covers the band.
      if (!carriesOnlyDDRResults(band.front())) {
        LLVM_DEBUG(llvm::dbgs() << "VTCM data is carried across iterations, "
                                   "not marking: "
                                << stagingLoop << "\n");
        continue;
      }

      // coalesceLoops rewrites the outermost loop in place.
      if (band.size() > 1 && failed(coalesceLoops(rewriter, band))) {
        LLVM_DEBUG(llvm::dbgs() << "failed to coalesce the VTCM band of: "
                                << stagingLoop << "\n");
        continue;
      }
      band.front()->setAttr(kVTCMTileLoopAttrName, rewriter.getUnitAttr());
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen
