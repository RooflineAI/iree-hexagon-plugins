// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/CodeGen/IR/HexagonAttrs.h"
#include "cellar-hexagon/CodeGen/IR/HexagonDialect.h"
#include "cellar-hexagon/CodeGen/IR/HexagonOps.h"
#include "cellar-hexagon/CodeGen/Passes.h"
#include "cellar-hexagon/CodeGen/Strategy/KernelDispatch.h"

#include "iree/compiler/Codegen/Common/TileAndFuseUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "iree-hexagon-vtcm-tiling"

namespace mlir::iree_compiler::cellar_hexagon::codegen {
namespace {

namespace IREEHexagon = mlir::iree_compiler::IREE::Hexagon;

#define GEN_PASS_DEF_HEXAGONVTCMTILINGPASS
#include "cellar-hexagon/CodeGen/Passes.h.inc"

static bool isEmptyBackedTensor(Value tensor) {
  if (tensor.getDefiningOp<tensor::EmptyOp>()) {
    return true;
  }
  if (auto extractSliceOp = tensor.getDefiningOp<tensor::ExtractSliceOp>()) {
    return isEmptyBackedTensor(extractSliceOp.getSource());
  }
  auto blockArg = dyn_cast<BlockArgument>(tensor);
  if (!blockArg) {
    return false;
  }
  auto forallOp = dyn_cast<scf::ForallOp>(blockArg.getOwner()->getParentOp());
  if (!forallOp || blockArg.getArgNumber() < forallOp.getRank()) {
    return false;
  }
  return isEmptyBackedTensor(
      forallOp.getOutputs()[blockArg.getArgNumber() - forallOp.getRank()]);
}

static Value stageToVTCM(IRRewriter &rewriter, Value tensor, Location loc) {
  auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType) {
    return Value();
  }
  if (isEmptyBackedTensor(tensor)) {
    // Allow for dynamic dimensions when imperfect VTCM tiling happens.
    // Carrying these values as operands prevents LICM from hoisting VTCMEmptyOp
    // until the size values are defined.
    SmallVector<Value> dynamicSizes;
    for (auto [dimIdx, dimSize] : llvm::enumerate(tensorType.getShape())) {
      if (ShapedType::isDynamic(dimSize)) {
        dynamicSizes.push_back(
            rewriter.createOrFold<tensor::DimOp>(loc, tensor, dimIdx));
      }
    }
    return IREEHexagon::VTCMEmptyOp::create(rewriter, loc, tensorType,
                                            dynamicSizes)
        .getResult();
  }
  return IREEHexagon::StageToVTCMOp::create(rewriter, loc, tensorType, tensor)
      .getResult();
}

// This pass currently implements only the first VTCM-tiling step: create an
// outer `scf.forall` over the configured op and fuse the surrounding dispatch
// into it using the same SCF tiling/fusion utilities that IREE uses elsewhere.
// The VTCM copies are intentionally deferred to a follow-up step.

/// Returns the dispatch tile sizes for the outer `scf.forall`, zeroing
/// reduction dimensions so this pass only tiles parallel loops.
static SmallVector<OpFoldResult>
getForallTileSizes(OpBuilder &builder, linalg::LinalgOp op,
                   ArrayRef<int64_t> tileSizes) {
  SmallVector<OpFoldResult> result;
  for (auto [tileSize, iterType] :
       llvm::zip_equal(tileSizes, op.getIteratorTypesArray())) {
    int64_t effectiveSize =
        (iterType == utils::IteratorType::reduction) ? 0 : tileSize;
    result.push_back(builder.getIndexAttr(effectiveSize));
  }
  return result;
}

/// Returns true if any result of `producer` flows into `user` as a DPS init
/// operand (output buffer, not read-only input). Used to exclude in-place
/// output edges when deciding which ops need results yielded from the forall.
static bool isUsedAsInit(Operation *producer, Operation *user) {
  auto dpsIface = dyn_cast<DestinationStyleOpInterface>(user);
  if (!dpsIface) {
    return false;
  }
  ValueRange results = producer->getResults();
  return llvm::any_of(dpsIface.getDpsInits(), [&](Value operand) {
    return llvm::is_contained(results, operand);
  });
}

/// Returns the subset of `ops` whose results must be yielded through the
/// forall loop body. An op must yield when it has at least one non-init use
/// that either lives outside the fused dispatch region (`!ops.contains(user)`)
/// or is dominated by `rootOp` (i.e., a user that comes after the point where
/// the forall will be inserted). Init-only uses are excluded because they
/// follow the output tensor chain rather than a data-flow edge.
/// Fully copied from LLVMCPUTileAndFuseProducerConsumer.cpp
static llvm::DenseSet<Operation *>
getOpsNeedingYieldReplacements(Operation *rootOp,
                               const llvm::SmallDenseSet<Operation *> &ops) {
  mlir::DominanceInfo dominanceInfo(rootOp);
  llvm::DenseSet<Operation *> yieldReplacementsFor;
  for (Operation *op : ops) {
    if (!isa<linalg::LinalgOp>(op)) {
      continue;
    }
    if (llvm::any_of(op->getUsers(), [&](Operation *user) {
          if (isUsedAsInit(op, user)) {
            return false;
          }
          return dominanceInfo.properlyDominates(rootOp, user) ||
                 !ops.contains(user);
        })) {
      yieldReplacementsFor.insert(op);
    }
  }
  return yieldReplacementsFor;
}

/// Applies the tiling step to `op`: create an outer dispatch-level
/// `scf.forall`, fuse producers into it, and then fuse consumers forward so the
/// loop spans the whole dispatch.
/// This dispatch-wide tiling is copied from
/// LLVMCPUTileAndFuseProducerConsumer.cpp.
LogicalResult
applyDispatchWideTiling(IRRewriter &rewriter, linalg::LinalgOp op,
                        IREEHexagon::VTCMTilingConfigAttr config) {
  ArrayRef<int64_t> tileSizes = config.getTileSizes();
  if (tileSizes.size() != op.getNumLoops()) {
    return op.emitOpError("expected VTCM tile size count to match loop count");
  }

  MLIRContext *context = rewriter.getContext();
  mlir::DominanceInfo dominanceInfo(op);
  llvm::SmallDenseSet<Operation *> tiledAndFusedOps;
  collectTiledAndFusedOps(op, tiledAndFusedOps);
  llvm::DenseSet<Operation *> yieldReplacementsFor =
      getOpsNeedingYieldReplacements(op, tiledAndFusedOps);

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(getForallTileSizes(rewriter, op, tileSizes));
  tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(tilingOptions);

  RewritePatternSet cleanupPatterns(context);
  tensor::ExtractSliceOp::getCanonicalizationPatterns(cleanupPatterns, context);
  tensor::DimOp::getCanonicalizationPatterns(cleanupPatterns, context);
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(cleanupPatterns);
  tensor::populateBubbleUpExtractSliceOpPatterns(cleanupPatterns);
  // When fusing pads we do not want to generate zeroSliceGuards when doing
  // workgroup tiling. In `GPUApplyTilingLevelPass` we do have an option called
  // `allowZeroSlices` that can control this but we do not want these
  // generated if workgroup tiling is happening first.
  cleanupPatterns.insert<linalg::ExtractSliceOfPadTensorSwapPattern>(
      context, [](tensor::ExtractSliceOp) { return /*zeroSliceGuard=*/false; });
  tileAndFuseOptions.cleanupPatterns =
      FrozenRewritePatternSet(std::move(cleanupPatterns));

  scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
      [&](tensor::ExtractSliceOp, OpResult originalProducer,
          bool) -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
    Operation *owner = originalProducer.getOwner();
    if (isa<tensor::PadOp>(owner)) {
      return std::nullopt;
    }
    return scf::SCFTileAndFuseOptions::ControlFnResult{
        yieldReplacementsFor.contains(owner)};
  };
  tileAndFuseOptions.setFusionControlFn(controlFn);

  rewriter.setInsertionPoint(op);
  FailureOr<scf::SCFTileAndFuseResult> tiledResults =
      scf::tileConsumerAndFuseProducersUsingSCF(
          rewriter, cast<TilingInterface>(op.getOperation()),
          tileAndFuseOptions);
  if (failed(tiledResults)) {
    return op.emitOpError("failed to tile root op with scf.forall");
  }

  // Perform the replacement of tiled and fused values.
  for (auto [origValue, replacement] : tiledResults->replacements) {
    Value replacementCopy = replacement;
    rewriter.replaceUsesWithIf(origValue, replacement, [&](OpOperand &use) {
      Operation *user = use.getOwner();
      return !isa<tensor::DimOp>(user) &&
             dominanceInfo.dominates(replacementCopy, user);
    });
  }

  // The VTCM tiling attribute should only be used by this pass, remove it
  for (Operation *tiledOp : tiledResults->tiledAndFusedOps) {
    tiledOp->removeAttr(kHexagonVTCMTilingConfigAttrName);
  }

  SmallVector<LoopLikeOpInterface> tilingLoops = tiledResults->loops;
  if (tilingLoops.empty() || tilingLoops.size() != 1 ||
      !isa<scf::ForallOp>(tilingLoops.front())) {
    return op.emitOpError("expected tiling to produce a single scf.forall");
  }

  FailureOr<std::queue<Operation *>> newFusionOpportunities =
      fuseConsumersIntoForall(rewriter,
                              tiledResults->tiledAndFusedOps.getArrayRef(),
                              tilingLoops, [&](Operation *consumer) {
                                return isa<linalg::LinalgOp>(consumer) &&
                                       tiledAndFusedOps.contains(consumer);
                              });
  if (failed(newFusionOpportunities)) {
    return op.emitOpError("failed to fuse consumers into scf.forall");
  }

  // Because we restrict to at most a single tilable consumer for yielding
  // a replacement, no new fusion opportunities will yield a replacement,
  // meaning there is no need to run consumer fusion again afterwards.
  // TODO: run producer and consumer fusion in one worklist.
  fuseProducersOfSlices(rewriter, *newFusionOpportunities, tileAndFuseOptions,
                        tilingLoops);

  return success();
}

/// Introduces explicit tensor-level VTCM staging markers around the tiled
/// dispatch:
///   - every tensor.extract_slice inside a forall body is marked for later
///     VTCM materialization and later uses of the slice are rewritten to the
///     staged value;
///   - the copy back to DDR is inserted during bufferization,
///     translated from the tensor.extract_slice and is therefore not managed
///     here
static LogicalResult introduceVTCMCopies(IRRewriter &rewriter,
                                         FunctionOpInterface funcOp) {
  SmallVector<tensor::ExtractSliceOp> extractSliceOps;

  funcOp.walk([&](scf::ForallOp forallOp) {
    forallOp.walk([&](tensor::ExtractSliceOp extractSliceOp) {
      extractSliceOps.push_back(extractSliceOp);
    });
  });

  for (tensor::ExtractSliceOp extractSliceOp : extractSliceOps) {
    rewriter.setInsertionPointAfter(extractSliceOp);
    Value vtcmStage = stageToVTCM(rewriter, extractSliceOp.getResult(),
                                  extractSliceOp.getLoc());
    if (!vtcmStage) {
      return extractSliceOp.emitOpError("expected ranked tensor slice result");
    }
    rewriter.replaceAllUsesExcept(extractSliceOp.getResult(), vtcmStage,
                                  vtcmStage.getDefiningOp());
  }

  return success();
}

struct HexagonVTCMTilingPass
    : impl::HexagonVTCMTilingPassBase<HexagonVTCMTilingPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREEHexagon::IREEHexagonDialect, linalg::LinalgDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    SmallVector<linalg::LinalgOp> candidates;
    funcOp.walk([&](linalg::LinalgOp op) {
      if (op->getAttrOfType<IREEHexagon::VTCMTilingConfigAttr>(
              kHexagonVTCMTilingConfigAttrName)) {
        candidates.push_back(op);
      }
    });
    if (candidates.size() > 1) {
      candidates[1].emitOpError()
          << "expected at most one VTCM tiling config per dispatch";
      return signalPassFailure();
    }

    IRRewriter rewriter(funcOp.getContext());
    for (linalg::LinalgOp op : candidates) {
      auto config = op->getAttrOfType<IREEHexagon::VTCMTilingConfigAttr>(
          kHexagonVTCMTilingConfigAttrName);
      if (!config) {
        continue;
      }
      if (failed(applyDispatchWideTiling(rewriter, op, config))) {
        return signalPassFailure();
      }
    }

    // Once tiling has been done, add tensor-level staging markers for VTCM
    // These need to be introduced here to avoid subsequent passes making
    // optimizations that would make them a lot harder to track again
    // (Ex: canonicalization might remove empty forall ops and the tensor slices
    // that are used as markers to introduce the copy operations)
    if (failed(introduceVTCMCopies(rewriter, funcOp))) {
      return signalPassFailure();
    }

    // Post-tiling cleanup: CSE equivalent slices/producers, fold
    // tensor.empty, canonicalize extract_slice chains, and remove other IR
    // remnants left by tileConsumerAndFuseProducersUsingSCF. This mirrors the
    // cleanup in IREE's TileAndDistributeToWorkgroupsPass, with CSE added
    // because VTCM staging can otherwise leave duplicated fused producers in
    // multi-use DAGs.
    {
      mlir::DominanceInfo dominanceInfo(funcOp);
      bool changed = false;
      mlir::eliminateCommonSubExpressions(rewriter, dominanceInfo,
                                          funcOp.getOperation(), &changed);
    }
    {
      RewritePatternSet patterns(funcOp.getContext());
      linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
      tensor::populateFoldTensorEmptyPatterns(patterns);
      funcOp.getContext()
          ->getOrLoadDialect<tensor::TensorDialect>()
          ->getCanonicalizationPatterns(patterns);
      scf::ForallOp::getCanonicalizationPatterns(patterns, funcOp.getContext());
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        funcOp.emitOpError("tiling canonicalizations failed");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createHexagonVTCMTilingPass() {
  return std::make_unique<HexagonVTCMTilingPass>();
}

} // namespace mlir::iree_compiler::cellar_hexagon::codegen
