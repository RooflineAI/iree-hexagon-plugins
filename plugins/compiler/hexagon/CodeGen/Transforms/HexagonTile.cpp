// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Derived from Codegen/LLVMCPU/LLVMCPUTile.cpp at IREE revision
// a45adeaa6115e446c898e6eb21fb6edc0e65ddc4. Fixed-width scheduling behavior is
// preserved.

#include "hexagon/CodeGen/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "iree-hexagon-tile"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONTILEPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

/// This pass tiles all the TilingInterface operations. The `tilingLevel` must
/// be specified. It picks the `tilingLevel`-th list as tiling sizes from
/// lowering_config.
struct HexagonTilePass : impl::HexagonTilePassBase<HexagonTilePass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, affine::AffineDialect,
                    linalg::LinalgDialect, scf::SCFDialect>();
  }

  void runOnOperation() override;
};

void HexagonTilePass::runOnOperation() {
  if (tilingLevel == IREE::CPU::TilingLevel::InvalidLevel) {
    LDBG() << "tilingLevel not set, skip tiling";
    return;
  }
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  for (auto computeOp : computeOps) {
    auto op = dyn_cast<TilingInterface>(computeOp);
    if (!op || op.getLoopIteratorTypes().empty()) {
      continue;
    }

    // For now do not tile `tensor.pad` operations. The `tensor.pad`
    // operations might be those introduced by the padding-based
    // codegeneration strategy. Those are not meant to be tiled again.
    // Need a better way for handling this, but this works for now.
    if (isa<tensor::PadOp>(computeOp)) {
      continue;
    }

    IREE::Codegen::LoweringConfigAttrInterface maybeLoweringConfig =
        getLoweringConfig(op);
    if (!maybeLoweringConfig) {
      LDBG() << "can't find lowering_config, skip tiling";
      continue;
    }
    if (!maybeLoweringConfig.hasTilingLevel(
            static_cast<unsigned>(tilingLevel.getValue()))) {
      LDBG() << "target tiling level does not exist";
      continue;
    }

    LDBG() << "candidate: " << op;
    if (skipRootOp && maybeLoweringConfig.hasWorkgroupTilingLevel()) {
      LDBG() << "skip tiling on the root op";
      continue;
    }

    auto tileSizesAttr = dyn_cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        getLoweringConfig(op).getTilingLevelAttr(
            static_cast<unsigned>(tilingLevel.getValue())));
    SmallVector<int64_t> tileSizes(tileSizesAttr.getSizes());
    tileSizes.resize(op.getLoopIteratorTypes().size(), 0);
    if (llvm::all_of(tileSizes, [](int64_t v) { return v == 0; })) {
      LDBG() << "tiling sizes are all zeros, skip tiling";
      continue;
    }

    IRRewriter rewriter(context);
    scf::SCFTilingOptions options{};
    options.setTileSizes(getAsIndexOpFoldResult(context, tileSizes));
    FailureOr<scf::SCFTilingResult> tiledResults =
        scf::tileUsingSCF(rewriter, op, options);
    if (failed(tiledResults)) {
      continue;
    }
    rewriter.replaceOp(op, tiledResults->replacements);
  }

  RewritePatternSet patterns(context);
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  context->getLoadedDialect<tensor::TensorDialect>()
      ->getCanonicalizationPatterns(patterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    LDBG() << "----- cleanup failed -----";
    return signalPassFailure();
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createHexagonTilePass(IREE::CPU::TilingLevel tilingLevel, bool skipRootOp) {
  HexagonTilePassOptions options;
  options.tilingLevel = tilingLevel;
  options.skipRootOp = skipRootOp;
  return std::make_unique<HexagonTilePass>(options);
}

} // namespace mlir::iree_compiler::hexagon::codegen
