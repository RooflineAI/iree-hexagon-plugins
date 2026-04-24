// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file owns the LLVMCPU-derived / IREE-oriented Hexagon lowering helpers
// and reusable subpipeline builders used by HexagonLowerExecutableTarget.

// For the time being, this file very closely mimics a fraction of passes.cpp
// file from LLVMCPU. It is therefore relevant, at least for now, to open a diff
// of these two files to see the hexagon specific differences in case you are
// curious.

#include "cellar-hexagon/CodeGen/Pipelines/IreeLoweringPipelines.h"
#include "cellar-hexagon/CodeGen/Conversion/HexagonConvertToLLVM.h"
#include "cellar-hexagon/CodeGen/Pipelines/Bufferization.h"
#include "cellar-hexagon/CodeGen/Pipelines/TranslationPipeline.h"
#include "hexagon/Conversion/HexKLToLLVM/HexKLToLLVM.h"
#include "hexagon/Conversion/HexagonMemToLLVM/HexagonMemToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Transforms/Transforms.h"

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "iree-hexagon-pass-pipelines"

namespace mlir::iree_compiler::cellar_hexagon::codegen {
namespace IREE = mlir::iree_compiler::IREE;

static llvm::cl::opt<bool> clHexagonPatchFuncOps(
    "iree-hexagon-debug-patch-func-ops",
    llvm::cl::desc(
        "Perform the patches on func ops for debugging purpose. It should be "
        "used with `--iree-codegen-debug-patched-func-ops-file-name`."),
    llvm::cl::init(false), llvm::cl::Hidden);

// Duplicate the LLVMCPU development flags with Hexagon-specific names to avoid
// cl::opt collisions with the upstream CPU backend while keeping the knobs
// available for future tweaking.
llvm::cl::opt<bool> clHexagonFailOnOutOfBoundsStackAllocation(
    "iree-hexagon-fail-on-out-of-bounds-stack-allocation",
    llvm::cl::desc("Fail if the upper bound of dynamic stack allocation cannot "
                   "be solved"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clHexagonFailOnLargeVector(
    "iree-hexagon-fail-on-large-vector",
    llvm::cl::desc("Fail if there are operations with large vectors"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clHexagonCheckLinalgVectorization(
    "iree-hexagon-check-linalg-vectorization",
    llvm::cl::desc(
        "Runs the pass to check if all the Linalg ops are vectorized"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clHexagonUseFastMinMaxOps(
    "iree-hexagon-use-fast-min-max-ops",
    llvm::cl::desc(
        "Use `arith.minf/maxf` instead of `arith.minimumf/maximumf` ops"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clHexagonEnableReassociateFpReductions(
    "iree-hexagon-reassociate-fp-reductions",
    llvm::cl::desc("Enables reassociation for FP reductions"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clHexagonSkipIntermediateRoundings(
    "iree-hexagon-skip-intermediate-roundings",
    llvm::cl::desc(
        "Allow skipping intermediate roundings. For example, in f16 matmul "
        "kernels on targets with only f32 arithmetic, we have to perform each "
        "multiply-accumulate in f32, and if this flag is false, then we have "
        "to round those f32 accumulators to the nearest f16 every time, which "
        "is slow."),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clHexagonInstrumentMemoryAccesses{
    "iree-hexagon-instrument-memory-accesses",
    llvm::cl::desc("Instrument memory accesses in dispatches when dispatch "
                   "instrumentation is enabled."),
    llvm::cl::init(false)};

static llvm::cl::opt<bool> clHexagonEnableVectorContractCustomKernels(
    "iree-hexagon-enable-vector-contract-custom-kernels",
    llvm::cl::desc("Enables vector contract custom kernels for "
                   "LLVMCPUMmt4dVectorLowering pass."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clHexagonTileDispatchUsingForall(
    "iree-hexagon-tile-dispatch-using-forall",
    llvm::cl::desc("Enable tile and distribute to workgroups using scf.forall"),
    llvm::cl::init(true));

bool isHexagonFailOnOutOfBoundsStackAllocationEnabled() {
  return clHexagonFailOnOutOfBoundsStackAllocation;
}

void buildHexagonIreeTranslationRoute(
    OpPassManager &variantPassManager,
    const HexagonPipelineOptions &pipelineOpt) {
  (void)pipelineOpt;
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  modulePassManager.addPass(createLowerExecutableUsingTransformDialectPass());
  FunctionLikeNest(modulePassManager)
      .addPass(createHexagonLowerExecutableTargetPass)
      .addPass(createVerifyWorkgroupDistributionPass);
  if (clHexagonPatchFuncOps) {
    modulePassManager.addPass(createPatchFuncOpsPass());
  }
}

void addHexagonTileAndDistributePasses(
    OpPassManager &funcPassManager, const HexagonPipelineOptions &pipelineOpt) {
  if (pipelineOpt.disableDistribution) {
    return;
  }
  if (clHexagonTileDispatchUsingForall) {
    funcPassManager.addPass(
        createTileAndDistributeToWorkgroupsUsingForallOpPass());
    funcPassManager.addPass(createBufferizeDispatchTensorLoadStorePass());
    funcPassManager.addPass(createCombineResultLayoutTransformationPass());
  } else {
    funcPassManager.addPass(createTileAndDistributeToWorkgroupsPass());
    funcPassManager.addPass(createCSEPass());
    funcPassManager.addPass(createConvertToDestinationPassingStylePass());
    funcPassManager.addPass(createFoldAffineMinInDistributedLoopsPass());
  }
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createFuseTensorPadWithConsumerPass());
  funcPassManager.addPass(createConcretizePadResultShapePass());
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
}

namespace {

//===---------------------------------------------------------------------===//
// Codegen pipelines.
//===---------------------------------------------------------------------===//

static void buildHexagonVectorLoweringPipeline(
    OpPassManager &funcPassManager,
    const HexagonVectorLoweringPassOptions &options) {
  funcPassManager.addPass(createDropVectorUnitDimsPass());
  funcPassManager.addPass(createLLVMCPUVirtualVectorLoweringPass(
      LLVMCPUVirtualVectorLoweringPassOptions{options.splitVectorTransfersTo}));

  // Make sure we remove redundant vector ops (e.g., vector transposes) before
  // we lower them and can't be optimized away anymore.
  funcPassManager.addPass(createCanonicalizerPass());

  // We disable scalable lowerings for Hexagon
  VectorTransferLoweringPassOptions transferLoweringOptions{false};
  funcPassManager.addPass(
      createVectorTransferLoweringPass(transferLoweringOptions));
  funcPassManager.addPass(createLLVMCPUVectorTransposeLoweringPass(
      // This disables special lowering patterns that are useless for Hexagon
      LLVMCPUVectorTransposeLoweringPassOptions{false}));

  // Potentially removes shape_cast and broadcast on unit dims before shape_cast
  // lowering.
  funcPassManager.addPass(createCanonicalizerPass());

  // 'vector.shape_cast' are very expensive operations that are even generated
  // by some of the lowerings above (e.g., transpose lowering). There are
  // chances to cancel them out if they are not lowered too early so we lower
  // them at the very end of the pass.
  funcPassManager.addPass(createLLVMCPUVectorShapeCastLoweringPass());
}

} // namespace

// TODO: Needs further verification, but this pipeline is called
// on vector operations from what I have seen so far.
// It is also very similar to hexagon-mlir by itself.
void addHexagonBufferOpsTileAndVectorizePipeline(
    OpPassManager &funcPassManager, const HexagonPipelineOptions &pipelineOpt) {
  addHexagonTileAndDistributePasses(funcPassManager, pipelineOpt);

  // Skip tiling reduction loops because this is expected to apply on copy ops
  // only.
  funcPassManager.addPass(createLLVMCPUTilePass(
      IREE::CPU::TilingLevel::VectorCommonParallelTiles, /*skipRootOp=*/false));
  funcPassManager.addPass(createLLVMCPUPeelPass());
  {
    GenericVectorizationPassOptions options;
    options.useConfiguredVectorSizes = pipelineOpt.useConfiguredVectorSizes;
    options.enableVectorMasking = pipelineOpt.enableVectorMasking;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    // TODO: Think and test if additional changes are needed for Hexagon.
    // This verification is intended to keep the number of vectors from growing
    // excessively large. When this happens, performance may be degraded because
    // of register spilling and we might additionally risk excessively growing
    // the stack. Hexagon has 32 vector registers, which is the same as, for
    // example, avx512 for which the CPU lowering pipeline and these passes are
    // designed. The only difference is that the actual vectors are twice as
    // big. Note that this pass establishes the maximum size through
    // nativeVectorSize * iree-llvmcpu-max-allowed-number-of-native-vectors
    // Note that this pass is used in all pipelines, not only here.
    if (clHexagonFailOnLargeVector) {
      funcPassManager.addPass(createLLVMCPUVerifyVectorSizeLegalityPass());
    }
  }

  // Run IREE specific passes before vector lowering expert.
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  {
    HexagonVectorLoweringPassOptions options;
    options.splitVectorTransfersTo = "linalg-copy";
    buildHexagonVectorLoweringPipeline(funcPassManager, options);
  }
}

// This is the pipeline executed when CPUDoubleTilingExpert is configured. This
// is the pipeline that executes, for example, when lowering linalg.batch_matmul
// operations that are lowered into linalg.generics in the current state.
// TODO: I believe this pipeline might be interesting given that it contains
// tiling for cache and vector dimensions. In our current execution model, we
// intend to move data destined to the vector unit (like linalg.generic) through
// the cache and it is therefore interesting to keep. To be discussed and
// analyzed (performance-wise!)

// TODO: Also need to verify where the cache size is currently defined
void addHexagonMultiTilingExpertPassPipeline(
    OpPassManager &funcPassManager,
    IREE::Codegen::LoweringConfigAttrInterface loweringConfig,
    const HexagonPipelineOptions &pipelineOpt) {
  // TODO: I still need a way to combine VTCM tiling into this multilevel tiling
  // here.

  if (isHexagonVTCMTilingEnabled()) {
    // Align with hexagon-mlir staging by normalizing tensor shapes before VTCM
    // tiling. This helps avoid generating tiny loop-carried tensors that are
    // copied back to default memory space inside the same loop.
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    funcPassManager.addPass(createLinalgFoldUnitExtentDimsPass());
    funcPassManager.addPass(mlir::hexagon::createVTCMTilingPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    funcPassManager.addPass(createLinalgFoldUnitExtentDimsPass());
  }

  addHexagonTileAndDistributePasses(funcPassManager, pipelineOpt);

  for (int i : IREE::CPU::getTilingLevelsAsInts()) {
    if (!loweringConfig.hasTilingLevel(i)) {
      continue;
    }
    auto level = static_cast<IREE::CPU::TilingLevel>(i);
    switch (level) {
    case IREE::CPU::TilingLevel::CacheParallelTiles:
    case IREE::CPU::TilingLevel::VectorCommonParallelTiles:
      funcPassManager.addPass(
          createLLVMCPUTileAndFuseProducerConsumerPass(level));
      break;
    case IREE::CPU::TilingLevel::CacheReductionTiles:
      funcPassManager.addPass(
          createLLVMCPUTileRootAndFuseInputOperandsPass(level));
      break;
    case IREE::CPU::TilingLevel::VectorReductionTiles:
      // Run SplitReductionPass before the final reduction Fuse pass, because
      // SplitReductionPass takes care of banked-tiling.
      funcPassManager.addPass(createLLVMCPUSplitReductionPass(
          clHexagonEnableReassociateFpReductions));
      funcPassManager.addPass(
          createLLVMCPUTileRootAndFuseInputOperandsPass(level));
      // Tile all the reduction ops for target vector sizes, which ensures
      // that all the dimensions are tiled in all the reduction ops. The root
      // op is already tiled, so it is skipped in the pass.
      funcPassManager.addPass(createLLVMCPUTilePass(
          static_cast<IREE::CPU::TilingLevel>(i), /*skipRootOp=*/true));
      break;
    case IREE::CPU::TilingLevel::VectorInnerParallelTiles:
    case IREE::CPU::TilingLevel::DistributionTiles:
    case IREE::CPU::TilingLevel::MaxNumTileLevels:
    case IREE::CPU::TilingLevel::InvalidLevel:
      continue;
    };
    funcPassManager.addPass(createFuseTensorPadWithConsumerPass());
    funcPassManager.addPass(createConcretizePadResultShapePass());
  }

  // `VectorInnerParallelTiles` level models the tiling and fusion for the
  // dimensions that are not captured in root op. I.e., root op may not have the
  // config for the level. Thus, we use the last operation that has the tiling
  // level as anchor.
  funcPassManager.addPass(createLLVMCPUTileLastOpAndFuseProducerConsumerPass(
      IREE::CPU::TilingLevel::VectorInnerParallelTiles));
  funcPassManager.addPass(createFuseTensorPadWithConsumerPass());
  funcPassManager.addPass(createConcretizePadResultShapePass());

  funcPassManager.addPass(createForallToForPass());
  if (pipelineOpt.enablePeeling) {
    funcPassManager.addPass(createLLVMCPUPeelPass());
  }

  {
    funcPassManager.addPass(createTensorToVectorVectorizePadPass());
    if (pipelineOpt.decomposePackUnPackOps) {
      funcPassManager.addPass(createDecomposePackUnPackOpsPass());
      funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
      funcPassManager.addPass(createCSEPass());
    }
    funcPassManager.addPass(createLLVMCPUTileToVectorSizePass());

    GenericVectorizationPassOptions options;
    options.useConfiguredVectorSizes = pipelineOpt.useConfiguredVectorSizes;
    options.enableVectorMasking = pipelineOpt.enableVectorMasking;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    if (clHexagonFailOnLargeVector) {
      funcPassManager.addPass(createLLVMCPUVerifyVectorSizeLegalityPass());
    }
  }

  addHexagonBufferizePasses(funcPassManager);

  // Run IREE specific passes before vector lowering expert.
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  {
    HexagonVectorLoweringPassOptions options;
    options.splitVectorTransfersTo = "linalg-copy";
    buildHexagonVectorLoweringPipeline(funcPassManager, options);
  }
}

// TODO: This whole pipeline could possibly be removed altogether. I decided to
// keep it in case there is some unexpected lowering going through it, but this
// might be useless.

// TODO: Hexagon-mlir has its own convolution tiling pass. This should be tested
// and compared to the performance from this pipeline. Right now, hexagon-mlir's
// convolution operation is not included
void addHexagonConvTileAndDecomposeExpertPassPipeline(
    OpPassManager &funcPassManager, const HexagonPipelineOptions &pipelineOpt) {
  addHexagonTileAndDistributePasses(funcPassManager, pipelineOpt);

  funcPassManager.addPass(createLLVMCPUTileAndFuseProducerConsumerPass(
      IREE::CPU::TilingLevel::VectorCommonParallelTiles));
  funcPassManager.addPass(createFuseTensorPadWithConsumerPass());
  funcPassManager.addPass(createConcretizePadResultShapePass());

  funcPassManager.addPass(createLLVMCPUTileRootAndFuseInputOperandsPass(
      IREE::CPU::TilingLevel::VectorReductionTiles));
  funcPassManager.addPass(createDecomposeConvolutionToLowerDimOpsPass());
  funcPassManager.addPass(createFuseTensorPadWithConsumerPass());
  funcPassManager.addPass(createConcretizePadResultShapePass());

  // Convert forall to for before vectorization preparation.
  funcPassManager.addPass(iree_compiler::createForallToForPass());

  if (pipelineOpt.enablePeeling) {
    funcPassManager.addPass(createLLVMCPUPeelPass());
  }

  {
    funcPassManager.addPass(createTensorToVectorVectorizePadPass());
    GenericVectorizationPassOptions options;
    options.useConfiguredVectorSizes = pipelineOpt.useConfiguredVectorSizes;
    options.enableVectorMasking = pipelineOpt.enableVectorMasking;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    if (clHexagonFailOnLargeVector) {
      funcPassManager.addPass(createLLVMCPUVerifyVectorSizeLegalityPass());
    }
  }

  // Eliminate redundant transfer_read/write to avoid stack allocations.
  funcPassManager.addPass(createOptimizeVectorTransferPass(
      OptimizeVectorTransferPassOptions{/*flatten=*/true}));

  addHexagonBufferizePasses(funcPassManager);

  // Run IREE specific passes before vector lowering expert.
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  {
    HexagonVectorLoweringPassOptions options;
    // TODO: During the meeting, we mentioned that this pipeline was inefficient
    // and could be ignored, but it looks like it even has mistakes on it...
    // This shuffle is nonsensical and does not get used. Copied from LLVMCPU
    options.splitVectorTransfersTo = "shuffle";
    buildHexagonVectorLoweringPipeline(funcPassManager, options);
  }
}

// TODO: Decide what to do with this lowering pipeline. This lowering pipeline
// takes special care for linalg.matmul and batch_matmul when data tiling is
// enabled. Since we are disabling iree's data tiling and managing through
// hexagon-mlir's passes that take advantage of the VTCM and vector unit, this
// pipeline should be useless for hexagon.
void addHexagonMmt4dTilingExpertPassPipeline(
    OpPassManager &funcPassManager, const HexagonPipelineOptions &pipelineOpt) {
  addHexagonTileAndDistributePasses(funcPassManager, pipelineOpt);

  funcPassManager.addPass(createLLVMCPUTileAndFuseProducerConsumerPass(
      IREE::CPU::TilingLevel::VectorCommonParallelTiles));

  funcPassManager.addPass(createLLVMCPUTileRootAndFuseInputOperandsPass(
      IREE::CPU::TilingLevel::VectorReductionTiles));
  funcPassManager.addPass(iree_compiler::createForallToForPass());
  funcPassManager.addPass(createLLVMCPUTileToVectorSizePass());

  {
    GenericVectorizationPassOptions options;
    options.useConfiguredVectorSizes = pipelineOpt.useConfiguredVectorSizes;
    options.enableVectorMasking = pipelineOpt.enableVectorMasking;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    if (clHexagonFailOnLargeVector) {
      funcPassManager.addPass(createLLVMCPUVerifyVectorSizeLegalityPass());
    }
  }

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addHexagonBufferizePasses(funcPassManager);

  // Vector lowering of Mmt4d.
  funcPassManager.addPass(createLLVMCPUMmt4dVectorLoweringPass(
      LLVMCPUMmt4dVectorLoweringPassOptions{
          clHexagonEnableVectorContractCustomKernels}));

  // Generic vector lowering.
  HexagonVectorLoweringPassOptions options;
  options.splitVectorTransfersTo = "linalg-copy";
  buildHexagonVectorLoweringPipeline(funcPassManager, options);
}

// TODO: This pipeline is used when only data layout transformations are needed
// but no reduction happens (only linalg.pack/unpack ops). This is currently
// completely unused for hexagon-mlir (or at least should be), since we disabled
// data tiling in order to make use of hexagon-mlir's passes.
void addHexagonDataTilingPipeline(OpPassManager &funcPassManager,
                                  const HexagonPipelineOptions &pipelineOpt) {
  addHexagonTileAndDistributePasses(funcPassManager, pipelineOpt);

  // funcPassManager.addPass(createCPUPrepareUkernelsPass());
  // funcPassManager.addPass(
  //     createCPULowerToUKernelsPass(clHexagonSkipIntermediateRoundings));

  funcPassManager.addPass(createLLVMCPUTilePass(
      IREE::CPU::TilingLevel::VectorCommonParallelTiles, /*skipRootOp=*/false));
  if (pipelineOpt.decomposePackUnPackOps) {
    funcPassManager.addPass(createDecomposePackUnPackOpsPass());
  }

  {
    GenericVectorizationPassOptions options;
    options.useConfiguredVectorSizes = pipelineOpt.useConfiguredVectorSizes;
    options.enableVectorMasking = pipelineOpt.enableVectorMasking;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    if (clHexagonFailOnLargeVector) {
      funcPassManager.addPass(createLLVMCPUVerifyVectorSizeLegalityPass());
    }
  }

  addHexagonBufferizePasses(funcPassManager);

  {
    HexagonVectorLoweringPassOptions options;
    options.splitVectorTransfersTo = "linalg-copy";
    buildHexagonVectorLoweringPipeline(funcPassManager, options);
  }
}

// TODO: This lowering pipeline is supposed to be used when linalgExt ops are
// present (attention, FFT, Winograd). It looks like it might still be useful
// for now, but again, decide this later.
void addHexagonLinalgExtTileAndVectorizePipeline(
    OpPassManager &funcPassManager, const HexagonPipelineOptions &pipelineOpt) {
  addHexagonTileAndDistributePasses(funcPassManager, pipelineOpt);
  funcPassManager.addPass(createLLVMCPUTileAndFuseProducerConsumerPass(
      IREE::CPU::TilingLevel::VectorCommonParallelTiles));
  funcPassManager.addPass(
      IREE::LinalgExt::createConvertAttentionToOnlineAttentionPass());
  funcPassManager.addPass(createLLVMCPUTileRootAndFuseInputOperandsPass(
      IREE::CPU::TilingLevel::VectorReductionTiles));
  funcPassManager.addPass(
      IREE::LinalgExt::createDecomposeWinogradTransformPass());
  funcPassManager.addPass(IREE::LinalgExt::createDecomposeAttentionPass());
  funcPassManager.addPass(iree_compiler::createForallToForPass());

  {
    GenericVectorizationPassOptions options;
    options.useConfiguredVectorSizes = pipelineOpt.useConfiguredVectorSizes;
    options.enableVectorMasking = pipelineOpt.enableVectorMasking;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    if (clHexagonFailOnLargeVector) {
      funcPassManager.addPass(createLLVMCPUVerifyVectorSizeLegalityPass());
    }
  }

  addHexagonBufferizePasses(funcPassManager);

  {
    HexagonVectorLoweringPassOptions options;
    options.splitVectorTransfersTo = "linalg-copy";
    buildHexagonVectorLoweringPipeline(funcPassManager, options);
  }
}

// TODO: It might not be a bad idea to only keep the
// default pass pipeline. I should analyze the situations under which this is
// not the pipeline that is called, but I do not believe there are many since
// most things are based on the matmul operations anyway. This would massively
// simplify the pass logic and finally unify all this shitty stuff going on
// everywhere.

// This is the executed pipeline when no tiling is used. When lowering
// linalg.matmul operations into hexkl API calls, we manage tiling and data
// layout rearrangements through hexagon-mlir passes (and therefore implicitly
// through the VTCM). This results in no tiling whatsoever and therefore this is
// the pipeline that is used.
void addHexagonDefaultPassPipeline(OpPassManager &funcPassManager,
                                   const HexagonPipelineOptions &pipelineOpt) {

  addHexagonTileAndDistributePasses(funcPassManager, pipelineOpt);
  funcPassManager.addPass(createLLVMCPUTileLastOpAndFuseProducerConsumerPass(
      IREE::CPU::TilingLevel::VectorCommonParallelTiles));

  addHexagonBufferizePasses(funcPassManager);

  // Keep decompose here (post-bufferization): the pass expects memref-based
  // hexkl.matmul operands and will crash on tensor-form ops.
  if (isHexKLMatmulLoweringEnabled()) {
    funcPassManager.addPass(mlir::hexagon::createDecomposeHexKLMatmulPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }
}

void addHexagonVariantFinalizationPasses(OpPassManager &variantPassManager) {
  variantPassManager.addPass(createReconcileTranslationInfoPass());
  variantPassManager.addPass(createCSEPass());
  variantPassManager.addPass(createResolveWorkgroupCountHintsPass());
  variantPassManager.addPass(createIREECodegenLowerAffinePass());
  variantPassManager.addPass(IREE::Util::createDropCompilerHintsPass());
}

void addHexagonLowerToLLVMPasses(OpPassManager &modulePassManager) {
  FunctionLikeNest(modulePassManager)
      .addPass(createEraseHALDescriptorTypeFromMemRefPass);

  FunctionLikeNest(modulePassManager)
      // LinalgExt -> SCF
      .addPass(IREE::LinalgExt::createLinalgExtToLoopsPass)
      // Linalg -> SCF
      .addPass(createMemrefCopyToLinalgPass)
      .addPredicatedPass(clHexagonCheckLinalgVectorization,
                         createLLVMCPUEmitVectorizationRemarksPass)
      .addPass(createConvertLinalgToLoopsPass)
      .addPass(createConvertBf16ArithToF32Pass)
      .addPass([]() {
        return createConvertUnsupportedFloatToIntBuffersPass(
            ConvertUnsupportedFloatToIntBuffersPassOptions{
                /*includeBf16=*/true,
                /*includeF8E5M2=*/false,
                /*includeF8E4M3FN=*/false,
                /*includeF8E5M2FNUZ=*/false,
                /*includeF8E4M3FNUZ=*/false,
                /*includeF8E8M0FNU=*/false,
            });
      })
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass);

  // Handled tensor-type constants.
  modulePassManager.addPass(createIREEBufferizeConstantsPass());

  FunctionLikeNest(modulePassManager)
      .addPass(createFoldTensorExtractOpPass)
      // Handle complex operation conversion.
      .addPass(createConvertComplexToStandardPass)
      // Math dialect ops rewrites, approximations, casts.
      .addPass(createMathTransformPass)
      .addPass(createHoistStaticallyBoundAllocationsPass)
      // Use `arith.minf/maxf` instead of `arith.minimumf/maximumf`.
      .addPredicatedPass(clHexagonUseFastMinMaxOps,
                         createReplaceSlowMinMaxOpsPass);

  VectorTransferLoweringPassOptions transferLoweringOptions;

  FunctionLikeNest(modulePassManager)
      // All structural buffer manipulations must conclude before this point.

      // The subview folding doesn't like potentially-out-of-bounds
      // vector.transfer_read and vector.transfer_write, lower them to loads and
      // stores here.
      .addPass([&]() {
        return createVectorTransferLoweringPass(transferLoweringOptions);
      })
      .addPass(memref::createFoldMemRefAliasOpsPass)
      .addPass(createIREEExpandStridedMetadataPass)
      .addPass(createCleanupBufferAllocViewPass)
      // Checking stack allocation before converting to CF dialect is easier.
      .addPass([&]() {
        return createLLVMCPUCheckIRBeforeLLVMConversionPass(
            LLVMCPUCheckIRBeforeLLVMConversionPassOptions{
                clHexagonFailOnOutOfBoundsStackAllocation});
      })
      // SCF -> CF
      .addPass(createSCFToControlFlowPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      // (HAL, IREE, Linalg, CF) -> LLVM
      .addPass(memref::createFoldMemRefAliasOpsPass)
      .addPass(createIREECodegenAffineExpandIndexOpsPass)
      .addPass([&]() {
        arith::ArithExpandOpsPassOptions options;
        options.includeBf16 = true;
        options.includeF4E2M1 = true;
        options.includeF8E8M0 = true;
        return arith::createArithExpandOpsPass(options);
      })
      .addPass(createConvertUnsupportedFloatArithPass)
      .addPass(createEmulateNarrowTypePass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      .addPredicatedPass(clHexagonInstrumentMemoryAccesses,
                         createInstrumentMemoryAccessesPass);

  // TODO: hexagon-mlir defines lowering for math operations (like tan or tanh).
  // IREE also does this. I should try to check which one of the two is more
  // efficient. This can be done by putting a flag before the convertToLLVM pass
  // that runs hexagon-mlir optimizations first
  modulePassManager.addPass(
      createHexagonConvertToLLVMPass(clHexagonEnableReassociateFpReductions));

  if (isHexKLMatmulLoweringEnabled()) {
    // Lower Hexagon dialect ops with their dedicated converters first.
    modulePassManager.addPass(mlir::hexkl::createHexKLToLLVMPass());
    modulePassManager.addPass(mlir::hexagonmem::createHexagonMemToLLVMPass());
    modulePassManager.addPass(createCanonicalizerPass());
    modulePassManager.addPass(createCSEPass());
  }

  modulePassManager.addPass(createMarkHexagonNativeRuntimeLinksPass());

  modulePassManager.addPass(createReconcileUnrealizedCastsPass());

  // We rely on MLIR symbol visibility being correct after this point and need
  // to mirror the LLVM linkage that was assigned during conversion.
  modulePassManager.addPass(createLLVMCPUSynchronizeSymbolVisibilityPass());

  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());
  modulePassManager.addNestedPass<LLVM::LLVMFuncOp>(
      createAddFastMathFlagsPass());
}

} // namespace mlir::iree_compiler::cellar_hexagon::codegen
