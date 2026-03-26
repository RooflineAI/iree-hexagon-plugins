// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// For the time being, this file very closely mimics the equivalent passes.cpp
// file from LLVMCPU. It is therefore relevant, at least for now, to open a diff
// of these two files to see the hexagon specific differences in case you are
// curious.

// Note that there is currently a noticeable interleave of functionality between
// the hexagon plugin's passes and those from LLVMCPU.
// Right now, we are reusing the LLVMCPU lowering selection strategy
// (LLVMCPUSelectLoweringStrategy.cpp + KernelDispatch.cpp), but we have our
// custom lowerExecutableTargetPass to map the selection to the pipelines
// defined in this file. When looking into the IR, you will therefore see
// pipeline selections such as CPUDefault, but note that they are actually
// mapped to the pipelines from this file despite that.
// TODO: In the future, it might be wiser to completely separate these
// behaviors... Decide on this later.

#include "Passes.h"

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
#include "mlir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-hexagon-pass-pipelines"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace cellar::target::hexagon {

/// Command line options used purely for development purposes. Not to be relied
/// on in any way.

// Duplicate the LLVMCPU development flags with Hexagon-specific names to avoid
// cl::opt collisions with the upstream CPU backend while keeping the knobs
// available for future tweaking.
static llvm::cl::opt<bool> clHexagonFailOnOutOfBoundsStackAllocation(
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

static llvm::cl::opt<bool> clHexagonUseSoftmaxInterFusion(
    "iree-hexagon-use-decompose-softmax-fuse",
    llvm::cl::desc("Enables inter-pass fusion for the DecomposeSoftmax pass."),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clHexagonEnableVectorContractCustomKernels(
    "iree-hexagon-enable-vector-contract-custom-kernels",
    llvm::cl::desc("Enables vector contract custom kernels for "
                   "LLVMCPUMmt4dVectorLowering pass."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clHexagonTileDispatchUsingForall(
    "iree-hexagon-tile-dispatch-using-forall",
    llvm::cl::desc("Enable tile and distribute to workgroups using scf.forall"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clHexagonPatchFuncOps(
    "iree-hexagon-debug-patch-func-ops",
    llvm::cl::desc(
        "Perform the patches on func ops for debugging purpose. It should be "
        "used with `--iree-codegen-debug-patched-func-ops-file-name`."),
    llvm::cl::init(false), llvm::cl::Hidden);

static void
addHexagonTileAndDistributePasses(OpPassManager &funcPassManager,
                                  const HexagonPipelineOptions &pipelineOpt) {
  if (pipelineOpt.disableDistribution) {
    return;
  }
  if (clHexagonTileDispatchUsingForall) {
    funcPassManager.addPass(
        createTileAndDistributeToWorkgroupsUsingForallOpPass());
    funcPassManager.addPass(createBufferizeDispatchTensorLoadStorePass());
    funcPassManager.addPass(createCombineLayoutTransformationPass());
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

void addHexagonMultiTilingExpertPassPipeline(
    OpPassManager &funcPassManager,
    IREE::Codegen::LoweringConfigAttrInterface loweringConfig,
    const HexagonPipelineOptions &pipelineOpt) {
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

  addCPUBufferizePasses(funcPassManager);

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

  addCPUBufferizePasses(funcPassManager);

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

void addHexagonMmt4dTilingExpertPassPipeline(
    OpPassManager &funcPassManager, const HexagonPipelineOptions &pipelineOpt) {
  addHexagonTileAndDistributePasses(funcPassManager, pipelineOpt);

  funcPassManager.addPass(createLLVMCPUTileAndFuseProducerConsumerPass(
      IREE::CPU::TilingLevel::VectorCommonParallelTiles));

  // For Hexagon, we currently disable ukernels no matter what
  // funcPassManager.addPass(createCPUPrepareUkernelsPass());
  // funcPassManager.addPass(
  //     createCPULowerToUKernelsPass(clHexagonSkipIntermediateRoundings));
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

  addCPUBufferizePasses(funcPassManager);

  // Vector lowering of Mmt4d.
  funcPassManager.addPass(createLLVMCPUMmt4dVectorLoweringPass(
      LLVMCPUMmt4dVectorLoweringPassOptions{
          clHexagonEnableVectorContractCustomKernels}));

  // Generic vector lowering.
  HexagonVectorLoweringPassOptions options;
  options.splitVectorTransfersTo = "linalg-copy";
  buildHexagonVectorLoweringPipeline(funcPassManager, options);
}

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

  addCPUBufferizePasses(funcPassManager);

  {
    HexagonVectorLoweringPassOptions options;
    options.splitVectorTransfersTo = "linalg-copy";
    buildHexagonVectorLoweringPipeline(funcPassManager, options);
  }
}

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

  addCPUBufferizePasses(funcPassManager);

  {
    HexagonVectorLoweringPassOptions options;
    options.splitVectorTransfersTo = "linalg-copy";
    buildHexagonVectorLoweringPipeline(funcPassManager, options);
  }
}

void addHexagonDefaultPassPipeline(OpPassManager &funcPassManager,
                                   const HexagonPipelineOptions &pipelineOpt) {
  addHexagonTileAndDistributePasses(funcPassManager, pipelineOpt);
  funcPassManager.addPass(createLLVMCPUTileLastOpAndFuseProducerConsumerPass(
      IREE::CPU::TilingLevel::VectorCommonParallelTiles));
  addCPUBufferizePasses(funcPassManager);
}

static void addHexagonLowerToLLVMPasses(OpPassManager &modulePassManager,
                                        bool enableAArch64SME) {
  FunctionLikeNest(modulePassManager)
      .addPass(createEraseHALDescriptorTypeFromMemRefPass);

  // Lower `ukernel.*` ops to function calls
  modulePassManager.addPass(createLowerUKernelOpsToCallsPass());

  FunctionLikeNest(modulePassManager)
      // LinalgExt -> SCF
      .addPass(IREE::LinalgExt::createLinalgExtToLoopsPass)
      // Linalg -> SCF
      .addPass(createMemrefCopyToLinalgPass)
      .addPredicatedPass(clHexagonCheckLinalgVectorization,
                         createLLVMCPUEmitVectorizationRemarksPass)
      .addPass(createConvertLinalgToLoopsPass)
      .addPass(createConvertBf16ArithToF32Pass)
      .addPass(createConvertBf16ToUInt16BuffersPass)
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
  if (!enableAArch64SME) {
    // The ArmSME dialect has its own (more specific) lowerings for scalable
    // vectors that occur later in the pipeline, so only enable the general
    // lowerings if SME is not available.
    transferLoweringOptions.enableScalableLowerings = true;
  }

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
      .addPass(createEmulateNarrowTypePass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      .addPredicatedPass(clHexagonInstrumentMemoryAccesses,
                         createInstrumentMemoryAccessesPass);

  if (enableAArch64SME) {
    FunctionLikeNest(modulePassManager).addPass([&] {
      return createConvertArmSMEToLLVMPass();
    });
  }
  modulePassManager.addPass(
      createConvertToLLVMPass(clHexagonEnableReassociateFpReductions));
  modulePassManager.addPass(createReconcileUnrealizedCastsPass());

  // We rely on MLIR symbol visibility being correct after this point and need
  // to mirror the LLVM linkage that was assigned during conversion.
  modulePassManager.addPass(createLLVMCPUSynchronizeSymbolVisibilityPass());

  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());
  modulePassManager.addNestedPass<LLVM::LLVMFuncOp>(
      createAddFastMathFlagsPass());
}

static void buildHexagonCodegenConfigurationPassPipelineImpl(
    OpPassManager &modulePassManager) {
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    addCommonTargetExecutablePreprocessingPasses(
        funcPassManager, clHexagonUseSoftmaxInterFusion);
  }
  modulePassManager.addPass(createMaterializeUserConfigsPass());
  FunctionLikeNest(modulePassManager)
      .addPass(createMaterializeDeviceEncodingPass)
      .addPass(createCPUPropagateDataLayoutPass)
      .addPass(createRematerializeParallelOpsPass)
      // This pass is removed for hexagon
      // .addPass(createExpandF16OpToF32Pass)
      .addPass(createConvertAccGEMMToGEMMPass)
      .addPass(createEraseHALDescriptorTypeFromMemRefPass);

  modulePassManager.addPass(createLLVMCPUSelectLoweringStrategyPass());
  LLVM_DEBUG({
    llvm::dbgs() << "Hexagon codegen configuration pass pipeline:\n";
    modulePassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

void buildHexagonConfigurationPassPipeline(OpPassManager &variantPassManager) {
  variantPassManager.addPass(createSpecializeExportsPass());
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  buildHexagonCodegenConfigurationPassPipelineImpl(modulePassManager);
}

void buildHexagonTranslationPassPipeline(
    OpPassManager &variantPassManager,
    const HexagonPipelineOptions &pipelineOptions) {

  {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    modulePassManager.addPass(createLowerExecutableUsingTransformDialectPass());
    FunctionLikeNest(modulePassManager)
        // This is a custom pass that maps LLVMCPU lowering pipeline selections
        // to Hexagon's implementation of those pipelines.
        .addPass(createHexagonLowerExecutableTargetPass)
        .addPass(createVerifyWorkgroupDistributionPass);
    if (clHexagonPatchFuncOps) {
      modulePassManager.addPass(createPatchFuncOpsPass());
    }
  }

  variantPassManager.addPass(createReconcileTranslationInfoPass());
  variantPassManager.addPass(createCSEPass());
  variantPassManager.addPass(createResolveWorkgroupCountHintsPass());
  variantPassManager.addPass(createIREECodegenLowerAffinePass());
  variantPassManager.addPass(IREE::Util::createDropCompilerHintsPass());

  // Run conversion to LLVM at `ModuleOp` granularity.
  {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    addHexagonLowerToLLVMPasses(modulePassManager, false);
  }
  LLVM_DEBUG({
    llvm::dbgs() << "Hexagon codegen pass pipeline:\n";
    variantPassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

// NOTE: this runs on the top-level program module containing all
// hal.executable ops.
void buildHexagonLinkingPassPipeline(OpPassManager &modulePassManager,
                                     std::optional<std::string> target) {
  // Link together executables. This may produce some IR duplication.
  LLVMCPULinkExecutablesPassOptions linkOptions;
  linkOptions.target = target.value_or("");
  modulePassManager.addPass(createLLVMCPULinkExecutablesPass(linkOptions));

  // Cleanup IR duplication.
  modulePassManager.addNestedPass<IREE::HAL::ExecutableOp>(
      mlir::createCanonicalizerPass());

  // Assign final executable constant and import ordinals.
  auto &variantPassManager = modulePassManager.nest<IREE::HAL::ExecutableOp>()
                                 .nest<IREE::HAL::ExecutableVariantOp>();
  variantPassManager.addPass(createLLVMCPUAssignConstantOrdinalsPass());
  variantPassManager.addPass(createLLVMCPUAssignImportOrdinalsPass());
}

//===---------------------------------------------------------------------===//
// Register Hexagon Passes
//===---------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "cellar-hexagon/CodeGen/Passes.h.inc" // IWYU pragma: keep
} // namespace

void registerHexagonPasses() {
  registerCodegenLLVMCPUPasses(); // In order to register LLVMCPU passes
  registerPasses();
}

void registerHexagonCodeGenPasses() {
  registerHexagonPasses();

  static PassPipelineRegistration<> configurationPipeline(
      "iree-hexagon-configuration-pipeline",
      "Runs the Hexagon HAL configuration pipeline.",
      [](OpPassManager &passManager) {
        buildHexagonConfigurationPassPipeline(passManager);
      });

  static PassPipelineRegistration<> translationPipeline(
      "iree-hexagon-translation-pipeline",
      "Runs the Hexagon HAL translation pipeline.",
      [](OpPassManager &passManager) {
        buildHexagonTranslationPassPipeline(passManager);
      });

  static PassPipelineRegistration<> linkingPipeline(
      "iree-hexagon-linking-pipeline", "Runs the Hexagon HAL linking pipeline.",
      [](OpPassManager &passManager) {
        buildHexagonLinkingPassPipeline(passManager);
      });
}
} // namespace cellar::target::hexagon
