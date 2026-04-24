// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file owns the experimental nested-module lowering path built primarily
// from hexagon-mlir passes.

#include "cellar-hexagon/CodeGen/Pipelines/HexagonMlirPipeline.h"
#include "cellar-hexagon/CodeGen/Conversion/HexagonConvertToLLVM.h"
#include "cellar-hexagon/CodeGen/Pipelines/Bufferization.h"
#include "cellar-hexagon/CodeGen/Pipelines/IreeLoweringPipelines.h"
#include "cellar-hexagon/CodeGen/Pipelines/TranslationPipeline.h"

#include "hexagon/Conversion/DMAToLLVM/Passes.h"
#include "hexagon/Conversion/HexKLToLLVM/HexKLToLLVM.h"
#include "hexagon/Conversion/HexagonMemToLLVM/HexagonMemToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Quant/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "iree-hexagon-pass-pipelines"

namespace mlir::iree_compiler::cellar_hexagon::codegen {
namespace Hexagon = mlir::hexagon;

enum class HexagonTilingPipeline {
  HexagonMlir = 0,
  IREE = 1,
};

// TOOD: Also experimental flag, think about this later
llvm::cl::opt<HexagonTilingPipeline> clHexagonTilingPipeline(
    "iree-hexagon-tiling-pipeline",
    llvm::cl::desc("Select which tiling passes are used in the Hexagon "
                   "lowering pipeline."),
    llvm::cl::values(clEnumValN(HexagonTilingPipeline::HexagonMlir,
                                "hexagon-mlir",
                                "Use Hexagon-mlir tiling passes."),
                     clEnumValN(HexagonTilingPipeline::IREE, "iree",
                                "Use IREE tiling/distribution passes.")),
    llvm::cl::init(HexagonTilingPipeline::HexagonMlir));

void addHexagonMlirLowerToLLVMPasses(OpPassManager &variantPassManager) {
  auto &pm = variantPassManager.nest<ModuleOp>();
  const bool enableCollapseAddressSpace = true;
  const bool enableHexagonRoutines = false;

  // Structural split of conversion:
  // - phase 1 performs HAL ABI + func/vector/index/cf conversion,
  //   but intentionally keeps hal.interface.binding.subspan + memref
  //   finalization out;
  // - phase 2 lowers the deferred pieces after address-space normalization.
  //
  // LLVMCPU does not need this split because it does not interleave Hexagon's
  // address-space collapsing with a custom HAL conversion pass. In this
  // hybrid Hexagon pipeline we must guarantee:
  //   phase1 -> collapse-address-space -> phase2
  // so dealloc lowering emits @free with ptr in the default address space and
  // binding subspans stay memref-typed while DMA/memref users are still
  // alive.
  pm.addPass(createHexagonConvertToLLVMPassPhase1(
      /*reassociateFpReductions=*/false));

  // These passes are self contained within functions are have no standard
  // lowering. Therefore, they can run in tadem with iree's standard lowering.
  pm.addPass(hexagonmem::createHexagonMemToLLVMPass());
  pm.addPass(Hexagon::createDMAToLLVMPass());
  pm.addPass(hexkl::createHexKLToLLVMPass());
  // Hexagon DMA/HexKL/HexagonMem runtime symbols are always kept as native
  // unresolved externs and resolved by the DSP loader.
  pm.addPass(createMarkHexagonNativeRuntimeLinksPass());

  // Must run after function lowering and before memref finalization.
  // The collapse pass rewrites ptr<addrspace> in descriptors/calls to default
  // address space so finalize-memref-to-llvm can lower deallocs without
  // producing @free(ptr<nonzero>).
  if (enableCollapseAddressSpace) {
    pm.addPass(Hexagon::createCollapseAddressSpacePass());
    pm.addPass(createReconcileUnrealizedCastsPass());
  }

  // Complete conversion after address-space normalization.
  pm.addPass(createHexagonConvertToLLVMPassPhase2(
      /*reassociateFpReductions=*/false));

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  if (enableHexagonRoutines)
    pm.addPass(Hexagon::createHexagonLLVMEnableHexagonRoutinesPass());
}

// TODO: Remember to come back to this, I am only testing stuff here for now.
// This is just dirty copy pasted code from hexagon-mlir
void buildHexagonMlirTranslationRoute(
    OpPassManager &variantPassManager,
    const HexagonPipelineOptions &pipelineOpt) {
  const bool useIREETilingPipeline =
      clHexagonTilingPipeline == HexagonTilingPipeline::IREE;
  // Note that hexagon-mlir passes are working on module ops, while iree's are
  // working on hal ops
  auto puntBuffer = true;
  auto enableHexKL = isHexKLMatmulLoweringEnabled();
  auto enableConvTiling = false;
  auto returnValueOptimization = false;
  auto enableSCFThreading = false;
  auto enableMultiThreading = false;
  auto enableVTCMTiling = isHexagonVTCMTilingEnabled();
  auto fusion = true;
  auto enableHexagonmemCopyToDMA = true;
  auto enableConvertToHexagonmem = true;
  auto enableDoubleBuffering = false;
  auto enableBufferization = true;
  auto addFastMath = true;
  auto enableVectorization = true;
  auto enableSlicing = false;
  auto useInterchangeVector = false;
  auto enableSplitReduction = false;
  auto expandBoolVec = true;
  auto slicingFactor = 10;

  // auto setIndexBitwidth = [&](auto passOption) {
  //   passOption.indexBitwidth = 32;
  //   return passOption;
  // };

  auto setuseInterchangeVector = [&](auto passOption) {
    passOption.useInterchangeVector = useInterchangeVector;
    return passOption;
  };

  auto setsplitTilingRange = [&](auto passOption) {
    passOption.splitTilingRange = true;
    return passOption;
  };

  auto setenableSplitReduction = [&](auto passOption) {
    passOption.enableSplitReduction = enableSplitReduction;
    return passOption;
  };

  auto setConvTiling = [&](auto passOption) {
    passOption.convTilingFactor = 32;
    passOption.convTileHeightDim = true;
    passOption.convTileOCDim = true;
    return passOption;
  };

  auto setOpSlicingFactor = [&](auto passOption) {
    passOption.slicingFactor = slicingFactor;
    return passOption;
  };

  auto setVTCMTiling = [&](auto passOption) {
    passOption.tileSizes = "";
    return passOption;
  };

  {
    auto &pm = variantPassManager.nest<ModuleOp>();
    pm.addNestedPass<func::FuncOp>(Hexagon::createReduceContractionRankPass());
    pm.addPass(createLinalgFoldUnitExtentDimsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (puntBuffer)
      pm.addNestedPass<func::FuncOp>(Hexagon::createHexagonPuntBufferPass());
    pm.addPass(createCanonicalizerPass()); // erase unstrung allocs

    // Quantization related passes in this block
    // Lower quant.qcast and quant.dcast ops to arith dialect
    pm.addNestedPass<func::FuncOp>(quant::createLowerQuantOps());
    // Convert arith ops to linalg elementwise ops
    pm.addPass(createConvertElementwiseToLinalgPass());
    // Remove quant.scast ops
    pm.addPass(createCSEPass());

    if (useIREETilingPipeline) {
      // Run IREE's dispatch tiling while lowering configs are still available
      // on the linalg ops. This is structurally different from the later
      // Hexagon tiler: TileAndDistributeToWorkgroupsUsingForallOpPass rewrites
      // the dispatch around workgroup-sized destination tiles and fuses
      // producer work into those tiles, which sinks many full-size
      // intermediate tensor.empty values into tile-local empties
      // (e.g. tensor<1x64x128xf32>) that later bufferize to much smaller
      // temporaries. HexagonTilingPass, in contrast, tiles each linalg op in
      // isolation for vectorization and preserves the original full tensor
      // result as loop-carried state updated via extract_slice/insert_slice, so
      // the logical intermediate object does not shrink even though each loop
      // iteration only touches a tile.
      addHexagonTileAndDistributePasses(pm.nest<func::FuncOp>(), pipelineOpt);
      // ReconcileTranslationInfo cannot resolve result-valued scf.forall ops.
      // Lower the distributed forall form before variant-level reconciliation.
      pm.addNestedPass<func::FuncOp>(createForallToForPass());
    }
  }

  {
    auto &pm = variantPassManager.nest<ModuleOp>();

    if (enableHexKL)
      pm.addNestedPass<func::FuncOp>(Hexagon::createMatmulToHexKLPass());

    if (enableConvTiling) {
      pm.addNestedPass<func::FuncOp>(Hexagon::createConvTilingPass(
          setConvTiling(Hexagon::ConvTilingOptions{})));
      pm.addPass(createCanonicalizerPass());
    }

    pm.addNestedPass<func::FuncOp>(Hexagon::createConvertLayoutPass());
    pm.addNestedPass<func::FuncOp>(Hexagon::createScheduleMatmulForHVXPass());
    pm.addNestedPass<func::FuncOp>(Hexagon::createLinalgGeneralizePass());

    if (returnValueOptimization)
      pm.addNestedPass<func::FuncOp>(Hexagon::createHexagonRVOPass());
    pm.addPass(createCanonicalizerPass()); // erase unstrung re-interprets
    pm.addPass(createCSEPass());

    if (enableSCFThreading) {
      assert(!enableMultiThreading && !enableVTCMTiling &&
             "currently scf-threading can be enabled only if"
             " linalg multi-threading and vtcm tiling are off");
      pm.addNestedPass<func::FuncOp>(Hexagon::createFormSCFThreadsPass());
    }

    pm.addPass(Hexagon::createEraseUnusedLinalgOperands());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (fusion)
      pm.addNestedPass<func::FuncOp>(Hexagon::createHexagonFusionPass());
    pm.addNestedPass<func::FuncOp>(Hexagon::createDecomposeTensorConcatPass());
    // Hexagon slicing reduce the number of outer loops in loops and unroll
    // those
    if (enableSlicing)
      pm.addPass(Hexagon::createHexagonSlicingPass(
          setOpSlicingFactor(Hexagon::HexagonSlicingOptions{})));

    if (enableVTCMTiling && !useIREETilingPipeline) {
      // TODO: This is a test, remove if unnecessary
      pm.addNestedPass<func::FuncOp>(
          createEraseHALDescriptorTypeFromMemRefPass());
      pm.addNestedPass<func::FuncOp>(Hexagon::createVTCMTilingPass(
          setVTCMTiling(Hexagon::VTCMTilingOptions{})));
      pm.addPass(createCanonicalizerPass());
      // Remove output staging alloc_tensor(copy, memory_space=0) introduced by
      // VTCM tiling before IREE comprehensive bufferization.
      // VTCM tiling introduces copies from tensors allocated in memory_space=0
      // (DDR) to #hal.descriptor_type<storage_buffer> which is also DDR. The
      // root issue is that hexagon-mlir memory-related passes conflict with
      // iree's types. Removing HAL descriptors is not enough here, since we are
      // actually dealing with a iree_tensor_ext.dispatch.tensor.store operation
      pm.addNestedPass<func::FuncOp>(createFoldDispatchOutputStagingPass());
    }

    if (enableMultiThreading) {
      pm.addNestedPass<func::FuncOp>(Hexagon::createFormVirtualThreadsPass(
          Hexagon::FormVirtualThreadsOptions{}));
    }

    pm.addPass(Hexagon::removeMLProgramPass());
    pm.addPass(createLinalgFoldUnitExtentDimsPass());
    if (enableVectorization && !useIREETilingPipeline) {
      pm.addPass(Hexagon::createHexagonTilingPass(
          setsplitTilingRange(setuseInterchangeVector(
              setenableSplitReduction(Hexagon::HexagonTilingOptions{})))));
    }

    pm.addPass(createLinalgFoldUnitExtentDimsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(Hexagon::createSmallExponentToMultiplyPass(
        Hexagon::SmallExponentToMultiplyOptions{}));
    // ===== STEP 1: HOIST SCALAR OPS =====
    // Run before vectorization to expose scalar invariants
    pm.addNestedPass<func::FuncOp>(Hexagon::createHoistScalarOpsPass());
    pm.addPass(Hexagon::createEraseUnusedLinalgOperands());
    pm.addPass(createCSEPass());

    // ===== STEP 1.5: LOOP INVARIANT CODE MOTION =====
    // Move hoisted scalars further up the loop nest
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // Run LICM again after canonicalization to catch newly exposed
    // opportunities
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());

    // ===== STEP 2: VECTORIZATION =====
    // Vectorizer now sees cleaner IR with hoisted scalars
    if (enableVectorization) {
      pm.addPass(Hexagon::createHexagonVectorizationPass());
    }
    pm.addPass(Hexagon::createRewriteUBPoisonToZeroPass());
    pm.addPass(Hexagon::createHexagonVectorLoweringPass());
    pm.addPass(createCanonicalizerPass());

    if (addFastMath) {
      pm.addPass(Hexagon::createHexagonAddFastMathPass());
      pm.addNestedPass<func::FuncOp>(Hexagon::createFoldMulFByZeroPass());
      pm.addPass(createCanonicalizerPass());
    }
    pm.addPass(memref::createResolveShapedTypeResultDimsPass());

    if (enableBufferization) {
      // We cannot reuse hexagon-mlir (actually normal mlir) bufferization
      // since iree handles buffers differently.
      // pm.addPass(bufferization::createEmptyTensorEliminationPass());
      // mlir::bufferization::OneShotBufferizePassOptions passOpts;
      // passOpts.bufferizeFunctionBoundaries = true;
      // passOpts.allowReturnAllocsFromLoops = true;
      // pm.addPass(bufferization::createOneShotBufferizePass(passOpts));

      // pm.addPass(createCSEPass());
      // pm.addPass(createCanonicalizerPass());

      // This is wrapping IREE's custom wrapper around bufferization for
      // Hexagon. Note that it is only meant to run on functions.
      addHexagonBufferizePasses(pm.nest<func::FuncOp>());

      if (enableDoubleBuffering) {
        pm.addNestedPass<func::FuncOp>(
            Hexagon::createHexagonDoubleBufferGenericS1Pass());
      }

      pm.addNestedPass<func::FuncOp>(
          bufferization::createBufferLoopHoistingPass());

      pm.addNestedPass<func::FuncOp>(Hexagon::createCopyCanonicalizationPass());
      pm.addPass(createCanonicalizerPass());

      // The functionality of this pass overlaps with what
      // addHexagonBufferizePass is doing. Both of them want to manage
      // deallocations. It looks like hexagon-mlir is dealing with buffer here
      // and all of this might overlap with iree's comprehensive bufferize,
      // since a decomposition of the passes is what is needed (double buffering
      // happens before and after bufferization in two phases, while
      // comprehensveBufferize skips directly to memref.dealloc. Maybe I can
      // introduce buffer operations instead? Check!)
      bufferization::buildBufferDeallocationPipeline(
          pm, bufferization::BufferDeallocationPipelineOptions{});

      pm.addPass(createCSEPass());
      if (enableDoubleBuffering) {
        pm.addNestedPass<func::FuncOp>(
            Hexagon::createHexagonDoubleBufferGenericS2Pass());
      }

      pm.addNestedPass<func::FuncOp>(
          Hexagon::createConvertZeroSizeMemrefPass());
      pm.addPass(createConvertBufferizationToMemRefPass());
    }

    // Hexagon-mlir's convertToHexagonmemPass expects integer memory spaces;
    // strip HAL descriptor memory-space attrs (strings) before
    // ConvertToHexagonmem. I am assuming that they are not needed anymore. Note
    // that this is not unique to hexagon, and the LLVMCPU normal backend also
    // does this before lowering to llvm through standard mlir passes. The only
    // open question is whether this removal is too early right now. Since this
    // pipeline is reusing hexagon-mlir as is, this should not be problematic
    // though.
    pm.addNestedPass<func::FuncOp>(
        createEraseHALDescriptorTypeFromMemRefPass());

    if (enableConvertToHexagonmem)
      pm.addNestedPass<func::FuncOp>(Hexagon::createConvertToHexagonmemPass());

    // Decompose hexkl.matmul to micro ops
    if (enableHexKL)
      pm.addNestedPass<func::FuncOp>(Hexagon::createDecomposeHexKLMatmulPass());

    // Lower linalg ops with library_call attribute set to custom fns.
    pm.addPass(Hexagon::createHexagonReplaceWithLibraryCallsPass());
    if (enableHexagonmemCopyToDMA)
      pm.addNestedPass<func::FuncOp>(Hexagon::createHexmemCpyToDMAPass());
    // HexmemCpyToDMA materializes short-lived DMA tag buffers as
    // `memref.alloc : memref<1xi32>` + `memref.dealloc`. These tags only carry
    // the dma token between `dma_start` and `dma_wait`, so keeping them on heap
    // introduces unnecessary malloc/free imports in the LLVM stage. Promote
    // just those DMA tag allocations to stack and drop their deallocs.
    pm.addNestedPass<func::FuncOp>(createPromoteDMATagAllocToStackPass());
    // Hoist the newly introduced allocas out of loops
    pm.addNestedPass<func::FuncOp>(createHoistStaticallyBoundAllocationsPass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createConvertLinalgToLoopsPass());

    // pm.addNestedPass<func::FuncOp>(createFormAsyncThreadsPass());
    // pm.addPass(createAsyncFuncToAsyncRuntimePass());
    // pm.addPass(createAsyncToAsyncRuntimePass());

    pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());

    // This is not a pass from hexagon-mlir, but it does help quite a bit with
    // debugging
    pm.addNestedPass<func::FuncOp>(createLLVMCPUCheckIRBeforeLLVMConversionPass(
        LLVMCPUCheckIRBeforeLLVMConversionPassOptions{
            isHexagonFailOnOutOfBoundsStackAllocationEnabled()}));

    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createSCFToControlFlowPass());

    pm.addPass(createConvertMathToLLVMPass());
    pm.addNestedPass<func::FuncOp>(Hexagon::createExpandMathOpsPass());

    if (expandBoolVec)
      pm.addNestedPass<func::FuncOp>(Hexagon::createExpandBoolVecPass());

    pm.addPass(Hexagon::createFastInversePass());

    // Here is where the hexagon-mlir usually calls standard mlir lowerings to
    // LLVM. These are currently removed in favour of iree ones that manage ABI
    // and function signatures adapted to the runtime
    // pm.addPass(createConvertVectorToLLVMPass());
    // pm.addPass(createConvertIndexToLLVMPass(
    //     setIndexBitwidth(ConvertIndexToLLVMPassOptions{})));

    // pm.addPass(createConvertAsyncToLLVMPass());
  }
}

} // namespace mlir::iree_compiler::cellar_hexagon::codegen
