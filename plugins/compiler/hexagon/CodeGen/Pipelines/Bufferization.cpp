// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Pipelines/Bufferization.h"

#include "hexagon/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/Passes.h"

#include <optional>

// This file is based on
// third-party/iree/compiler/src/iree/compiler/Codegen/Common/CPU/Passes.cpp,
// open it to the side for comparison if needed.

namespace mlir::iree_compiler::hexagon::codegen {
namespace {

mlir::LogicalResult memrefCopyFn(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value from, mlir::Value to) {
  // Hexagon-mlir relies on memref copies to manage DMA operations instead of
  // linalg.copy, which is what was originally used by the LLVMCPU pipeline.
  mlir::memref::CopyOp::create(builder, loc, from, to);
  return mlir::success();
}

void addHexagonBufferizePassesCommon(mlir::OpPassManager &funcPassManager) {
  // In order to take full advantage of hexagon's VTCM, we need to be able to
  // return allocations from within tiled loops and therefore it is necessary to
  // pass the the allowReturnAllocsFromLoops option here
  mlir::iree_compiler::addIREEComprehensiveBufferizePasses(
      funcPassManager, /*allocationFn*/ {}, memrefCopyFn,
      /*allowReturnAllocsFromLoops=*/true);
}

} // namespace

void addHexagonBufferizePasses(mlir::OpPassManager &funcPassManager) {
  addHexagonBufferizePassesCommon(funcPassManager);

  // Convert dynamic memref.alloc ops to static ones before
  // ConvertToHexagonmemPass, which only accepts fully static VTCM memrefs.
  // Remainder/partial tiles produce dynamic-sized allocs after bufferization;
  // this pass uses IntegerRangeAnalysis to find upper bounds and replaces each
  // with a static alloc + memref.subview.
  funcPassManager.addPass(createPadDynamicAllocPass());
  funcPassManager.addPass(mlir::bufferization::createBufferLoopHoistingPass());
  funcPassManager.addPass(
      mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
  funcPassManager.addPass(
      mlir::bufferization::createBufferDeallocationSimplificationPass());
  funcPassManager.addPass(mlir::bufferization::createLowerDeallocationsPass());
  funcPassManager.addPass(mlir::createCanonicalizerPass());
  funcPassManager.addPass(createEraseHALDescriptorTypeFromMemRefPass());
  funcPassManager.addPass(hexagon::createConvertToHexagonmemPass());
}

void addHexagonBufferizePassesForHexagonMlir(
    mlir::OpPassManager &funcPassManager) {
  addHexagonBufferizePassesCommon(funcPassManager);
}

} // namespace mlir::iree_compiler::hexagon::codegen
