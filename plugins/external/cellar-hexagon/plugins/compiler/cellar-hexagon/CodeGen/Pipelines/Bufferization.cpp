// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/CodeGen/Pipelines/Bufferization.h"

#include <limits>
#include <optional>

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

// This file is based on
// third-party/iree/compiler/src/iree/compiler/Codegen/Common/CPU/Passes.cpp,
// open it to the side for comparison if needed.

namespace mlir::iree_compiler::cellar_hexagon::codegen {
namespace {

// This is a completely arbitrary value for now.
static constexpr int64_t kMaxStackScratchBytes = 8 * 1024;

static std::optional<int64_t>
getStaticAllocationSizeInBytes(mlir::MemRefType memRefType) {
  if (!memRefType.hasStaticShape()) {
    return std::nullopt;
  }
  int64_t elementBytes =
      mlir::iree_compiler::IREE::Util::getRoundedElementByteWidth(
          memRefType.getElementType());
  int64_t numElements = memRefType.getNumElements();
  if (numElements < 0 || elementBytes <= 0 ||
      numElements > std::numeric_limits<int64_t>::max() / elementBytes) {
    return std::nullopt;
  }
  return numElements * elementBytes;
}

static mlir::FailureOr<mlir::Value>
hexagonAllocationFn(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::MemRefType memRefType, mlir::ValueRange dynamicSizes,
                    unsigned alignment) {
  // Hexagon DSP worker threads only expose a small stack in practice, so large
  // statically shaped temporaries are routed to heap-backed `memref.alloc`
  // instead of `memref.alloca`. This is a pragmatic fallback for currently
  // unfused kernels such as attention/softmax, where whole-tensor scratch
  // buffers would otherwise trip the stack checker or overflow at runtime.

  // Do note however that I should try to get rid of those intermediate buffers
  // at some point. This is more a patch for this problem.
  bool preferHeapAllocation = false;
  std::optional<int64_t> staticSize =
      getStaticAllocationSizeInBytes(memRefType);
  if (staticSize) {
    preferHeapAllocation = *staticSize >= kMaxStackScratchBytes;
  }

  // TODO: This has to be more heavily tested:
  // This file is currently being called by both pipelines being tested, which
  // results in questionable decisions. Do we still want to use this policy of
  // allocation for using the VTCM? Do we want IREE and Hexagon MLIR to deal
  // with it differently?

  // IREE usually does hoisting here (this pass also introduces deallocation
  // operations when using allocs instead of allocas). Nevertheless,
  // hexagon-mlir uses a custom pass to do it. For simplicity to manage the
  // double buffering passes, do the hoisting when hexagon-mlir expects.

  // FIXME: bring back this hoisting, ROO-1478
  // auto funcOp =
  // builder.getInsertionPoint()->getParentOfType<mlir::FunctionOpInterface>();
  // if (funcOp) {
  //   std::optional<mlir::Value> hoistedAllocation =
  //       preferHeapAllocation
  //           ? mlir::iree_compiler::hoistOneStaticallyBoundAllocation<
  //                 mlir::memref::AllocOp>(funcOp, builder, loc, memRefType,
  //                                        dynamicSizes, alignment)
  //           : mlir::iree_compiler::hoistOneStaticallyBoundAllocation<
  //                 mlir::memref::AllocaOp>(funcOp, builder, loc, memRefType,
  //                                         dynamicSizes, alignment);
  //   if (hoistedAllocation) {
  //     return *hoistedAllocation;
  //   }
  // }
  if (preferHeapAllocation) {
    if (!memRefType.getMemorySpace()) {
      auto diag = mlir::emitWarning(loc)
                  << "created large heap-allocated buffer of type "
                  << memRefType;
      if (staticSize) {
        diag << " (" << *staticSize << " bytes)";
      }
      diag << " because it exceeds the Hexagon stack scratch threshold of "
           << kMaxStackScratchBytes << " bytes";
    }
    return mlir::memref::AllocOp::create(builder, loc, memRefType, dynamicSizes,
                                         builder.getI64IntegerAttr(alignment))
        .getResult();
  }
  return mlir::memref::AllocaOp::create(builder, loc, memRefType, dynamicSizes,
                                        builder.getI64IntegerAttr(alignment))
      .getResult();
}

static mlir::LogicalResult hexagonCopyFn(mlir::OpBuilder &builder,
                                         mlir::Location loc, mlir::Value from,
                                         mlir::Value to) {
  // Hexagon-mlir relies on memref copies to manage DMA operations instead of
  // linalg.copy, which is what was originally used by the LLVMCPU pipeline.
  mlir::memref::CopyOp::create(builder, loc, from, to);
  return mlir::success();
}

} // namespace

void addHexagonBufferizePasses(mlir::OpPassManager &funcPassManager) {
  mlir::iree_compiler::BufferizationOptions::AllocationFn allocationFn =
      hexagonAllocationFn;
  mlir::iree_compiler::BufferizationOptions::MemCpyFn memcpyFn = hexagonCopyFn;
  // In order to take full advantage of hexagon's VTCM, we need to be able to
  // return allocations from within tiled loops and therefore it is necessary to
  // pass the the allowReturnAllocsFromLoops option here
  mlir::iree_compiler::addIREEComprehensiveBufferizePasses(
      funcPassManager, allocationFn, memcpyFn,
      /*allowReturnAllocsFromLoops=*/true);
}

} // namespace mlir::iree_compiler::cellar_hexagon::codegen
