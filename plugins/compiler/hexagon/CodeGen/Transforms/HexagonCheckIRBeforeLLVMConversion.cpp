// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Derived from Codegen/LLVMCPU/LLVMCPUCheckIRBeforeLLVMConversion.cpp at IREE
// revision a45adeaa6115e446c898e6eb21fb6edc0e65ddc4, without scalable-vector
// (vscale) bounds.

#include "hexagon/CodeGen/Passes.h"
#include "hexagon/CodeGen/TargetConfig.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONCHECKIRBEFORELLVMCONVERSIONPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {
struct HexagonCheckIRBeforeLLVMConversionPass
    : impl::HexagonCheckIRBeforeLLVMConversionPassBase<
          HexagonCheckIRBeforeLLVMConversionPass> {
  using impl::HexagonCheckIRBeforeLLVMConversionPassBase<
      HexagonCheckIRBeforeLLVMConversionPass>::
      HexagonCheckIRBeforeLLVMConversionPassBase;
  void runOnOperation() override;
};
} // namespace

/// Returns success if the cumulative stack allocation size is less than the
/// target-configured limit (or the 32KB fallback).
static LogicalResult
checkStackAllocationSize(mlir::FunctionOpInterface funcOp) {
  if (funcOp.getFunctionBody().empty()) {
    return success();
  }

  // In rare cases where the attribute is not present in the module, a value of
  // 32KB will be taken.
  unsigned maxAllocationSizeInBytes = 32 * 1024;
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (targetAttr) {
    std::optional<int64_t> nativeAllocationSize =
        getConfigMaxStackAllocationSize(targetAttr.getConfiguration());
    if (nativeAllocationSize) {
      maxAllocationSizeInBytes = nativeAllocationSize.value();
    }
  }

  SmallVector<memref::AllocaOp> allocaOps;
  funcOp.walk(
      [&](memref::AllocaOp allocaOp) { allocaOps.push_back(allocaOp); });
  if (allocaOps.empty()) {
    return success();
  }

  int64_t cumSize = 0;
  for (auto allocaOp : allocaOps) {
    if (allocaOp->getBlock() != &funcOp.getFunctionBody().front()) {
      return allocaOp->emitOpError(
          "all stack allocations need to be hoisted to the entry block of the "
          "function");
    }
    int64_t allocaSize = 1;
    auto allocaType = cast<ShapedType>(allocaOp.getType());
    for (auto dimSize : allocaType.getShape()) {
      if (ShapedType::isDynamic(dimSize)) {
        continue;
      }
      allocaSize *= dimSize;
    }
    for (auto operand : allocaOp.getDynamicSizes()) {
      FailureOr<int64_t> ub = ValueBoundsConstraintSet::computeConstantBound(
          presburger::BoundType::UB, operand,
          [](Value, std::optional<int64_t>, ValueBoundsConstraintSet &) {
            return false;
          },
          ValueBoundsOptions{/*closedUB=*/true});
      if (succeeded(ub)) {
        allocaSize *= *ub;
        continue;
      }
      return allocaOp.emitOpError("expected no unbounded stack allocations");
    }
    allocaSize *= IREE::Util::getTypeBitWidth(allocaType.getElementType());
    if (allocaOp.getAlignment()) {
      int64_t alignmentInBits = *allocaOp.getAlignment() * 8;
      allocaSize =
          (llvm::divideCeil(allocaSize, alignmentInBits) * alignmentInBits);
    }
    cumSize += allocaSize / 8;
  }
  if (cumSize > maxAllocationSizeInBytes) {
    return funcOp.emitOpError("exceeded stack allocation limit of ")
           << maxAllocationSizeInBytes << " bytes for function. Got " << cumSize
           << " bytes";
  }
  return success();
}

void HexagonCheckIRBeforeLLVMConversionPass::runOnOperation() {
  if (!failOnOutOfBounds) {
    return;
  }

  mlir::FunctionOpInterface funcOp = getOperation();
  if (failed(checkStackAllocationSize(funcOp))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler::hexagon::codegen
