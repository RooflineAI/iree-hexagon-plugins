// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Measures the hardware resources of each dispatch on the final IR before LLVM
// conversion and records them on its hal.executable.export.

#include "hexagon/CodeGen/IR/HexagonAttrs.h"
#include "hexagon/CodeGen/IR/HexagonDialect.h"
#include "hexagon/CodeGen/IR/HexagonOps.h"
#include "hexagon/CodeGen/Passes.h"
#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/dsp/rt/vtcm_allocation.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONANNOTATEDISPATCHRESOURCESPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

using IREE::Hexagon::DispatchResourcesAttr;

/// The runtime's VTCM allocation sizes (hexagon/dsp/rt/vtcm_allocation.h).
constexpr int64_t kVTCMPoolMinimumBytes = HEXAGON_DSP_VTCM_MIN_ALLOCATION_BYTES;
constexpr int64_t kVTCMLargeAlignment = HEXAGON_DSP_VTCM_LARGE_ALIGNMENT;

/// Bytes of a statically shaped memref, or std::nullopt if it is dynamic.
std::optional<int64_t> getStaticSizeInBytes(MemRefType type,
                                            const DataLayout &dataLayout) {
  if (!type.hasStaticShape())
    return std::nullopt;
  return type.getNumElements() *
         static_cast<int64_t>(
             dataLayout.getTypeSize(type.getElementType()).getFixedValue());
}

/// VTCM bytes `memref.alloc` takes. It lowers to `malloc`, which the runtime
/// serves from the VTCM pool (plugins/runtime/hexagon/dsp/rt/malloc_free.c).
/// MemRefToLLVM pads the request by the alignment, which is the element size
/// for non-scalar elements when no alignment is given.
std::optional<int64_t> getAllocVTCMBytes(memref::AllocOp op,
                                         const DataLayout &dataLayout) {
  MemRefType type = op.getType();
  std::optional<int64_t> bytes = getStaticSizeInBytes(type, dataLayout);
  if (!bytes)
    return std::nullopt;
  if (std::optional<uint64_t> alignment = op.getAlignment())
    *bytes += *alignment;
  else if (!type.getElementType().isSignlessIntOrIndexOrFloat())
    *bytes += static_cast<int64_t>(
        dataLayout.getTypeSize(type.getElementType()).getFixedValue());
  return std::max(*bytes, kVTCMPoolMinimumBytes);
}

/// VTCM bytes `hexagonmem.alloc` takes. Outside VTCM, HexagonMemToLLVM
/// rewrites it to a `memref.alloc` - without its alignment -
/// In VTCM it lowers to `hexagon_runtime_alloc_1d`, which rounds the size
/// up to the allocation granularity instead of padding it by the alignment.
std::optional<int64_t> getAllocVTCMBytes(hexagonmem::AllocOp op,
                                         const DataLayout &dataLayout) {
  auto type = cast<MemRefType>(op.getBuffer().getType());
  std::optional<int64_t> bytes = getStaticSizeInBytes(type, dataLayout);
  if (!bytes)
    return std::nullopt;
  if (type.getMemorySpaceAsInt() != ::mlir::hexagon::VTCM_ADDRESS_SPACE) {
    if (!type.getElementType().isSignlessIntOrIndexOrFloat())
      *bytes += static_cast<int64_t>(
          dataLayout.getTypeSize(type.getElementType()).getFixedValue());
    return std::max(*bytes, kVTCMPoolMinimumBytes);
  }
  int64_t granularity = op.getAlignment() == kVTCMLargeAlignment
                            ? kVTCMLargeAlignment
                            : kVTCMPoolMinimumBytes;
  return static_cast<int64_t>(llvm::alignTo(*bytes, granularity));
}

/// Whether `op` allocates on the heap, which the runtime serves from VTCM.
/// Stack allocations use a different resource.
bool allocatesHeapMemory(Operation *op) {
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects)
    return false;
  SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  return llvm::any_of(instances, [](const MemoryEffects::EffectInstance &e) {
    // AutomaticAllocationScopeResource (the stack) is a subresource of
    // DefaultResource, so it has to be excluded explicitly.
    return isa<MemoryEffects::Allocate>(e.getEffect()) &&
           isa<SideEffects::DefaultResource>(e.getResource()) &&
           !isa<SideEffects::AutomaticAllocationScopeResource>(e.getResource());
  });
}

bool isHmxOp(Operation *op) {
  return isa<IREE::Hexagon::HmxMatmulOp, IREE::Hexagon::HmxPackOp,
             IREE::Hexagon::HmxUnpackOp, IREE::Hexagon::HmxAccZeroOp,
             IREE::Hexagon::HmxAccSetupReadOp, IREE::Hexagon::HmxMmaOp,
             IREE::Hexagon::HmxAccReadOp>(op);
}

bool hasVectorValue(Operation *op) {
  auto isVector = [](Type type) { return isa<VectorType>(type); };
  return llvm::any_of(op->getOperandTypes(), isVector) ||
         llvm::any_of(op->getResultTypes(), isVector);
}

/// Whether `block` is part of a control-flow cycle within its region.
bool isInCFGCycle(Block *block) {
  llvm::SmallPtrSet<Block *, 8> visited;
  SmallVector<Block *> worklist(block->getSuccessors());
  while (!worklist.empty()) {
    Block *current = worklist.pop_back_val();
    if (current == block)
      return true;
    if (visited.insert(current).second)
      llvm::append_range(worklist, current->getSuccessors());
  }
  return false;
}

/// Whether `op` may execute repeatedly within one call of `funcOp`: it is
/// nested in a loop op or in a control-flow cycle, the form loops take after
/// SCF-to-CF conversion.
bool isInLoop(Operation *op, FunctionOpInterface funcOp) {
  for (Block *block = op->getBlock(); block;) {
    if (isInCFGCycle(block))
      return true;
    Operation *parent = block->getParentOp();
    if (!parent || parent == funcOp)
      return false;
    if (isa<LoopLikeOpInterface>(parent))
      return true;
    block = parent->getBlock();
  }
  return false;
}

/// The resources one dispatch uses, accumulated over the entry point and the
/// functions it calls.
struct DispatchResources {
  bool hvx = false;
  bool hmx = false;
  int64_t vtcmBytes = 0;
  /// Set when the VTCM use cannot be bounded; the dispatch then declares
  /// `DispatchResourcesAttr::kUnboundedVTCMBytes`.
  bool vtcmUnbounded = false;
};

/// Accumulates the resources `funcOp` and its callees defined in the module
/// use into `resources`. Each allocation is counted once, so the sum bounds the
/// peak only if no allocation sits in a loop. HoistStaticallyBoundAllocations
/// moves them out before this pass, so one that remains in a loop is an error.
LogicalResult collectResources(FunctionOpInterface funcOp, SymbolTable &symbols,
                               const DataLayout &dataLayout,
                               llvm::SmallPtrSetImpl<Operation *> &visited,
                               DispatchResources &resources) {
  if (!visited.insert(funcOp).second || funcOp.isExternal())
    return success();
  WalkResult walkResult = funcOp.walk([&](Operation *op) {
    if (isHmxOp(op))
      resources.hmx = true;
    if (hasVectorValue(op))
      resources.hvx = true;

    if (auto callOp = dyn_cast<func::CallOp>(op)) {
      auto callee = symbols.lookup<FunctionOpInterface>(callOp.getCallee());
      if (callee && failed(collectResources(callee, symbols, dataLayout,
                                            visited, resources)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    if (!allocatesHeapMemory(op))
      return WalkResult::advance();
    if (isInLoop(op, funcOp)) {
      op->emitError("VTCM allocation inside a loop; it should have been "
                    "hoisted out of the loop");
      return WalkResult::interrupt();
    }
    std::optional<int64_t> bytes =
        llvm::TypeSwitch<Operation *, std::optional<int64_t>>(op)
            .Case<memref::AllocOp, hexagonmem::AllocOp>([&](auto allocOp) {
              return getAllocVTCMBytes(allocOp, dataLayout);
            })
            .Default([](Operation *) { return std::nullopt; });
    if (!bytes) {
      op->emitWarning("cannot bound the VTCM this allocation takes; the "
                      "dispatch declares all of VTCM");
      resources.vtcmUnbounded = true;
      return WalkResult::advance();
    }
    resources.vtcmBytes += *bytes;
    return WalkResult::advance();
  });
  return failure(walkResult.wasInterrupted());
}

struct HexagonAnnotateDispatchResourcesPass
    : impl::HexagonAnnotateDispatchResourcesPassBase<
          HexagonAnnotateDispatchResourcesPass> {
  void runOnOperation() override {
    IREE::HAL::ExecutableVariantOp variantOp = getOperation();
    ModuleOp innerModule = variantOp.getInnerModule();
    if (!innerModule)
      return;

    SymbolTable innerSymbols(innerModule);
    DataLayout dataLayout(innerModule);
    for (auto exportOp : variantOp.getExportOps()) {
      auto funcOp =
          innerSymbols.lookup<FunctionOpInterface>(exportOp.getSymName());
      if (!funcOp)
        continue;

      DispatchResources resources;
      llvm::SmallPtrSet<Operation *, 4> visited;
      if (failed(collectResources(funcOp, innerSymbols, dataLayout, visited,
                                  resources)))
        return signalPassFailure();

      int64_t vtcmBytes = resources.vtcmUnbounded
                              ? DispatchResourcesAttr::kUnboundedVTCMBytes
                              : resources.vtcmBytes;
      exportOp->setAttr(
          DispatchResourcesAttr::getAttrName(),
          DispatchResourcesAttr::get(
              &getContext(),
              resources.hvx ||
                  resources.hmx, // HMX functions use HVX instructions
              resources.hmx, vtcmBytes));
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen
