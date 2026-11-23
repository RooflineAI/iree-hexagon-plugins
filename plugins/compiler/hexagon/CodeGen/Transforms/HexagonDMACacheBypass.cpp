// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Finds the bindings each dispatch accesses only by DMA, lets those DMA
// transfers bypass the data cache, and records the bindings on the
// hal.executable.export for the DSP runtime's cache maintenance.

#include "hexagon/CodeGen/IR/HexagonDialect.h"
#include "hexagon/CodeGen/Passes.h"
#include "hexagon/Common/Common.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONDMACACHEBYPASSPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

/// Unit attributes on `memref.dma_start` that let its source or destination
/// bypass the data cache (read by hexagon-mlir's DMA lowering).
constexpr StringLiteral kBypassSrcAttrName = "bypass_src";
constexpr StringLiteral kBypassDstAttrName = "bypass_dst";

/// Offset alignment of buffers a DMA writes by bypassing the data cache: the L2
/// line size, so the first line of the range holds no data of another buffer.
constexpr int64_t kBypassWriteAlignment = 128;

/// How the memory of one binding is accessed, over all its uses.
struct BindingAccess {
  /// DMA transfers using the binding, and whether it is their source (else
  /// their destination).
  SmallVector<std::pair<memref::DmaStartOp, bool>> dmas;
  /// Some use is not a DMA transfer (or cannot be followed).
  bool other = false;
};

bool isInVTCM(Value memref) {
  auto type = dyn_cast<MemRefType>(memref.getType());
  return type &&
         type.getMemorySpaceAsInt() == ::mlir::hexagon::VTCM_ADDRESS_SPACE;
}

/// collect every place a binding's memory is used, even through aliases
void collectBindingAccess(Value value, SymbolTable &symbols,
                          ModuleOp innerModule,
                          llvm::SmallPtrSetImpl<Value> &visited,
                          BindingAccess &access) {
  if (!visited.insert(value).second)
    return;
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (auto dmaOp = dyn_cast<memref::DmaStartOp>(user)) {
      bool isSrc = use.getOperandNumber() == 0;
      bool isDst = use.getOperandNumber() == dmaOp.getSrcMemRefRank() + 1;
      if (isSrc || isDst)
        access.dmas.push_back({dmaOp, isSrc});
      else
        access.other = true;
      continue;
    }
    if (auto viewOp = dyn_cast<ViewLikeOpInterface>(user);
        viewOp && viewOp.getViewSource() == value) {
      for (Value result : user->getResults())
        collectBindingAccess(result, symbols, innerModule, visited, access);
      continue;
    }
    if (isa<memref::CastOp, memref::AssumeAlignmentOp>(user)) {
      for (Value result : user->getResults())
        collectBindingAccess(result, symbols, innerModule, visited, access);
      continue;
    }
    if (auto callOp = dyn_cast<func::CallOp>(user)) {
      auto callee = symbols.lookup<func::FuncOp>(callOp.getCallee());
      std::optional<SymbolTable::UseRange> calleeUses =
          callee ? SymbolTable::getSymbolUses(callee, innerModule)
                 : std::nullopt;
      if (callee && !callee.isExternal() && calleeUses &&
          llvm::hasSingleElement(*calleeUses)) {
        collectBindingAccess(callee.getArgument(use.getOperandNumber()),
                             symbols, innerModule, visited, access);
        continue;
      }
    }
    // other access, e.g. a direct load
    access.other = true;
  }
}

/// Finds the bindings `funcOp` accesses only by DMA transfers, lets those
/// transfers bypass the data cache, and returns the bit mask of the bindings.
///
/// A written binding only bypasses the cache if no other binding of the
/// dispatch is written through it: a cache line the two share at their edges
/// could be written back over the result of the DMA. The export's layout tells
/// which bindings are only read. Fails if a written binding would bypass the
/// cache but buffers are aligned to less than kBypassWriteAlignment.
FailureOr<uint64_t> annotateDMABypass(IREE::HAL::ExecutableExportOp exportOp,
                                      FunctionOpInterface funcOp,
                                      int64_t bufferAlignment,
                                      SymbolTable &symbols,
                                      ModuleOp innerModule) {
  IREE::HAL::PipelineLayoutAttr layout = exportOp.getLayout();
  llvm::SmallDenseMap<uint64_t, BindingAccess> bindings;
  llvm::SmallPtrSet<Value, 16> visited;
  funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
    collectBindingAccess(subspanOp.getResult(), symbols, innerModule, visited,
                         bindings[subspanOp.getBinding().getZExtValue()]);
  });

  auto isDMAOnly = [](const BindingAccess &access) {
    return !access.other && !access.dmas.empty();
  };
  auto isReadOnly = [&](uint64_t ordinal) {
    ArrayRef<IREE::HAL::PipelineBindingAttr> layoutBindings =
        layout ? layout.getBindings()
               : ArrayRef<IREE::HAL::PipelineBindingAttr>();
    return ordinal < layoutBindings.size() &&
           IREE::HAL::bitEnumContainsAll(layoutBindings[ordinal].getFlags(),
                                         IREE::HAL::DescriptorFlags::ReadOnly);
  };
  auto writes = [&](uint64_t ordinal, const BindingAccess &access) {
    if (isReadOnly(ordinal))
      return false;
    return access.other || llvm::any_of(access.dmas, [](const auto &dma) {
             return !dma.second;
           });
  };
  bool writeThroughCache = llvm::any_of(bindings, [&](const auto &entry) {
    return !isDMAOnly(entry.second) && writes(entry.first, entry.second);
  });

  uint64_t mask = 0;
  for (const auto &[ordinal, access] : bindings) {
    if (ordinal >= 64 || !isDMAOnly(access) ||
        (writes(ordinal, access) && writeThroughCache))
      continue;
    if (writes(ordinal, access) && bufferAlignment < kBypassWriteAlignment) {
      return exportOp.emitError()
             << "binding " << ordinal
             << " is written by DMA bypassing the data cache, which needs "
                "buffers aligned to at least "
             << kBypassWriteAlignment << " bytes, but they are aligned to "
             << bufferAlignment
             << "; pass --iree-stream-resource-min-offset-alignment="
             << kBypassWriteAlignment;
    }
    mask |= uint64_t{1} << ordinal;
    for (auto [dmaOp, isSrc] : access.dmas)
      dmaOp->setAttr(isSrc ? kBypassSrcAttrName : kBypassDstAttrName,
                     UnitAttr::get(dmaOp->getContext()));
  }
  return mask;
}

/// VTCM does not have a cache, so it is fine to set the bypass cache flag of
/// DMA transfers on the VTCM side.
void annotateVTCMBypass(ModuleOp innerModule) {
  innerModule.walk([](memref::DmaStartOp dmaOp) {
    if (isInVTCM(dmaOp.getSrcMemRef()))
      dmaOp->setAttr(kBypassSrcAttrName, UnitAttr::get(dmaOp->getContext()));
    if (isInVTCM(dmaOp.getDstMemRef()))
      dmaOp->setAttr(kBypassDstAttrName, UnitAttr::get(dmaOp->getContext()));
  });
}

struct HexagonDMACacheBypassPass
    : impl::HexagonDMACacheBypassPassBase<HexagonDMACacheBypassPass> {
  void runOnOperation() override {
    IREE::HAL::ExecutableVariantOp variantOp = getOperation();
    ModuleOp innerModule = variantOp.getInnerModule();
    if (!innerModule)
      return;

    SymbolTable innerSymbols(innerModule);
    // stream.resources if set, else the --iree-stream-resource-* defaults
    int64_t bufferAlignment =
        IREE::Stream::ResourceConfigAttr::lookup(variantOp)
            .getMinBufferOffsetAlignment();
    Builder builder(&getContext());
    for (auto exportOp : variantOp.getExportOps()) {
      auto funcOp =
          innerSymbols.lookup<FunctionOpInterface>(exportOp.getSymName());
      if (!funcOp)
        continue;
      FailureOr<uint64_t> mask = annotateDMABypass(
          exportOp, funcOp, bufferAlignment, innerSymbols, innerModule);
      if (failed(mask))
        return signalPassFailure();
      if (*mask != 0)
        exportOp->setAttr(
            IREE::Hexagon::IREEHexagonDialect::getDMABypassBindingsAttrName(),
            builder.getI64IntegerAttr(static_cast<int64_t>(*mask)));
    }
    annotateVTCMBypass(innerModule);
  }
};

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen
