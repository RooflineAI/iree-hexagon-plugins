// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lets DMA transfers bypass the data cache where that is safe by construction:
// on their VTCM side, which is not cached, and when reading a Constant binding.
// A Constant binding is an immutable global; its writer ran in an earlier
// execution, and the runtime flushes the cache at the end of every execution,
// so its memory is up to date and stays so. A DMA transfer that bypasses the
// cache becomes a hexagonmem.dma_start.

#include "hexagon/CodeGen/Passes.h"
#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONDMACACHEBYPASSPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

bool isInVTCM(Value memref) {
  auto type = dyn_cast<MemRefType>(memref.getType());
  return type &&
         type.getMemorySpaceAsInt() == ::mlir::hexagon::VTCM_ADDRESS_SPACE;
}

/// Collects the DMA transfers that read `value`, even through aliases.
void collectDMAReads(Value value, SymbolTable &symbols, ModuleOp innerModule,
                     llvm::SmallPtrSetImpl<Value> &visited,
                     llvm::DenseSet<Operation *> &dmaReads) {
  if (!visited.insert(value).second)
    return;
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (isa<memref::DmaStartOp>(user)) {
      if (use.getOperandNumber() == 0)
        dmaReads.insert(user);
      continue;
    }
    if (auto viewOp = dyn_cast<ViewLikeOpInterface>(user);
        viewOp && viewOp.getViewSource() == value) {
      for (Value result : user->getResults())
        collectDMAReads(result, symbols, innerModule, visited, dmaReads);
      continue;
    }
    if (isa<memref::CastOp, memref::AssumeAlignmentOp>(user)) {
      for (Value result : user->getResults())
        collectDMAReads(result, symbols, innerModule, visited, dmaReads);
      continue;
    }
    if (auto callOp = dyn_cast<func::CallOp>(user)) {
      auto callee = symbols.lookup<func::FuncOp>(callOp.getCallee());
      std::optional<SymbolTable::UseRange> calleeUses =
          callee ? SymbolTable::getSymbolUses(callee, innerModule)
                 : std::nullopt;
      if (callee && !callee.isExternal() && calleeUses &&
          llvm::hasSingleElement(*calleeUses)) {
        collectDMAReads(callee.getArgument(use.getOperandNumber()), symbols,
                        innerModule, visited, dmaReads);
      }
    }
  }
}

/// Collects the DMA transfers of `funcOp` that read a Constant binding.
void collectConstantDMAReads(IREE::HAL::ExecutableExportOp exportOp,
                             FunctionOpInterface funcOp, SymbolTable &symbols,
                             ModuleOp innerModule,
                             llvm::DenseSet<Operation *> &dmaReads) {
  IREE::HAL::PipelineLayoutAttr layout = exportOp.getLayout();
  if (!layout)
    return;
  ArrayRef<IREE::HAL::PipelineBindingAttr> bindings = layout.getBindings();
  llvm::SmallPtrSet<Value, 16> visited;
  funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
    uint64_t ordinal = subspanOp.getBinding().getZExtValue();
    if (ordinal < bindings.size() &&
        IREE::HAL::bitEnumContainsAll(bindings[ordinal].getFlags(),
                                      IREE::HAL::DescriptorFlags::Constant))
      collectDMAReads(subspanOp.getResult(), symbols, innerModule, visited,
                      dmaReads);
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
    llvm::DenseSet<Operation *> constantReads;
    for (auto exportOp : variantOp.getExportOps()) {
      if (auto funcOp =
              innerSymbols.lookup<FunctionOpInterface>(exportOp.getSymName()))
        collectConstantDMAReads(exportOp, funcOp, innerSymbols, innerModule,
                                constantReads);
    }

    SmallVector<memref::DmaStartOp> dmaOps;
    innerModule.walk(
        [&](memref::DmaStartOp dmaOp) { dmaOps.push_back(dmaOp); });
    IRRewriter rewriter(innerModule.getContext());
    for (memref::DmaStartOp dmaOp : dmaOps) {
      bool bypassSrc =
          constantReads.contains(dmaOp) || isInVTCM(dmaOp.getSrcMemRef());
      bool bypassDst = isInVTCM(dmaOp.getDstMemRef());
      if (!bypassSrc && !bypassDst)
        continue;
      rewriter.setInsertionPoint(dmaOp);
      rewriter.replaceOpWithNewOp<hexagonmem::DmaStartOp>(
          dmaOp, dmaOp.getSrcMemRef(), dmaOp.getSrcIndices(),
          dmaOp.getDstMemRef(), dmaOp.getDstIndices(), dmaOp.getNumElements(),
          dmaOp.getTagMemRef(), dmaOp.getTagIndices(), dmaOp.getStride(),
          dmaOp.getNumElementsPerStride(), bypassSrc, bypassDst);
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen
