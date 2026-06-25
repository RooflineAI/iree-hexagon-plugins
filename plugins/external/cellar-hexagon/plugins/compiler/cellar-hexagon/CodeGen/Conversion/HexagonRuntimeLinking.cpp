// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/CodeGen/Conversion/HexagonRuntimeLinking.h"

#include "hexagon/Conversion/DMAToLLVM/DMAExternalFnNames.h"
#include "hexagon/Conversion/HexKLToLLVM/HexKLExternalFnNames.h"
#include "hexagon/Conversion/HexagonMemToLLVM/HexagonMemExternalFnNames.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {

const llvm::StringSet<> &getHexagonRuntimeSymbolNames() {
  static const llvm::StringSet<> names = [] {
    llvm::StringSet<> set;
    set.insert(mlir::hexagon::getDMAStartFnName());
    set.insert(mlir::hexagon::getDMA2DStartFnName());
    set.insert(mlir::hexagon::getDMAWaitFnName());
    set.insert(mlir::hexagonmem::getAllocFnName());
    set.insert(mlir::hexagonmem::getDeallocFnName());
    set.insert(mlir::hexagonmem::getCopyFnName());
    set.insert("hexagon_runtime_profiling_zone_begin");
    set.insert("hexagon_runtime_profiling_zone_end");
    return set;
  }();
  return names;
}

const llvm::StringSet<> &getHexKLSymbolNames() {
  static const llvm::StringSet<> names = [] {
    llvm::StringSet<> set;
    set.insert(mlir::hexkl::getMatmulMicroFnName());
    set.insert(mlir::hexkl::getHmxConfigSizeFnName());
    set.insert(mlir::hexkl::getHmxSetupAccReadF16FnName());
    set.insert(mlir::hexkl::getHmxAccClearF16FnName());
    set.insert(mlir::hexkl::getHmxAccReadF16FnName());
    set.insert(mlir::hexkl::getHmxCopySubmatrixToF16FnName());
    set.insert(mlir::hexkl::getHmxRmToAhF16FnName());
    set.insert(mlir::hexkl::getHmxRmToWhF16FnName());
    set.insert(mlir::hexkl::getHmxMmF16FnName());
    set.insert(mlir::hexkl::getHmxAhToRmF16FnName());
    set.insert(mlir::hexkl::getHmxCopyF16ToF32SubmatrixFnName());
    return set;
  }();
  return names;
}

namespace {

const llvm::StringMap<llvm::StringLiteral> &getNativeRuntimeHelperNameMap() {
  static const llvm::StringMap<llvm::StringLiteral> names = [] {
    llvm::StringMap<llvm::StringLiteral> map;
    map.insert({"malloc", "hexagon_runtime_malloc"});
    map.insert({"free", "hexagon_runtime_free"});
    map.insert({"memrefCopy", "hexagon_runtime_memref_copy"});
    return map;
  }();
  return names;
}

} // namespace

const llvm::StringSet<> &getNativeRuntimeHelperAliasNames() {
  static const llvm::StringSet<> names = [] {
    llvm::StringSet<> set;
    for (const auto &entry : getNativeRuntimeHelperNameMap()) {
      set.insert(entry.getKey());
    }
    return set;
  }();
  return names;
}

const llvm::StringSet<> &getCanonicalNativeRuntimeHelperNames() {
  static const llvm::StringSet<> names = [] {
    llvm::StringSet<> set;
    for (const auto &entry : getNativeRuntimeHelperNameMap()) {
      set.insert(entry.getValue());
    }
    return set;
  }();
  return names;
}

llvm::StringRef getNativeRuntimeLinkedSymbolName(llvm::StringRef symbolName) {
  if (const auto it = getNativeRuntimeHelperNameMap().find(symbolName);
      it != getNativeRuntimeHelperNameMap().end()) {
    return it->second;
  }
  return symbolName;
}

void markNativeRuntimeLinkingVariant(Operation *op) {
  if (!op)
    return;
  if (auto variantOp = op->getParentOfType<
                       mlir::iree_compiler::IREE::HAL::ExecutableVariantOp>()) {
    variantOp->setAttr(kNativeRuntimeLinkVariantAttrName,
                       UnitAttr::get(op->getContext()));
  }
}

LogicalResult renameAndTagNativeRuntimeLinkedFunc(ModuleOp moduleOp,
                                                  LLVM::LLVMFuncOp funcOp) {
  if (!funcOp || !funcOp.isExternal())
    return success();

  auto symName = funcOp.getSymNameAttr();
  if (!symName || !isNativeRuntimeLinkCandidate(symName.getValue()))
    return success();

  llvm::StringRef linkedName =
      getNativeRuntimeLinkedSymbolName(symName.getValue());
  if (linkedName != symName.getValue()) {
    auto linkedNameAttr = StringAttr::get(moduleOp.getContext(), linkedName);
    if (failed(SymbolTable::replaceAllSymbolUses(funcOp, linkedNameAttr,
                                                 moduleOp.getOperation()))) {
      return funcOp.emitOpError()
             << "failed to rewrite symbol uses for native runtime-linked "
                "helper '"
             << symName.getValue() << "'";
    }
    funcOp.setSymName(linkedName);
  }

  funcOp->setAttr(kNativeRuntimeLinkAttrName,
                  UnitAttr::get(moduleOp.getContext()));
  markNativeRuntimeLinkingVariant(moduleOp);
  return success();
}

} // namespace mlir::iree_compiler::cellar_hexagon::codegen
