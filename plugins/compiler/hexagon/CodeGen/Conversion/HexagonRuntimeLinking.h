// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_CONVERSION_HEXAGONRUNTIMELINKING_H_
#define ROOF_HEXAGON_CODEGEN_CONVERSION_HEXAGONRUNTIMELINKING_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::iree_compiler::hexagon::codegen {

// This is meant to be used as an attribute marker for functions that are linked
// from the Hexagon runtime.
inline constexpr llvm::StringLiteral kNativeRuntimeLinkAttrName =
    "hexagon.native_runtime_link";
// This is meant to be used as an attribute marker for a hal.executable.variant.
// It will be used during serialization to allow for undefined symbols.
inline constexpr llvm::StringLiteral kNativeRuntimeLinkVariantAttrName =
    "hexagon.native_runtime_linking";

const llvm::StringSet<> &getHexagonRuntimeSymbolNames();
const llvm::StringSet<> &getHexKLSymbolNames();
const llvm::StringSet<> &getNativeRuntimeHelperAliasNames();
const llvm::StringSet<> &getCanonicalNativeRuntimeHelperNames();

inline bool isHexagonRuntimeSymbol(llvm::StringRef symbolName) {
  return getHexagonRuntimeSymbolNames().contains(symbolName);
}

inline bool isHexKLSymbol(llvm::StringRef symbolName) {
  return getHexKLSymbolNames().contains(symbolName);
}

inline bool isNativeRuntimeHelperAlias(llvm::StringRef symbolName) {
  return getNativeRuntimeHelperAliasNames().contains(symbolName);
}

inline bool isCanonicalNativeRuntimeHelper(llvm::StringRef symbolName) {
  return getCanonicalNativeRuntimeHelperNames().contains(symbolName);
}

llvm::StringRef getNativeRuntimeLinkedSymbolName(llvm::StringRef symbolName);

inline bool isNativeRuntimeLinkCandidate(llvm::StringRef symbolName) {
  return isHexagonRuntimeSymbol(symbolName) || isHexKLSymbol(symbolName) ||
         isNativeRuntimeHelperAlias(symbolName) ||
         isCanonicalNativeRuntimeHelper(symbolName);
}

// Marks a parent variant with the attribute above to indicate that it should
// allow undefined symbols during serialization
void markNativeRuntimeLinkingVariant(Operation *op);

// Rename the function to custom hexagon naming and tag it with the attribute
// above.
LogicalResult renameAndTagNativeRuntimeLinkedFunc(ModuleOp moduleOp,
                                                  LLVM::LLVMFuncOp funcOp);

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_CONVERSION_HEXAGONRUNTIMELINKING_H_
