// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HEXAGON_CODEGEN_CONVERSION_HEXAGONRUNTIMELINKING_H_
#define IREE_HEXAGON_CODEGEN_CONVERSION_HEXAGONRUNTIMELINKING_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::iree_compiler::hexagon::codegen {

// This is meant to be used as an attribute marker for a hal.executable.variant.
// It will be used during serialization to allow for undefined symbols.
inline constexpr llvm::StringLiteral kNativeRuntimeLinkVariantAttrName =
    "hexagon.native_runtime_linking";

// Allocates VTCM from the dispatch's partition:
//   void *(const hexagon_executable_dispatch_state_t *, size_t nbytes,
//          uint64_t alignment)
inline constexpr llvm::StringLiteral kVtcmAllocFnName =
    "hexagon_runtime_vtcm_alloc";
// Allocator of the coroutine frames of the async threads, in DDR:
//   void *(int64_t alignment, int64_t size) and void (void *)
inline constexpr llvm::StringLiteral kCoroAllocFnName =
    "hexagon_runtime_coro_alloc";
inline constexpr llvm::StringLiteral kCoroFreeFnName =
    "hexagon_runtime_coro_free";
// HexKL matmul variants that take their VTCM scratch from the caller.
inline constexpr llvm::StringLiteral kHexKLMatmulScratchBytesFnName =
    "hexkl_matmul_f16f16_f32_scratch_bytes";
inline constexpr llvm::StringLiteral kHexKLMatmulWithScratchFnName =
    "hexkl_matmul_f16f16_f32_with_scratch";

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

// Verifies that every direct external call uses the static/native linking
// policy supported by the Hexagon plugin.
LogicalResult validateHexagonExternalCalls(ModuleOp moduleOp);

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // IREE_HEXAGON_CODEGEN_CONVERSION_HEXAGONRUNTIMELINKING_H_
