// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Derived from Codegen/LLVMCPU/Utils.cpp at IREE revision
// a45adeaa6115e446c898e6eb21fb6edc0e65ddc4.

#ifndef ROOF_HEXAGON_CODEGEN_TARGET_CONFIG_H_
#define ROOF_HEXAGON_CODEGEN_TARGET_CONFIG_H_

#include <optional>

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::iree_compiler::hexagon::codegen {

// Executable target configuration keys, written by the Hexagon target backend.
inline constexpr llvm::StringLiteral kNativeVectorSizeAttrName =
    "native_vector_size";
inline constexpr llvm::StringLiteral kMaxStackAllocationSizeAttrName =
    "max_stack_allocation_size";

// Configuration values are in bytes. Callers supply their own defaults.
inline std::optional<int64_t>
getConfigNativeVectorSize(DictionaryAttr targetConfig) {
  if (auto attr = targetConfig.getAs<IntegerAttr>(kNativeVectorSizeAttrName)) {
    return attr.getInt();
  }
  return std::nullopt;
}

inline std::optional<int64_t>
getConfigMaxStackAllocationSize(DictionaryAttr targetConfig) {
  if (auto attr =
          targetConfig.getAs<IntegerAttr>(kMaxStackAllocationSizeAttrName)) {
    return attr.getInt();
  }
  return std::nullopt;
}

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_TARGET_CONFIG_H_
