// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/IR/HexagonAttrs.h"
#include "hexagon/CodeGen/IR/HexagonDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>

#include "hexagon/CodeGen/IR/HexagonEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hexagon/CodeGen/IR/HexagonAttrs.cpp.inc"

using namespace mlir;

namespace mlir::iree_compiler::IREE::Hexagon {

LogicalResult
VTCMTilingConfigAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             ArrayRef<int64_t> tileSizes) {
  // Checking that the VTCM tiles are valid cannot be done here, since we need
  // information about the operation they are being applied to.
  if (tileSizes.empty()) {
    return emitError() << "expected at least one VTCM tile size";
  }
  for (int64_t tileSize : tileSizes) {
    if (tileSize <= 0) {
      return emitError() << "expected positive VTCM tile sizes";
    }
  }

  return success();
}

} // namespace mlir::iree_compiler::IREE::Hexagon

void mlir::iree_compiler::IREE::Hexagon::IREEHexagonDialect::
    registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "hexagon/CodeGen/IR/HexagonAttrs.cpp.inc"
      >();
}
