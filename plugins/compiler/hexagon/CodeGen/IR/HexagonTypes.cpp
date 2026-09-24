// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/IR/HexagonTypes.h"

#include "hexagon/CodeGen/IR/HexagonDialect.h"
#include "hexagon/CodeGen/IR/HmxContracts.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "hexagon/CodeGen/IR/HexagonTypes.cpp.inc"

namespace mlir::iree_compiler::IREE::Hexagon {

// Parses `<32x32xf32>` into (shape, elementType).
Type HmxAccType::parse(AsmParser &parser) {
  if (parser.parseLess()) {
    return {};
  }
  SmallVector<int64_t> shape;
  if (parser.parseDimensionList(shape, /*allowDynamic=*/false,
                                /*withTrailingX=*/true)) {
    return {};
  }
  Type elementType;
  if (parser.parseType(elementType) || parser.parseGreater()) {
    return {};
  }
  return HmxAccType::getChecked(
      [&] { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), shape, elementType);
}

LogicalResult HmxAccType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<int64_t> shape, Type elementType) {
  if (shape != ArrayRef<int64_t>({kHmxLogicalTileSize, kHmxLogicalTileSize}) ||
      !elementType.isF32()) {
    return emitError() << "expected the fixed HMX accumulator type 32x32xf32";
  }
  return success();
}

void HmxAccType::print(AsmPrinter &printer) const {
  printer << "<";
  for (int64_t dim : getShape()) {
    printer << dim << "x";
  }
  printer << getElementType() << ">";
}

} // namespace mlir::iree_compiler::IREE::Hexagon

void mlir::iree_compiler::IREE::Hexagon::IREEHexagonDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "hexagon/CodeGen/IR/HexagonTypes.cpp.inc"
      >();
}
