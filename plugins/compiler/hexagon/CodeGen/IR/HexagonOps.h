// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_IR_HEXAGONOPS_H
#define ROOF_HEXAGON_CODEGEN_IR_HEXAGONOPS_H

#include "hexagon/CodeGen/IR/HexagonAttrs.h"
#include "hexagon/CodeGen/IR/HexagonDialect.h"
#include "hexagon/CodeGen/IR/HexagonTypes.h"

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::iree_compiler::IREE::Hexagon {

class IREEHexagonDialect;

} // namespace mlir::iree_compiler::IREE::Hexagon

#define GET_OP_CLASSES
#include "hexagon/CodeGen/IR/HexagonOps.h.inc"

#endif // ROOF_HEXAGON_CODEGEN_IR_HEXAGONOPS_H
