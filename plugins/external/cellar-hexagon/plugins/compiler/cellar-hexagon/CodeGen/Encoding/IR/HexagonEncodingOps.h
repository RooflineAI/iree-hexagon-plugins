// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGOPS_H
#define PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGOPS_H

#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingDialect.h"
#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingOps.h.inc"

#endif // PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGOPS_H
