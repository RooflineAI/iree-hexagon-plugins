// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGTYPES
#define PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGTYPES

include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Profiling record type.
//===----------------------------------------------------------------------===//

def IREEHexagon_ProfilingRecordType :
    TypeDef<IREEHexagonEncoding_Dialect, "ProfilingRecord"> {
  let mnemonic = "profiling_record";
  let summary = "Handle to an in-flight Hexagon profiling record";
  let description = [{
    Opaque handle produced by `profiling.begin` and consumed by the matching
    `profiling.end`. It threads the start of a profiling zone to its end. The
    type lowers to an `!llvm.ptr` pointing at the runtime profiling record.
  }];
}

#endif // PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGTYPES
