// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONTYPES
#define PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONTYPES

include "cellar-hexagon/CodeGen/IR/HexagonDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Profiler record type.
//===----------------------------------------------------------------------===//

def IREEHexagon_ProfilerRecordType :
    TypeDef<IREEHexagon_Dialect, "ProfilerRecord"> {
  let mnemonic = "profiler_record";
  let summary = "Handle to an in-flight Hexagon profiler record";
  let description = [{
    Opaque handle produced by `profiler.begin` and consumed by the matching
    `profiler.end`. It threads the start of a profiler zone to its end. The
    type lowers to an `!llvm.ptr` pointing at the runtime profiler record.
  }];
}

#endif // PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONTYPES
