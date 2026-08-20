// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_IR_HEXAGONTYPES
#define ROOF_HEXAGON_CODEGEN_IR_HEXAGONTYPES

include "hexagon/CodeGen/IR/HexagonDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Runtime state type.
//===----------------------------------------------------------------------===//

def IREEHexagon_RuntimeStateType :
    TypeDef<IREEHexagon_Dialect, "RuntimeState"> {
  let mnemonic = "runtime_state";
  let summary = "Handle to the Hexagon runtime state";
  let description = [{
    Opaque handle to the runtime state. The type lowers to an `!llvm.ptr`
    pointing at the runtime state.
  }];
}

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

#endif // ROOF_HEXAGON_CODEGEN_IR_HEXAGONTYPES
