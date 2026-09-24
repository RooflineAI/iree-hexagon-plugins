// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_CODEGEN_IR_HEXAGONTYPES
#define HEXAGON_CODEGEN_IR_HEXAGONTYPES

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

//===----------------------------------------------------------------------===//
// HMX accumulator type.
//===----------------------------------------------------------------------===//

def IREEHexagon_HmxAccType :
    TypeDef<IREEHexagon_Dialect, "HmxAcc"> {
  let mnemonic = "hmx.acc";
  let summary = "HMX tensor-unit accumulator (register) state";
  let description = [{
    Opaque SSA value modeling the HMX accumulator, which is physical
    tensor-unit register state rather than addressable memory. It is produced
    by `hmx.acc.zero`, threaded through the K reduction (as an `scf.for`
    iter_arg) by `hmx.mma`, and consumed by `hmx.acc.read`. There is no
    load-from-memory into the accumulator in hardware, so this value can only be
    created by a clear and updated by a multiply-accumulate.

    The `shape`/`elementType` parameters describe the logical accumulator tile
    (e.g. `32x32xf32`).
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "::mlir::Type":$elementType
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

#endif // ROOF_HEXAGON_CODEGEN_IR_HEXAGONTYPES
