// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGOPS
#define PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGOPS

include "cellar-hexagon/CodeGen/Encoding/IR/HexagonEncodingDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base op class.
//===----------------------------------------------------------------------===//

class IREEHexagonEncoding_Op<string mnemonic, list<Trait> traits = []> :
    Op<IREEHexagonEncoding_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Profiling marker ops.
//===----------------------------------------------------------------------===//

def IREEHexagon_ProfilingBeginOp :
    IREEHexagonEncoding_Op<"profiling.begin", [MemoryEffects<[MemWrite]>]> {
  let summary = "Begin a Hexagon DSP profiling marker zone";
  let description = [{
    Marks the beginning of a generated profiling zone. The operation lowers to
    `hexagon_runtime_profiling_zone_begin`.
  }];

  let arguments = (ins
      I32Attr:$zone_type,
      OptionalAttr<StrAttr>:$extra_info
  );

  let assemblyFormat = "attr-dict";
}

def IREEHexagon_ProfilingEndOp :
    IREEHexagonEncoding_Op<"profiling.end", [MemoryEffects<[MemWrite]>]> {
  let summary = "End a Hexagon DSP profiling marker zone";
  let description = [{
    Marks the end of a generated profiling zone. The operation lowers to
    `hexagon_runtime_profiling_zone_end`.
  }];

  let assemblyFormat = "attr-dict";
}

#endif // PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGOPS
