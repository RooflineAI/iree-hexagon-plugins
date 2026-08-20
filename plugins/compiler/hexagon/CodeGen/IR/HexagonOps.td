// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_IR_HEXAGONOPS
#define ROOF_HEXAGON_CODEGEN_IR_HEXAGONOPS

include "hexagon/CodeGen/IR/HexagonAttrs.td"
include "hexagon/CodeGen/IR/HexagonDialect.td"
include "hexagon/CodeGen/IR/HexagonTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base op class.
//===----------------------------------------------------------------------===//

class IREEHexagon_Op<string mnemonic, list<Trait> traits = []> :
    Op<IREEHexagon_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Runtime state ops.
//===----------------------------------------------------------------------===//

def IREEHexagon_GetRuntimeStateOp :
    IREEHexagon_Op<"get_runtime_state", [Pure]> {
  let summary = "Get the active Hexagon runtime state";
  let description = [{
    Produces the runtime state used by the surrounding dispatch. The operation
    lowers to loading the runtime state pointer from Hexagon's extended dispatch
    state. As the loaded pointer is not changing while a dispatch is running and
    nothing else that is compiled in a dispatch can read or write this pointer,
    it is safe here to not annotate [MemoryEffects<[MemRead]>].
  }];

  let arguments = (ins);

  let results = (outs IREEHexagon_RuntimeStateType:$state);

  let assemblyFormat = "attr-dict `:` type($state)";
}

//===----------------------------------------------------------------------===//
// Profiler marker ops.
//===----------------------------------------------------------------------===//

def IREEHexagon_ProfilerBeginOp :
    IREEHexagon_Op<"profiler.begin", [MemoryEffects<[MemWrite]>]> {
  let summary = "Begin a Hexagon DSP profiler marker zone";
  let description = [{
    Marks the beginning of a generated profiler zone. The operation lowers to
    `hexagon_runtime_profiler_zone_begin`.
  }];

  let arguments = (ins
      IREEHexagon_RuntimeStateType:$state,
      IREEHexagon_ProfilerZoneAttr:$zone_type,
      OptionalAttr<StrAttr>:$extra_info
  );

  let results = (outs IREEHexagon_ProfilerRecordType:$record);

  let assemblyFormat =
      "$state attr-dict `:` type($state) `->` type($record)";
}

def IREEHexagon_ProfilerEndOp :
    IREEHexagon_Op<"profiler.end", [MemoryEffects<[MemWrite]>]> {
  let summary = "End a Hexagon DSP profiler marker zone";
  let description = [{
    Marks the end of a generated profiler zone. The operation lowers to
    `hexagon_runtime_profiler_zone_end`. The `record` operand is the value
    produced by the corresponding `profiler.begin` operation.
  }];

  let arguments = (ins IREEHexagon_ProfilerRecordType:$record);

  let assemblyFormat = "$record attr-dict `:` type($record)";
}

//===----------------------------------------------------------------------===//
// VTCM staging operations.
//===----------------------------------------------------------------------===//

def IREEHexagon_StageToVTCMOp :
    IREEHexagon_Op<"stage_to_vtcm",
                   [AllTypesMatch<["source", "result"]>, Pure]> {
  let summary = [{
    Mark a tensor value for later materialization in VTCM.
  }];
  let description = [{
    Tensor-level marker indicating that the given tensor value should be
    materialized in VTCM after tiling is complete.

    This op does not perform a copy, allocate storage, initiate DMA, or
    otherwise model a memory side effect. It is intentionally `Pure`: generic
    tensor tiling and fusion passes may clone, move, speculate, or DCE this op
    using normal SSA value semantics.

    This op is used as an opaque fusion barrier. Unlike
    `bufferization.alloc_tensor copy(...)`, it does not implement
    `TilingInterface` and is not a `linalg::LinalgOp`, so downstream SCF
    producer-fusion utilities do not tile/fuse through it.

    This is important because `bufferization.alloc_tensor copy(...)` carries
    allocation effects. Some tiling/fusion utilities may speculatively clone
    producers before rejecting them for fusion; if those producers already
    carry allocation effects, the dead clones are not removed by standard DCE.
    Keeping this op as a pure tensor-level marker avoids introducing real
    allocation effects before tiling is complete.

    `HexagonLowerVTCMStagingPass` must lower this op immediately before
    bufferization to `bufferization.alloc_tensor copy(...)` with the
    appropriate VTCM memory-space/materialization semantics. No
    `stage_to_vtcm` ops should survive past that lowering boundary.
  }];

  let arguments = (ins
    AnyRankedTensor:$source
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = [{
    $source attr-dict `:` type($result)
  }];
}

def IREEHexagon_VTCMEmptyOp :
    IREEHexagon_Op<"vtcm_empty", [Pure]> {
  let summary = [{
    Create a tensor-level empty value that should later materialize in VTCM.
  }];
  let description = [{
    Tensor-level marker for an uninitialized tensor value that should be
    materialized in VTCM after tiling is complete.

    This op is analogous to a VTCM-placed `tensor.empty` at the tensor IR
    level.

    For more information, read the description from its sister op `IREEHexagon_StageToVTCMOp`.
  }];

  // Explicit size values for each dynamic dimension in $result, following the
  // same convention as tensor.empty. Must have exactly as many entries as
  // there are `?` dimensions in the result type.
  let arguments = (ins Variadic<Index>:$dynamicSizes);

  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = [{
    `(` $dynamicSizes `)` attr-dict `:` type($result)
  }];
}

#endif // ROOF_HEXAGON_CODEGEN_IR_HEXAGONOPS
