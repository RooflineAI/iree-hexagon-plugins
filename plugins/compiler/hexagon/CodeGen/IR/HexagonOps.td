// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_CODEGEN_IR_HEXAGONOPS
#define HEXAGON_CODEGEN_IR_HEXAGONOPS

include "hexagon/CodeGen/IR/HexagonAttrs.td"
include "hexagon/CodeGen/IR/HexagonDialect.td"
include "hexagon/CodeGen/IR/HexagonTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"

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
    level. It is expected to be lowered to
    `bufferization.alloc_tensor` before bufferization.

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

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// HMX compute operations.
//===----------------------------------------------------------------------===//

def IREEHexagon_HmxMatmulOp :
    IREEHexagon_Op<"hmx.matmul",
                   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = [{
    Buffer-level HMX matmul over operands already in the HMX tile layout.
  }];
  let description = [{
    Side-effecting buffer form of the HMX tensor-unit multiply. Tensor-level
    tiling uses `hmx.tensor_matmul`; bufferization converts it to this op, which
    writes the accumulator read-out into `acc`. `HexagonExpandHmxMatmulPass`
    expands this op into the real HMX op sequence (`hmx.acc.zero`, K-loop of
    `hmx.mma`, and `hmx.acc.read`).
  }];

  let arguments = (ins
    AnyMemRef:$lhs,
    AnyMemRef:$rhs,
    AnyMemRef:$acc
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $acc `:` type($acc) `)` attr-dict
  }];

  let hasVerifier = 1;
}

def IREEHexagon_HmxPackOp :
    IREEHexagon_Op<"hmx.pack",
                   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Buffer-level in-tile row-pair interleave into the HMX layout";
  let description = [{
    Side-effecting buffer form of the HMX input layout transform. Tensor-level
    tiling uses `hmx.tensor_pack`; bufferization converts it to this op, which
    writes into `dest`.
  }];

  let arguments = (ins
    AnyMemRef:$source,
    AnyMemRef:$dest,
    I64Attr:$dim
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $source `:` type($source) `)` `outs` `(` $dest `:` type($dest) `)`
    attr-dict
  }];

  let hasVerifier = 1;
}

def IREEHexagon_HmxUnpackOp :
    IREEHexagon_Op<"hmx.unpack",
                   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Buffer-level unpack HMX accumulator read-out into destination";
  let description = [{
    Side-effecting buffer form of the HMX accumulator read-out layout transform.
    Tensor-level tiling uses `hmx.tensor_unpack`; bufferization converts it to
    this op, which adds the unpacked product into `dest`.
  }];

  let arguments = (ins
    AnyMemRef:$source,
    AnyMemRef:$dest,
    I64Attr:$dim
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $source `:` type($source) `)` `outs` `(` $dest `:` type($dest) `)`
    attr-dict
  }];

  let hasVerifier = 1;
}

def IREEHexagon_TensorHmxMatmulOp :
    IREEHexagon_Op<"hmx.tensor_matmul",
                   [AllTypesMatch<["acc", "result"]>,
                    DestinationStyleOpInterface,
                    DeclareOpInterfaceMethods<TilingInterface,
                       ["getIterationDomain",
                        "getLoopIteratorTypes",
                        "isOpFusableWithConsumerSlice",
                        "getResultTilePosition",
                        "getTiledImplementation",
                        "getIterationDomainTileFromResultTile",
                        "generateResultTileValue"]>,
                    Pure]> {
  let summary = [{
    Tensor-level HMX matmul placeholder over packed operands.
  }];
  let description = [{
    Pure tensor form used for tiling/fusion before bufferization. It consumes
    `lhs`/`rhs` operands in HMX tile-major layout and reduces over the whole K
    tile grid into the accumulator-read tensor `acc`. The `acc` DPS operand
    supplies the result shape and destination storage, but its contents are not
    read: HMX accumulation starts from cleared hardware register state and fully
    overwrites the accumulator-read tensor. A logical matmul initializer,
    including bias or other non-zero contents, must instead be passed as the
    destination of `hmx.tensor_unpack`, which adds the unpacked product to that
    initializer.
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    AnyRankedTensor:$acc
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $acc `:` type($acc) `)` attr-dict `->` type($result)
  }];

  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getAccMutable(); }
  }];

  let hasVerifier = 1;
}

def IREEHexagon_TensorHmxPackOp :
    IREEHexagon_Op<"hmx.tensor_pack",
                   [AllTypesMatch<["dest", "result"]>,
                    DestinationStyleOpInterface, Pure]> {
  let summary = "Tensor-level in-tile row-pair interleave into HMX layout";
  let description = [{
    Pure tensor form of the HMX input layout transform. It interleaves `source`
    along dimension `dim` by a factor of 2 into `dest`; bufferization converts
    it to side-effecting `hmx.pack`.
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    AnyRankedTensor:$dest,
    I64Attr:$dim
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    `ins` `(` $source `:` type($source) `)` `outs` `(` $dest `:` type($dest) `)`
    attr-dict `->` type($result)
  }];

  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDestMutable(); }
  }];

  let hasVerifier = 1;
}

def IREEHexagon_TensorHmxUnpackOp :
    IREEHexagon_Op<"hmx.tensor_unpack",
                   [AllTypesMatch<["dest", "result"]>,
                    DestinationStyleOpInterface,
                    DeclareOpInterfaceMethods<TilingInterface,
                       ["getIterationDomain",
                        "getLoopIteratorTypes",
                        "isOpFusableWithProducerSlices",
                        "getResultTilePosition",
                        "getTiledImplementation",
                        "getTiledImplementationFromOperandTiles",
                        "getIterationDomainTileFromOperandTiles",
                        "generateResultTileValue"]>,
                    Pure]> {
  let summary = "Tensor-level unpack HMX accumulator read-out";
  let description = [{
    Pure tensor form of the HMX accumulator read-out layout transform.
    De-interleaves `source` along dimension `dim`
    (splitting the trailing size-2 tile back out), the mirror of the input
    interleave applied to the accumulator read-out. In the HMX matmul lowering,
    the hardware accumulator is computed from zero; this tensor op models adding
    the unpacked product into `dest`, preserving the original linalg.matmul
    init value.
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    AnyRankedTensor:$dest,
    I64Attr:$dim
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    `ins` `(` $source `:` type($source) `)` `outs` `(` $dest `:` type($dest) `)`
    attr-dict `->` type($result)
  }];

  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDestMutable(); }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// HMX tensor-unit primitives (post-bufferization).
//===----------------------------------------------------------------------===//

def IREEHexagon_HmxAccZeroOp :
    IREEHexagon_Op<"hmx.acc.zero", [MemoryEffects<[MemWrite]>]> {
  let summary = "Clear the HMX accumulator, producing a zeroed accumulator value";
  let description = [{
    Models `Q6_mxclracc_hf`: clears the HMX accumulator register state and
    yields a fresh accumulator SSA value. The memory-write effect pins ordering
    against the other HMX ops (all touch the single global accumulator).
  }];

  let results = (outs IREEHexagon_HmxAccType:$acc);

  let assemblyFormat = "attr-dict `:` qualified(type($acc))";
}

def IREEHexagon_HmxAccSetupReadOp :
    IREEHexagon_Op<"hmx.acc.setup_read", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Program HMX accumulator read conversion state";
  let description = [{
    Programs the bias/scale state consumed by `hmx.acc.read`. For the FP16
    read-out path this is a pass-through bias block in VTCM: scale=1.0 and all
    other terms zero. The setup is separate from `hmx.acc.read` because the
    hardware state persists and should be initialized once before the tile loop,
    not reloaded for every accumulator read.
  }];

  let arguments = (ins AnyMemRef:$config);

  let results = (outs);

  let assemblyFormat = "$config attr-dict `:` type($config)";

  let hasVerifier = 1;
}

def IREEHexagon_HmxMmaOp :
    IREEHexagon_Op<"hmx.mma",
                   [AllTypesMatch<["accIn", "accOut"]>,
                    MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "One 32x32x32 HMX multiply-accumulate into the accumulator";
  let description = [{
    Models one activation/weight multiply pair
    (`Q6_activation_hf_mxmem_RR` + `Q6_weight_hf_mxmem_RR`): reads one HMX-layout
    activation tile (`lhs`) and weight tile (`rhs`) from VTCM and accumulates
    their product into `accIn`, yielding the updated accumulator `accOut`.
  }];

  let arguments = (ins
    AnyMemRef:$lhs,
    AnyMemRef:$rhs,
    IREEHexagon_HmxAccType:$accIn
  );

  let results = (outs IREEHexagon_HmxAccType:$accOut);

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $accIn attr-dict `:`
    type($lhs) `,` type($rhs) `,` qualified(type($accIn)) `->`
    qualified(type($accOut))
  }];

  let hasVerifier = 1;
}

def IREEHexagon_HmxAccReadOp :
    IREEHexagon_Op<"hmx.acc.read", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Read the HMX accumulator out to a VTCM buffer";
  let description = [{
    Models `Q6_mxmem_AR_after_hf`: converts/reads the accumulator register state
    and writes it into the destination VTCM buffer `dest`. Destination-passing
    style (no SSA result); consumers read `dest`.
  }];

  let arguments = (ins
    IREEHexagon_HmxAccType:$acc,
    AnyMemRef:$dest
  );

  let results = (outs);

  let assemblyFormat = "$acc `,` $dest attr-dict `:` qualified(type($acc)) `,` type($dest)";

  let hasVerifier = 1;
}

#endif // ROOF_HEXAGON_CODEGEN_IR_HEXAGONOPS
