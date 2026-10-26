// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HEXAGON_CODEGEN_IR_HEXAGONATTRS
#define IREE_HEXAGON_CODEGEN_IR_HEXAGONATTRS

include "hexagon/CodeGen/IR/HexagonDialect.td"
include "hexagon/CodeGen/IR/HexagonEnums.td"
include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def IREEHexagon_LoweringConfigAttr :
    AttrDef<IREEHexagon_Dialect, "LoweringConfig", [
      DeclareAttrInterfaceMethods<IREECodegen_LoweringConfigAttrInterface, [
        "getWorkgroupTileSizes", "getTilingLevelSizes", "hasTilingLevel",
        "hasWorkgroupTilingLevel", "getNumTilingLevels",
        "getStaticTilingLevelSizes", "getTilingLevelAttr", "getVectorSizes"
      ]>
    ]> {
  let mnemonic = "lowering_config";
  let summary = "Fixed-width tiling configuration for Hexagon lowering";
  let description = [{
    Named tiling stages use nonnegative, fixed tile sizes; zero leaves a
    dimension untiled. All present stages describe the same iteration rank.
    Stages are distribution, cache_parallel, cache_reduction,
    vector_common_parallel, vector_reduction, vector_inner_parallel, vtcm and
    hmx.

    Distribution-stage presence identifies the root, even when all sizes are
    zero. Vector sizes merge the three vector stages along disjoint dimensions;
    overlapping nonzero dimensions have no merged vector shape.

      #iree_hexagon.lowering_config<distribution = [0, 0, 0],
        vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 8]>

    The vtcm stage, set only on the root, is the dispatch-wide tile staged into
    VTCM. HexagonVTCMTilingPass consumes it and removes it from the tiled
    operations.

    The hmx stage tiles operations in the HMX pipeline into HMX-sized work. On
    the root matmul it tiles batch dimensions to one before HMX conversion; on
    the hmx.tensor_unpack created by the conversion it selects the number of
    HMX output tiles per iteration of the tile grid.

    Shared Codegen tiling-level attributes store the sizes, allowing generic
    interface consumers to keep using this configuration.
  }];
  let parameters = (ins AttrParameter<"DictionaryAttr", "Named tiling stages">:$config);
  let builders = [
    AttrBuilder<(ins "llvm::SmallVector<mlir::NamedAttribute>":$items)>,
  ];
  let extraClassDeclaration = [{
    static Attribute getTilingLevelAttr(MLIRContext *ctx,
                                       ArrayRef<int64_t> tileSizes);

    /// Returns this configuration without `level`, or a null attribute when
    /// no stage remains.
    LoweringConfigAttr withoutTilingLevel(TilingLevel level) const;
  }];
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Profiler zone type.
//===----------------------------------------------------------------------===//

def IREEHexagon_ProfilerZoneAttr :
    EnumAttr<IREEHexagon_Dialect, IREEHexagon_ProfilerZone, "profiler_zone"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Lowering pipeline.
//===----------------------------------------------------------------------===//

def IREEHexagon_PipelineAttr : AttrDef<IREEHexagon_Dialect, "Pipeline", [
    DeclareAttrInterfaceMethods<IREECodegen_PipelineAttrInterface>]> {
  let mnemonic = "pipeline";
  let summary = "Hexagon lowering pipeline identifier.";
  let description = [{
    Identifies an Hexagon lowering pipeline. Pipeline construction is delegated
    to the builder callback registered by Hexagon codegen.
  }];
  let parameters = (ins
    EnumParameter<IREEHexagon_LoweringPipeline>:$value
  );
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Dispatch resources.
//===----------------------------------------------------------------------===//

def IREEHexagon_DispatchResourcesAttr :
    AttrDef<IREEHexagon_Dialect, "DispatchResources"> {
  let mnemonic = "dispatch_resources";
  let summary = "Hardware resources a dispatch needs on the DSP.";
  let description = [{
    Records the hardware units and the VTCM a dispatch uses, so the
    DSP part of the runtime can decide which dispatches may run concurrently.
    HexagonAnnotateDispatchResourcesPass measures them on the final IR before
    LLVM conversion and sets the attribute on the `hal.executable.export`;
    serialization encodes it into the Hexagon extension of the executable's
    dispatch attributes.

    `threads` is set if the dispatch runs parts of its work on async worker
    threads (threads inside a VTCM tile).

    `vtcm_bytes` bounds the peak VTCM the dispatch allocates, 0 if it does not
    use VTCM, and `kUnboundedVTCMBytes` (INT64_MAX) if it cannot be bounded,
    so the dispatch must be treated as needing all of VTCM.

      #iree_hexagon.dispatch_resources<hvx = true, hmx = false,
                                       threads = false, vtcm_bytes = 1048576>
  }];
  let parameters = (ins
    "bool":$hvx,
    "bool":$hmx,
    "bool":$threads,
    "int64_t":$vtcm_bytes
  );
  let assemblyFormat = [{
    `<` `hvx` `=` $hvx `,` `hmx` `=` $hmx `,`
    `threads` `=` $threads `,` `vtcm_bytes` `=` $vtcm_bytes `>`
  }];
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// Name under which the attribute is stored on the
    /// `hal.executable.export`.
    static constexpr StringLiteral getAttrName() {
      return StringLiteral("iree_hexagon.dispatch_resources");
    }

    /// `vtcm_bytes` of a dispatch whose VTCM use cannot be bounded.
    static constexpr int64_t kUnboundedVTCMBytes =
        std::numeric_limits<int64_t>::max();
  }];
}

#endif // IREE_HEXAGON_CODEGEN_IR_HEXAGONATTRS
