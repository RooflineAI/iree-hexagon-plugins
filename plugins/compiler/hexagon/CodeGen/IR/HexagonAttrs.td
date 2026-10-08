// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HEXAGON_CODEGEN_IR_HEXAGONATTRS
#define HEXAGON_CODEGEN_IR_HEXAGONATTRS

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
// Encoding Resolvers.
//===----------------------------------------------------------------------===//

def IREEHexagon_HexagonEncodingResolverAttr :
    AttrDef<IREEHexagon_Dialect, "HexagonEncodingResolver"> {
  let mnemonic = "hexagon_encoding_resolver";
  let summary = [{Hexagon-specific encoding resolver.}];
  let description = [{
    Custom resolver for Hexagon inspired from IREECPU_CPUEncodingResolverAttr.
    The attribute stores the executable target configuration so external models
    can materialize encodings with Hexagon specific layout and tiling
    information.
  }];

  let assemblyFormat = "`<` struct(params) `>`";

  let parameters = (ins
    OptionalParameter<"DictionaryAttr",
                      "Executable target configuration consumed by the "
                      "external models during encoding materialisation.">:
        $configuration
  );
}

#endif // ROOF_HEXAGON_CODEGEN_IR_HEXAGONATTRS
