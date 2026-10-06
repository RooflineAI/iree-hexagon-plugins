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
    vector_common_parallel, vector_reduction and vector_inner_parallel.

    Distribution-stage presence identifies the root, even when all sizes are
    zero. Vector sizes merge the three vector stages along disjoint dimensions;
    overlapping nonzero dimensions have no merged vector shape.

      #iree_hexagon.lowering_config<distribution = [0, 0, 0],
        vector_common_parallel = [8, 32, 0], vector_reduction = [0, 0, 8]>

    Shared Codegen tiling-level attributes store the sizes, allowing generic
    interface consumers to keep using this configuration. VTCM staging is
    configured separately by VTCMTilingConfigAttr.
  }];
  let parameters = (ins AttrParameter<"DictionaryAttr", "Named tiling stages">:$config);
  let builders = [
    AttrBuilder<(ins "llvm::SmallVector<mlir::NamedAttribute>":$items)>,
  ];
  let extraClassDeclaration = [{
    static Attribute getTilingLevelAttr(MLIRContext *ctx,
                                       ArrayRef<int64_t> tileSizes);
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

//===----------------------------------------------------------------------===//
// VTCM tiling configuration.
//===----------------------------------------------------------------------===//

def IREEHexagon_VTCMTilingConfigAttr :
    AttrDef<IREEHexagon_Dialect, "VTCMTilingConfig"> {
  let mnemonic = "vtcm_tiling_config";
  let genMnemonicAlias = 1;
  let summary = [{Hexagon-specific VTCM staging and tiling configuration.}];
  let description = [{
    Configuration consumed by the Hexagon VTCM tiling pass. The tile sizes are
    indexed by linalg loop dimension.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "Loop tile sizes for VTCM staging.">:
        $tile_sizes
  );

  let assemblyFormat = [{
    `<` `tile_sizes` `=` `[` $tile_sizes `]` `>`
  }];
  let genVerifyDecl = 1;
}

#endif // ROOF_HEXAGON_CODEGEN_IR_HEXAGONATTRS
