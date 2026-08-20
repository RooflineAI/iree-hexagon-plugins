// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_IR_HEXAGONATTRS
#define ROOF_HEXAGON_CODEGEN_IR_HEXAGONATTRS

include "hexagon/CodeGen/IR/HexagonDialect.td"
include "hexagon/CodeGen/IR/HexagonEnums.td"
include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Profiler zone type.
//===----------------------------------------------------------------------===//

def IREEHexagon_ProfilerZoneAttr :
    EnumAttr<IREEHexagon_Dialect, IREEHexagon_ProfilerZone, "profiler_zone"> {
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
