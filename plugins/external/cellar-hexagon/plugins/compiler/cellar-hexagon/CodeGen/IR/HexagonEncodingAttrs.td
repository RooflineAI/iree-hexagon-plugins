// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGATTRS
#define PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGATTRS

include "cellar-hexagon/CodeGen/IR/HexagonEncodingDialect.td"
include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Encoding Resolvers.
//===----------------------------------------------------------------------===//

def IREEHexagon_HexagonEncodingResolverAttr :
    AttrDef<IREEHexagonEncoding_Dialect, "HexagonEncodingResolver"> {
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

#endif // PATIO_PLUGINS_CELLARHEXAGON_CODEGEN_IR_HEXAGONENCODINGATTRS
