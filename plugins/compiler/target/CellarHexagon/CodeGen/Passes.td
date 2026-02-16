// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_CELLAR_CODEGEN_HEXAGON_PASSES
#define ROOF_CELLAR_CODEGEN_HEXAGON_PASSES

include "mlir/Pass/PassBase.td"

//------------------------------------------------------------------------------
// Hexagon Codegen passes (keep alphabetical)
//------------------------------------------------------------------------------

def HexagonLowerExecutableTargetPass :
    InterfacePass<"iree-hexagon-lower-executable-target",
                  "mlir::FunctionOpInterface"> {
  let summary =
      "Lower executable target using an IREE::HAL::DispatchLoweringPassPipeline."
      "The purpose of this pass is to replace the equivalent pass in LLVMCPU of the same name."
      "This is the pass responsible for scheduling the pipeline selected during LLVMCPUSelectLoweringStrategyPass."
      "Instead of calling the original LLVMCPU pipelines, it now calls the Hexagon ones.";
}

#endif // ROOF_CELLAR_CODEGEN_HEXAGON_PASSES
