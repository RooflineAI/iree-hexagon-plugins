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

// TODO: Decide if we want to do this from the target instead
def EmbedHexKLLinkObjectsPass :
    Pass<"iree-hexagon-embed-hexkl-link-objects", "mlir::ModuleOp"> {
  let summary = "Attach required HexKL static archives as hal.executable objects";
}

def FoldDispatchOutputStagingPass :
    InterfacePass<"iree-hexagon-fold-dispatch-output-staging",
                  "mlir::FunctionOpInterface"> {
  let summary = [{
      Fold output staging tensors produced as
      `bufferization.alloc_tensor copy(...) {memory_space = 0}` directly into
      `iree_tensor_ext.dispatch.tensor.store` value operands.
  }];
}

def HexagonConvertToLLVMPass :
    Pass<"iree-hexagon-convert-to-llvm", "mlir::ModuleOp"> {
  let summary =
      "Perform final conversion from Linalg/HAL/Shape/Vector/Standard to LLVMIR dialect";
  let options = [
    Option<"reassociateFpReductions", "reassociateFpReductions", "bool",
            /*default=*/"false",
           "Specifies if FP add and mult reductions can be reordered">,
    Option<"lowerMemRefFinalization", "lower-memref-finalization", "bool",
           /*default=*/"true",
           "Lower memref descriptors/deallocations to LLVM in this pass. "
           "When false, keep this step external so callers can place "
           "address-space normalization in between.">,
    Option<"lowerHalBindingSubspan", "lower-hal-binding-subspan", "bool",
           /*default=*/"true",
           "Lower hal.interface.binding.subspan in this pass. "
           "When false, keep it legal so memref users lowered in a later phase "
           "can still consume memref descriptors.">,
    Option<"targetTriple", "target-triple", "std::string", /*default=*/"",
           "Code generation target triple.">,
    Option<"targetDataLayout", "target-data-layout", "std::string",
           /*default=*/"",
           "Code generation target data layout.">,
  ];
}

def HexagonLowerExecutableTargetPass :
    InterfacePass<"iree-hexagon-lower-executable-target",
                  "mlir::FunctionOpInterface"> {
  let summary =
      "Lower executable target using an IREE::HAL::DispatchLoweringPassPipeline."
      "The purpose of this pass is to replace the equivalent pass in LLVMCPU of the same name."
      "This is the pass responsible for scheduling the pipeline selected during the Hexagon lowering-strategy pass."
      "Instead of calling the original LLVMCPU pipelines, it now calls the Hexagon ones.";
}

def HexagonSelectLoweringStrategyPass :
    Pass<"iree-hexagon-select-lowering-strategy", "mlir::ModuleOp"> {
  let summary = [{
      Select lowering configs and translation_info for Hexagon dispatches
      without reusing LLVMCPU's full KernelDispatch policy.
  }];
}

def MarkHexKLStaticImportsPass :
    Pass<"iree-hexagon-mark-hexkl-static-imports", "mlir::ModuleOp"> {
  let summary = [{"Mark HexKL runtime llvm.func declarations as hal.import.static.
      This avoids IREE's 'RewriteExternCallOpToDynamicImportCallOp' rewriting hexagon function calls to HAL import thunks."
  }];
}

def MarkHexagonRuntimeStaticImportsPass :
    Pass<"iree-hexagon-mark-runtime-static-imports", "mlir::ModuleOp"> {
  let summary = [{
      Mark Hexagon runtime llvm.func declarations as hal.import.static.
      Covers Hexagon DMA/HexagonMem runtime APIs to avoid
      dynamic import-thunk rewriting for symbols expected to resolve at link time.
  }];
}

def PromoteDMATagAllocToStackPass :
    InterfacePass<"iree-hexagon-promote-dma-tag-alloc-to-stack",
                  "mlir::FunctionOpInterface"> {
  let summary = [{
      Convert HexmemCpyToDMA-generated DMA tag allocations from heap
      (memref.alloc memref<1xi32>) to stack (memref.alloca), and drop the
      matching dealloc.
  }];
}

def StripEncodingFromHexKLMatmulPass :
    InterfacePass<"iree-hexagon-strip-encoding-from-hexkl-matmul",
                  "mlir::FunctionOpInterface"> {
  let summary = "Strip and restore IREE encodings around hexkl.matmul tensor ops";
}

#endif // ROOF_CELLAR_CODEGEN_HEXAGON_PASSES
