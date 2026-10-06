// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PASSES_TD_
#define ROOF_HEXAGON_CODEGEN_PASSES_TD_

include "mlir/Pass/PassBase.td"

//------------------------------------------------------------------------------
// Hexagon Codegen passes (keep alphabetical)
//------------------------------------------------------------------------------

def HexagonPeelPass :
    InterfacePass<"iree-hexagon-peel", "mlir::FunctionOpInterface"> {
  let summary = "Pass to perform peeling on non-distributed loops.";
}

def HexagonSplitReductionPass :
    InterfacePass<"iree-hexagon-split-reduction", "mlir::FunctionOpInterface"> {
  let summary = "Pass to splitReduce linalg operations.";
  let options = [
    Option<"enableFpReductionReordering", "enable-fp-reduction-reordering",
      "bool", /*default=*/"false",
      "Flag to enable reduction reordering on floating points.">,
  ];
}

def HexagonTilePass :
    InterfacePass<"iree-hexagon-tile", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile TilingInterface operations.";
  let description = [{
    Walk through all the TilingInterface operations and apply the tiling, if the
    given level is found in the operation's lowering config.
  }];
  let options = [
    Option<"tilingLevel", "tiling-level", "IREE::CPU::TilingLevel",
      /*default=*/"IREE::CPU::TilingLevel::InvalidLevel",
      "Tiling level used to retrieve the configuration from lowering_config.",
      [{llvm::cl::values(
        clEnumValN(IREE::CPU::TilingLevel::DistributionTiles, "distribution",
                   "Use `distribution` tile sizes."),
        clEnumValN(IREE::CPU::TilingLevel::CacheParallelTiles, "cache_parallel",
                   "Use `cache_parallel` tile sizes."),
        clEnumValN(IREE::CPU::TilingLevel::CacheReductionTiles, "cache_reduction",
                   "Use `cache_reduction` tile sizes."),
        clEnumValN(IREE::CPU::TilingLevel::VectorCommonParallelTiles, "vector_common_parallel",
                   "Use `vector_common_parallel` tile sizes."),
        clEnumValN(IREE::CPU::TilingLevel::VectorReductionTiles, "vector_reduction",
                   "Use `vector_reduction` tile sizes."),
        clEnumValN(IREE::CPU::TilingLevel::VectorInnerParallelTiles, "vector_inner_parallel",
                   "Use `vector_inner_parallel` tile sizes.")
      )}]>,
    Option<"skipRootOp", "skip-root-op", "bool", /*default=*/"false",
      "Do not tile the root op if the option is true.">
  ];
}

def HexagonTileAndFuseProducerConsumerPass
    : InterfacePass<"iree-hexagon-tile-and-fuse-producer-consumer",
                    "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile root op and fuse with producer and consumer "
                "TilingInterface ops.";
  let options =
      [Option<"tilingLevel", "tiling-level", "IREE::CPU::TilingLevel",
           /*default=*/"IREE::CPU::TilingLevel::VectorInnerParallelTiles",
           "The tiling level used to retrieve the configuration from lowering_config",
           [{llvm::cl::values(
              clEnumValN(IREE::CPU::TilingLevel::DistributionTiles, "distribution",
                         "Use `distribution` tile sizes."),
              clEnumValN(IREE::CPU::TilingLevel::CacheParallelTiles, "cache_parallel",
                         "Use `cache_parallel` tile sizes."),
              clEnumValN(IREE::CPU::TilingLevel::CacheReductionTiles, "cache_reduction",
                         "Use `cache_reduction` tile sizes."),
              clEnumValN(IREE::CPU::TilingLevel::VectorCommonParallelTiles, "vector_common_parallel",
                         "Use `vector_common_parallel` tile sizes."),
              clEnumValN(IREE::CPU::TilingLevel::VectorReductionTiles, "vector_reduction",
                         "Use `vector_reduction` tile sizes."),
              clEnumValN(IREE::CPU::TilingLevel::VectorInnerParallelTiles, "vector_inner_parallel",
                         "Use `vector_inner_parallel` tile sizes.")

           )}]>,
       Option<"onlyFuseProducerInputOperands",
              "only-fuse-producer-input-operands", "bool",
              /*default=*/"false",
              "Specifies if we only want to fuse producer's input operands. "
              "This is helpful to tile&fuse in case of reduction dimensions.">,
       Option<"anchorOnRootOp",
              "anchor-on-root-op", "bool",
              /*default=*/"true",
              "Anchor on the root op to start the tiling. Otherwise, find the "
              "last configured operations before and after the root op or ukernel.">];
}

def HexagonTileToVectorSizePass :
    InterfacePass<"iree-hexagon-tile-to-vector-size", "mlir::FunctionOpInterface"> {
  let summary = "Tile TilingInterface operations to target vector size.";
  let description = [{
    Walk through all the TilingInterface operations and tiling the dimensions to
    target vector sizes, if the lowering config is present and the dimension is
    known as greater than the vector size.

    It is intended to be used before vectorization that avoids big vectors and
    stack buffers.
  }];
}

def HexagonVectorShapeCastLoweringPass :
    InterfacePass<"iree-hexagon-vector-shape-cast-lowering", "mlir::FunctionOpInterface"> {
  let summary = "Pass to lower vector.shape_cast ops.";
}

def HexagonVectorTransposeLoweringPass :
    InterfacePass<"iree-hexagon-vector-transpose-lowering", "mlir::FunctionOpInterface"> {
  let summary = "Pass to lower vector.transpose ops.";
}

def HexagonVirtualVectorLoweringPass :
    InterfacePass<"iree-hexagon-virtual-vector-lowering", "mlir::FunctionOpInterface"> {
  let summary = "Pass to lower high level vector operations like contract or multidim reduce ops to lower level vector ops.";
  let options = [
    Option<"splitVectorTransfersTo", "split-transfers", "std::string",
      /*default=*/"",
      [{Split vector transfers between slow (masked) and fast "
        "(unmasked) variants. Possible options are:\n"
          "\tnone [default]: keep unsplit vector.transfer and pay the price\n"
          "\tlinalg-copy: use linalg.fill + linalg.generic for the slow path\n"
          "\tvector-transfers: use extra small unmasked vector.transfers for"
          " the slow path\n}]>,
  ];
}

def HexagonVTCMTilingPass :
    InterfacePass<"iree-hexagon-vtcm-tiling",
                  "mlir::FunctionOpInterface"> {
  let summary = [{
      Apply Hexagon config-driven VTCM staging and tiling to a dispatch.
  }];
}

def HexagonLowerVTCMStagingPass :
    InterfacePass<"iree-hexagon-lower-vtcm-staging",
                  "mlir::FunctionOpInterface"> {
  let summary = [{
      Lower iree_hexagon VTCM staging ops to bufferization.alloc_tensor,
      preserving copies and materializing VTCM allocations.
  }];
}

def HexagonLowerHmxToCallsPass :
    Pass<"iree-hexagon-lower-hmx-to-calls", "mlir::ModuleOp"> {
  let summary = [{
      Validate the bufferized HMX runtime ABI, lower HMX layout and accumulator
      primitives to native DSP kernel calls, and remove the opaque accumulator
      token.
  }];
}

def HexagonConvertMatmulToHmxPass :
    InterfacePass<"iree-hexagon-convert-matmul-to-hmx",
                  "mlir::FunctionOpInterface"> {
  let summary = [{
      Convert eligible f16 linalg.matmul operations into tensor-level HMX ops.
      Runs after VTCM tiling and before LLVMCPU inner tiling. Inputs and output
      must originate from Hexagon VTCM staging operations. Static, ragged, and
      bounded dynamic dimensions are padded to complete HMX tiles by the pack
      kernels. Results may be f16 or f32.
  }];
}

def HexagonExpandHmxMatmulPass :
    InterfacePass<"iree-hexagon-expand-hmx-matmul",
                  "mlir::FunctionOpInterface"> {
  let summary = [{
      Expand the bufferized iree_hexagon.hmx.matmul (whole-K reduction for one
      32x32 output tile) into the real HMX op sequence: an hmx.acc.zero, a K
      reduction loop of hmx.mma over the per-k activation/weight tiles, and a
      final hmx.acc.read into the accumulator buffer. Runs after bufferization,
      on VTCM memrefs.
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
      "Lower executable target using the pipeline carried on translation_info."
      "The purpose of this pass is to replace the equivalent pass in LLVMCPU of the same name."
      "This is the pass responsible for scheduling the pipeline selected during the Hexagon lowering-strategy pass."
      "Instead of calling the original LLVMCPU pipelines, it now calls the Hexagon ones.";
}

def HexagonSelectLoweringStrategyPass :
    Pass<"iree-hexagon-select-lowering-strategy", "mlir::ModuleOp"> {
  let summary = [{
      Select lowering configs and translation_info for Hexagon dispatches.
  }];
}

def InsertProfilerMarkersPass :
    InterfacePass<"iree-hexagon-insert-profiler-markers",
                  "mlir::FunctionOpInterface"> {
  let summary = [{
      Insert Hexagon DSP profiler marker ops around selected high-level
      operations.
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

#endif // ROOF_HEXAGON_CODEGEN_PASSES_TD_
