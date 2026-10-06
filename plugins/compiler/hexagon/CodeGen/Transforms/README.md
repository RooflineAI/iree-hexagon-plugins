# Transforms

Hexagon passes between strategy selection and LLVM conversion. Ordering is in
[LoweringPipelines.cpp](../Pipelines/LoweringPipelines.cpp); pass contracts are
in [Passes.td](../Passes.td).

## Copies of LLVMCPU passes

These are copies of `Codegen/LLVMCPU/LLVMCPU<Name>.cpp` at IREE `a45adeaa61`,
made to remove CodeGen's dependency on the LLVMCPU backend. Their behaviour,
and the reasons for it, come from upstream.

| file | difference from upstream |
|---|---|
| `HexagonTile.cpp` | scalable tile sizes removed |
| `HexagonTileAndFuseProducerConsumer.cpp` | scalable tile sizes removed |
| `HexagonTileToVectorSize.cpp` | none |
| `HexagonSplitReduction.cpp` | none |

The other passes here (HMX, VTCM, DMA, profiler) were written for Hexagon.
