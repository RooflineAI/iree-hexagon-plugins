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
| `HexagonCheckIRBeforeLLVMConversion.cpp` | scalable (vscale) allocation bounds and their option removed |
| `HexagonEmitVectorizationRemarks.cpp` | none |
| `HexagonPeel.cpp` | none |
| `HexagonTile.cpp` | scalable tile sizes removed |
| `HexagonTileAndFuseProducerConsumer.cpp` | scalable tile sizes removed |
| `HexagonTileToVectorSize.cpp` | none |
| `HexagonSplitReduction.cpp` | none |
| `HexagonVectorShapeCastLowering.cpp` | none |
| `HexagonVerifyVectorSizeLegality.cpp` | native-vector budget is a pass option, set by `--iree-hexagon-max-allowed-number-of-native-vectors` |
| `HexagonVectorTransposeLowering.cpp` | AVX2 patterns and option removed; AVX-512 16x16 shuffle network removed |
| `HexagonVirtualVectorLowering.cpp` | ARM/RVV/AArch64 policies and ARM i8mm option removed |

The other passes here (HMX, VTCM, DMA, profiler) were written for Hexagon.

## Shape-cast and transpose lowering

`HexagonVectorShapeCastLowering.cpp` and `HexagonVectorTransposeLowering.cpp`
only apply upstream MLIR pattern sets. Neither pass encodes Hexagon policy yet;
they are where Hexagon-specific lowerings of these ops belong.

Transposes become a single 1D `vector.shuffle`, which leaves the whole
permutation to LLVM's HVX shuffle selection. If profiling shows the generic
shuffle lowers poorly, HVX transpose networks built on `vshuff`/`vdeal` would go
in this pass, sized for HVX registers. Currently, HMX's 32x32 tile layouts are
produced by the HMX conversion, not by `vector.transpose`.

Shape casts use the upstream lowering unchanged. Hexagon-specific cancellations,
or reshapes that stay within HVX register boundaries, could also be considered
to improve performance.
