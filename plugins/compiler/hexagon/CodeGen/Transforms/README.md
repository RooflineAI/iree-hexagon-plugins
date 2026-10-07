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
| `HexagonAssignConstantOrdinals.cpp` | none |
| `HexagonCheckIRBeforeLLVMConversion.cpp` | scalable (vscale) allocation bounds and their option removed |
| `HexagonEmitVectorizationRemarks.cpp` | none |
| `HexagonLinkExecutables.cpp` | none |
| `HexagonPeel.cpp` | none |
| `HexagonTile.cpp` | scalable tile sizes removed; Hexagon tiling levels (adds `hmx`) |
| `HexagonTileAndFuseProducerConsumer.cpp` | scalable tile sizes removed; Hexagon tiling levels (adds `hmx`) |
| `HexagonTileToVectorSize.cpp` | fixed-width config interface |
| `HexagonSplitReduction.cpp` | fixed sizes; absent or unusable split size skips |
| `HexagonSynchronizeSymbolVisibility.cpp` | none |
| `HexagonVectorShapeCastLowering.cpp` | none |
| `HexagonVerifyVectorSizeLegality.cpp` | native-vector budget is a pass option, set by `--iree-hexagon-max-allowed-number-of-native-vectors` |
| `HexagonVectorTransposeLowering.cpp` | AVX2 patterns and option removed; AVX-512 16x16 shuffle network removed |
| `HexagonVirtualVectorLowering.cpp` | ARM/RVV/AArch64 policies and ARM i8mm option removed |

The other passes here (HMX, VTCM, DMA, profiler) were written for Hexagon.

## Unused ukernel handling

`HexagonTileAndFuseProducerConsumer.cpp` still treats
`iree_codegen.ukernel.generic` ops as tiling anchors, as LLVMCPU does, and
`test/codegen/transforms/tile*.mlir` keep the matching upstream cases. Nothing
in the Hexagon pipelines creates these ops, because the Hexagon pipelines never
used the LLVMCPU ukernel lowering passes. This handling may be reworked or
removed in the future. The HMX runtime kernels in
`plugins/runtime/hexagon/dsp/ukernel/hmx` are not affected. They are not IREE
ukernels: HMX lowering calls them as `hal.import.static` runtime functions.

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
