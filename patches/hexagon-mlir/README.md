# hexagon-mlir patches

The pinned hexagon-mlir source uses older LLVM/MLIR APIs. Our build compiles it
against the same LLVM checkout as IREE, through the local build overlays;
hexagon-mlir does not bring a second LLVM version into the plugin. Compatibility
patches adapt its source and copied runtime utilities to those shared headers.
Other patches provide plugin integration or fix bugs independently of LLVM.

| Patch | Category | Reason |
| --- | --- | --- |
| [convert_to_llvm_pattern_interface_ctor.patch](convert_to_llvm_pattern_interface_ctor.patch) | LLVM compatibility | Add a public forwarding constructor because `ConvertToLLVMPatternInterface` now has a protected constructor. |
| [dialect_use_properties_for_attributes.patch](dialect_use_properties_for_attributes.patch) | LLVM compatibility | Remove the retired `usePropertiesForAttributes` setting and put allocation/layout attributes in `prop-dict`, allowing all vendor dialects to use strict assembly properties. |
| [fork_decompose_hexkl_matmul_vtcm_layout.patch](fork_decompose_hexkl_matmul_vtcm_layout.patch) | Runtime layout fix | Match HexKL's 2048-byte FP16 tile regions and reserve the exact trailing HMX configuration size; the older fixed layout can overlap configuration and tile data. |
| [fork_hexkl_matmul_unsigned_dims.patch](fork_hexkl_matmul_unsigned_dims.patch) | Runtime API compatibility | Match the HexKL matmul declaration/definition to the unsigned 64-bit dimensions expected by its API. This fixed-width interface is independent of index lowering. |
| [fork_runtime_dma_lazy_init.patch](fork_runtime_dma_lazy_init.patch) | Runtime initialization fix | Initialize UserDMA on first use instead of relying on global C++ constructors being run for generated kernel shared objects. |
| [fork_userdma_2d_field_overflow.patch](fork_userdma_2d_field_overflow.patch) | DMA correctness fix | Avoid silent truncation of 2D descriptor fields above 65535 by issuing row-by-row 1D transfers. The fallback trades copy parallelism for correctness. |
| [index_bitwidth_independent_lowering.patch](index_bitwidth_independent_lowering.patch) | 32-bit ABI fix / LLVM compatibility | Skip invalid i32-to-i32 truncations, retain the vendor allocator's explicit i64 alignment, and construct LLVM constants with attributes matching the converted integer type. |
| [linalg_api_compatibility.patch](linalg_api_compatibility.patch) | LLVM compatibility | Adapt the changed `generalizeNamedOp` return type; replace retired elementwise and transposed-matmul classes with `ElementwiseOp` and matmul indexing maps. Preserve transpose folding and the prior transposed-B batch-matmul selection. |
| [llvm_adt_api_updates.patch](llvm_adt_api_updates.patch) | LLVM runtime-header compatibility | Align copied StringMap/hash implementations with current LLVM headers: initialization, linear probing/removal and the core `xxh3_64bits` signature. |
| [misc_build_fixes.patch](misc_build_fixes.patch) | Release-build bug fix / portability | Move stride/width computation out of assertions so it still executes with `NDEBUG`; initialize outputs and reject unsupported layouts. Also fix an unused capture and use `int64_t` rather than platform-dependent `long`. |
| [pass_registration_macros.patch](pass_registration_macros.patch) | Expose plugin APIs / LLVM compatibility | Export DMA, HexKL and HexagonMem conversion-pattern population functions and labels so the plugin can use one configured type converter. Also update generated-pass macros and base namespaces. |
| [remove_copy_op_interface.patch](remove_copy_op_interface.patch) | LLVM compatibility / remove unused API | Remove the unused `CopyOpInterface` trait and its two includes from `hexagonmem.copy`, following LLVM's removal. Replaces the interface copy previously built by the Bazel/CMake overlay. |
| [vector_math_pattern_population_api.patch](vector_math_pattern_population_api.patch) | LLVM compatibility | Replace removed math expansion helpers with the aggregate API and split vector multi-reduction lowering into reorder, flatten and unroll population calls. |

## Index width versus constant attributes

`index` remains the type used for high-level loop bounds, tensor dimensions and
memref indexing. The plugin's earlier 32-bit commit chooses i32 when lowering
it to the LLVM dialect and adjusts runtime metadata to match.

The newer LLVM verifier additionally requires an integer constant's attribute
type to equal its result type. Thus `(1 : index) : i32` becomes
`(1 : i32) : i32` in `llvm.mlir.constant`; this does not select the index width
or replace the earlier commit. The patch uses the converted type, so the same
builder also works with an i64 converter. See the
[upstream verifier change](https://github.com/llvm/llvm-project/commit/7ae1bef46d1d180ede4400df617fe49b152ddec7).
