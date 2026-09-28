# Hexagon conversion to LLVM

This directory contains the final conversion of a prepared Hexagon dispatch to
LLVM dialect. The conversion is one dialect-conversion
transaction: all participating pattern families share one type converter, one
conversion target, and one legality decision.

## Components

`HexagonConvertToLLVM.cpp` owns the conversion driver. It configures the target
triple, data layout, and address-space conversions; populates standard MLIR,
local-executable ABI, Hexagon runtime, HexagonMem, DMA, and HexKL patterns; and
invokes `applyPartialConversion` once. It also reconciles temporary
materializations, classifies native runtime declarations, and invokes the
post-conversion import rewrites.

Hexagon dialect operations that map directly to the DSP runtime are in this
file. This includes `iree_hexagon.get_runtime_state` and the profiler begin/end
operations. Their `runtime_state` and `profiler_record` values are opaque
runtime objects represented as LLVM pointers. Keeping these patterns in the
driver distinguishes target runtime lowering from the IREE local-executable
ABI implementation.

`HexagonDispatchABI.{h,cpp}` owns runtime-visible ABI structure definitions and
semantic accessors.

`HexagonABIToLLVM.{h,cpp}` owns the Hexagon copy of IREE's local-executable ABI
conversion:

- public entry-point conversion;
- executable and interface constant loads;
- workgroup ID, size, and count loads;
- interface binding descriptor construction;
- all dispatch instrumentation records; and
- bitcode and dynamic import ABI rewrites.

`HexagonRuntimeLinking.{h,cpp}` owns recognition and tagging of native DSP
runtime symbols. Tagged calls remain direct unresolved references and are not
rewritten through the HAL dynamic-import thunk.

## Conversion contract

The driver maintains these invariants:

1. All dialect-conversion patterns use the same `LLVMTypeConverter`.
2. Hexagon memory-space conversions are installed before any patterns are
   populated.
3. Standard MLIR, HAL ABI, Hexagon runtime, HexagonMem, DMA, and HexKL
   operations are legalized by one `applyPartialConversion` invocation.
4. Temporary `unrealized_conversion_cast` operations are reconciled inside the
   pass, and any remaining cast is a pass failure.
5. Native runtime symbols are classified before eligible external calls are
   rewritten as HAL imports.
6. Import rewriting is post-conversion canonicalization. It does not construct
   another type converter or create a second LLVM conversion phase.

The production Hexagon data layout has 32-bit pointers. The current type
converter intentionally retains a 64-bit index representation because the
Hexagon-MLIR conversions expect it. ABI tests therefore use the production data
layout and check both pointer-sensitive structures and the required
`i16`/`i32` to `i64` index extensions.

## Dispatch ABI

A public dispatch entry point uses the standard IREE local-executable
signature and returns an `i32` status:

```text
environment pointer, dispatch-state pointer, workgroup-state pointer -> i32
```

The three arguments are non-null, no-alias, no-undef, and 16-byte aligned. A
successful dispatch currently returns zero.

Hexagon extends the allocation addressed by the second argument:

```text
{
  iree_hal_executable_dispatch_state_v0_t base;
  hexagon_rt_state_t *runtime_state;
}
```

The base state remains at offset zero and the runtime-state pointer immediately
follows it. Compiler access is centralized in `loadRuntimeState`; runtime-side
static assertions in `plugins/runtime/hexagon/dsp/rt/dispatch_state.h` protect
the matching C layout.

Changing any entry argument, structure member, field width, import parameter
packing, or instrumentation record is an ABI change. Compiler and runtime
changes must be made and tested together.

## Debugging

The public pass is `iree-hexagon-convert-to-llvm`. Custom patterns retain stable
debug labels:

- `iree-hal-abi-to-llvm`
- `hexagon-runtime-to-llvm`
- `hexagon-mem-to-llvm`
- `hexagon-dma-to-llvm`
- `hexkl-to-llvm`

In a debug build, `--debug-only=dialect-conversion` reports individual pattern
applications and match failures. The labels provide component-level
observability even though there are no intermediate production passes between
the pattern families.

## Tests

The focused tests are organized by runtime-visible contract:

- `hexagon_abi_to_llvm.mlir`: entry signature, attributes, workgroup fields,
  constants, production data layout, and extended runtime state;
- `hexagon_abi_bindings.mlir`: binding ordinal and byte-offset handling plus
  static, dynamic, strided, and sub-byte memref descriptors;
- `hexagon_abi_instrumentation.mlir`: all four instrumentation conversions,
  record headers/sizes, ring-buffer operations, and unsupported values;
- `hexagon_import_abi.mlir`: dynamic, aliased, static, native, and bitcode
  imports, extra ABI fields, and multiple results;
- `lower_profiler_markers.mlir`: profiler helper reuse, string deduplication,
  empty metadata, zone values, and record threading;
- `hexagon_runtime_to_llvm_invalid.mlir`: malformed runtime-state access and
  incompatible profiler helper diagnostics; and
- `single_driver_custom_lowerings.mlir`: composition of standard, HexagonMem,
  DMA, and HexKL lowering in the single conversion transaction.
