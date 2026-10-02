# Hexagon Compiler Plugin

## Purpose

The plugin adds a Hexagon HAL target to IREE's codegen and provides the pieces needed to:

1. register the plugin with the compiler,
2. expose a `hexagon` target backend/device,
3. run Hexagon-owned codegen pipelines and passes,
4. serialize linked Hexagon executables into the final VMFB.

## Control Flow

The main compiler-side flow is:

1. `PluginRegistration.cpp` registers the plugin entrypoint.
2. `HexagonSession` registers dialects, passes, target device, and target
   backend.
3. `HexagonTargetBackend` exposes the `hexagon` executable target to HAL.
4. `HexagonTargetBackend::getExecutableTarget(...)` injects target config,
   including the Hexagon encoding resolver attribute.
5. HAL calls back into the plugin to build:
   - the configuration pipeline,
   - the translation pipeline,
   - the linking pipeline.
6. The translation pipeline lowers to LLVM IR / Hexagon object code.
7. `serializeHexagonExecutable(...)` serializes the final executable payload.
8. `HexagonLinkerTool` is used during linking/serialization to produce the
   shared object embedded in the VMFB.

## Linking of functions from Hexagon-mlir

Hexagon currently supports direct external calls resolved through static or
native DSP runtime linking. Such declarations use `hal.import.static`. A
variant-level `hexagon.native_runtime_linking` marker separately permits the
known runtime symbols to remain undefined in the kernel shared object so that
the DSP loader can resolve them from `libhexagon_dsp_skel.so`.

Generic HAL dynamic imports and bitcode-import ABI rewriting are currently
unsupported. Compilation diagnoses an unclassified external call instead of
silently emitting an executable that the runtime cannot load. See
[External calls and HAL imports](CodeGen/Conversion/README.md#external-calls-and-hal-imports)
for more information.

The native policy applies to the hardcoded DMA, HexKL, HexagonMem, HMX, and
profiler symbols, plus lowering helpers such as `malloc`, `free`, and
`memrefCopy` after they are renamed to their Hexagon runtime entry points.

## Code Relationships

### `Target/`

`Target/` was originally created in the image of `LLVMCPUTarget.cpp`.
Given the size of the original file, it was decomposed into multiple smaller files with
divided responsibilities, but it mainly covers the same file with hexagon-specific adaptations.
As Hexagon-specific behavior accumulates, these files are diverging more and more from LLVMCPU,
but the original design lineage is still important context for maintenance.
The decomposition covers the following files:

- `HexagonSession.*`
  - plugin lifecycle,
  - dialect/pass registration,
  - target registration.
- `HexagonTargetBackend.*`
  - HAL target backend implementation,
  - executable target attribute creation,
  - hooks into configuration/translation/linking pipelines,
  - executable serialization entrypoint.
- `HexagonTargetDevice.*`
  - target device registration and device-level configuration.
- `HexagonLLVMTarget.*`
  - LLVM target options and target triple/data layout related material.
- `HexagonExecutableSerialization.*`
  - final executable packaging.

The `Target` layer depends on `CodeGen`, not the other way around.

This folder also contains:

- `Linking/HexagonLinkerTool.*`
  - tool invocation used by serialization/linking.

Note that making the Hexagon plugin independent from LLVMCPU is a work in progress.
Currently, there exists an overlap between the plugins, and the linker classes and structure are an example of this.

### `CodeGen/Passes.*`

`CodeGen/Passes.h` is the main public header for the codegen package.
The original implementation is copied from LLVMCPU's `passes.cpp`, and `CodeGen`
contains a decomposition of this big original monolithic file.

It exposes:

- pass registration,
- pipeline registration,
- pipeline builder entrypoints,
- TableGen-generated pass declarations.

Everything else in `CodeGen/` should be thought of as implementation detail for
those entrypoints.

### `CodeGen/Pipelines/`

This directory is about pipeline construction, not individual transformations.

IREE's HAL expects three main pipelines that will be called in order:

- `ConfigurationPipeline.cpp`
- `TranslationPipeline.cpp`
- `LinkingPipeline.cpp`

Finally, Hexagon's Translation pipeline is currently in an experimental state. As such,
it currently has two different pipelines under development that are likely to be removed in the future:

- `HexagonMlirPipeline.*`
  - experimental route inspired by hexagon-mlir.
- `IreeLoweringPipelines.*`
  - Hexagon-adapted versions of IREE/LLVMCPU lowering sequences.

### `CodeGen/Strategy/`

This directory contains passes centered about deciding how a dispatch should be lowered,
not about performing the lowering itself. It is currently only usable when triggering the
IreeLoweringPipelines for translation using the appropriate flags.

- `HexagonSelectLoweringStrategy.cpp`
  - pass wrapper that drives strategy selection.
- `KernelDispatch.*`
  - actual launch-config and lowering-config policy logic.
- `Planning/` contains the implementation of the pass that drives strategy selection
  See [`Planning/README.md`](CodeGen/Strategy/Planning/README.md) for more information about it.

### `CodeGen/Conversion/` and `CodeGen/Transforms/`

These directories contain the actual Hexagon-specific passes used by the
pipelines.

The final LLVM conversion architecture, including its relationship
to HMX lowering, debugging requirements, Hexagon-MLIR integration, and the
follow-up LLVMCPU dependency work, is documented in
[`CodeGen/Conversion/README.md`](CodeGen/Conversion/README.md).

- `Conversion/`
  - module-level conversions: HMX-to-runtime-call lowering and the final
    conversion to LLVM, which also classifies native runtime symbols.
- `Transforms/`
  - smaller, local canonicalization or adaptation passes used inside the
    Hexagon lowering flow.
