# Cellar Hexagon Compiler Plugin

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

### `CodeGen/Encoding/`

This subtree has two layers:

- `Encoding/IR/`
  - defines the Hexagon encoding dialect,
  - defines the Hexagon encoding resolver attribute,
  - owns the generated IR/TableGen files.
- `Encoding/HexagonEncodingExternalModels.*`
  - attaches external interfaces to the Hexagon resolver attribute,
  - implements layout resolution/materialization/serialization hooks,
  - consumes configuration embedded by `HexagonTargetBackend`.

This is the key relationship:

- `HexagonTargetBackend` embeds `#iree_hexagon.hexagon_encoding_resolver<>`
  into the executable target configuration.
- encoding materialization later reads that attribute through the external
  models in `CodeGen/Encoding/`.

All of this was originally intended to reuse LLVMCPU's lowerings, particularly around `mmt4d`
and the associated data layout reorderings (`pack`/`unpack` operations).
Nevertheless, this is no longer used. Given the Hexagon hardware architecture,
the value of these reorderings is questionable and these layout rearrangements are expected
to be unused in the current lowering pipelines (`--iree-opt-data-tiling=false`).

In the future, it is likely that this folder will be removed altogether. Nevertheless, it
currently is a good skeleton for adding new dialects, if necessary. Note that the Hexagon
plugin currently does not declare any plugin owned dialects that would fit under a `IR/`
equivalent of more mature plugins.

### `CodeGen/Pipelines/`

This directory is about pipeline construction, not individual transformations.

IREE's HAL expects three main pipelines that will be called in order:

- `ConfigurationPipeline.cpp`
- `TranslationPipeline.cpp`
- `LinkingPipeline.cpp`

Additionally, this folder contains some additional helper functionality:

- `Internal.h`
  - internal helpers and flags shared only by pipeline implementation files.
- `Bufferization.cpp`
  - Hexagon-specific bufferization helper used by pipelines.

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
This pattern is copied from the LLVMCPU plugin, and in a *highly* experimental state.

- `HexagonSelectLoweringStrategy.cpp`
  - pass wrapper that drives strategy selection.
- `KernelDispatch.*`
  - actual launch-config and lowering-config policy logic.

### `CodeGen/Conversion/` and `CodeGen/Transforms/`

These directories contain the actual Hexagon-specific passes used by the
pipelines.

- `Conversion/`
  - module-level conversions and import-marking passes,
  - includes final conversion toward LLVM-oriented IR.
- `Transforms/`
  - smaller, local canonicalization or adaptation passes used inside the
    Hexagon lowering flow.
