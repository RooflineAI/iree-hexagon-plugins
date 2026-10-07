# `Target/`: the Hexagon HAL target

This folder plugs the Hexagon backend into IREE's HAL. It does four jobs:

- registers the plugin;
- describes the executable target;
- hooks the `CodeGen/` pipelines into the HAL;
- serializes each linked executable into a Hexagon shared object inside a
  flatbuffer.

The `Target` layer depends on `CodeGen`, never the other way around.

`Target/` has **no direct dependency on IREE's LLVMCPU target plugin**
(`compiler/plugins/target/LLVMCPU`), nor on `Codegen/LLVMCPU`.  Some pieces have
some overlap with LLVMCPU, see [Possible upstream
refactor](#possible-upstream-refactor) for ideas/a plan of what can be factored.

## Control flow

```
HexagonSession                  plugin "hexagon": dialects, passes, options
 ├─ HexagonTargetDevice         device "hexagon" → default executable targets
 └─ HexagonTargetBackend        backend "hexagon"
     ├─ getExecutableTarget     #hal.executable.target<"hexagon", "embedded-elf-hexagon", {config}>
     │    └─ HexagonTarget::storeToConfigAttrs
     ├─ build{Configuration,Translation,Linking}PassPipeline  → CodeGen/Pipelines
     └─ serializeExecutable → serializeHexagonExecutable
          1. translate the LLVM dialect module to llvm::Module
          2. LibraryBuilder: emit iree_hal_executable_library_query
          3. createHexagonTargetMachine(config)
          4. runLLVMOptimizationPasses (O2)
          5. emit object file (+ per-frame stack-size check)
          6. collect hal.executable.objects
          7. linking::linkHexagonSharedObject → .so
          8. wrap export names + .so in the hexagon ExecutableDef flatbuffer
```

`HexagonTarget::storeToConfigAttrs` writes the configuration dictionary. It is
the **interface between this folder and `CodeGen/`**.

## Files

| file | contents |
|---|---|
| `HexagonOptions.*` | `--iree-hexagon-v`, `--iree-hexagon-features`, `--iree-hexagon-linker-path`. Shared by the device and the backend. |
| `HexagonSession.*` | `PluginSession`: registers the dialects, the external models (bufferization, encoding) and the CodeGen passes, then adds the `hexagon` target device and target backend. |
| `HexagonTargetDevice.*` | `HAL::TargetDevice`: builds the default `#hal.device.target` from the backend's executable targets, and matches devices by `hexagon*`. |
| `HexagonTargetBackend.*` | `HAL::TargetBackend`: builds the executable target attribute, declares the dependent dialects, and forwards to the CodeGen pipelines and to serialization. |
| `HexagonLLVMTarget.*` | `HexagonTarget` (triple, DSP version, DSP features, data layout, vector width, stack limit) and its config-dict writer; `createHexagonTargetMachine`; `initializeHexagonTarget`. The DSP version and features are stored under the generic `cpu` / `cpu_features` keys, which `createHexagonTargetMachine` reads back. The vector width follows the HVX length feature (64 or 128 bytes). |
| `HexagonExecutableSerialization.*` | `serializeHexagonExecutable` and its helpers: metadata, the LLVM optimization pipeline, object emission, temporary files, linking, the flatbuffer, and the `--iree-hexagon-fail-on-stack-frames-larger-than` check. |
| `LibraryBuilder.*` | Vendored copy of LLVMCPU's `LibraryBuilder`. It emits the `iree_hal_executable_library_v0_t` tables and the query function. |
| `Linking/HexagonLinker.*` | `linkHexagonSharedObject`: finds the linker and runs it with the Hexagon flags. |

## Serialization details

- **Library metadata.**
  - `buildExecutableMetadata` declares every `hal.executable.export` in the
    `LibraryBuilder` and builds `iree_hal_executable_library_query`. The query
    function is the only symbol the DSP runtime looks up
    (`plugins/runtime/hexagon/dsp/executable.c`). It has external linkage and
    default visibility.
  - Reflection attributes are always included.
  - Sanitizers are not supported.
  - The export order is also written into the flatbuffer's `entry_points`, so
    that index equals the dispatch ordinal on both host and DSP.
- **Target machine.**
  - Created from the config with PIC relocation, `CodeGenOptLevel::Aggressive`
    and hard float ABI.
  - Function and data sections are on, as in LLVMCPU, so the linker can
    drop unused code.
- **Optimization pipeline.**
  - `buildPerModuleDefaultPipeline(O2)`.
  - Loop interleaving, loop vectorization, loop unrolling and SLP
    vectorization are all disabled. These are the defaults LLVMCPU used.
  - The module is verified afterwards.
  - LLVM's own debugging flags (`-print-after-all`, …) are honoured through
    `StandardInstrumentations`.
- **Stack frames.**
  - Every defined function gets `warn-stack-size`.
  - A diagnostic handler turns LLVM's post-register-allocation reports into
    an MLIR error listing the offending functions.
  - This is a per-frame check only.
- **Temporary files.**
  - The object file, inline `hal.executable.objects` data and the linker
    output are written to temporary files.
  - Each file is owned by an `llvm::FileRemover` and deleted when
    serialization returns, whether it succeeded or failed.
  - Use `--iree-hal-dump-executable-binaries-to` to keep the `.o`, `.s` and
    `.so` files.
- **Dumps.**
  - With `--iree-hal-dump-executable-intermediates-to`:
    `*.codegen.mlir`, `*.ll`/`*.bc` (before optimization) and
    `*.opt.ll`/`*.opt.bc` (after).
  - With `--iree-hal-dump-executable-binaries-to`: `*.s`, `*.o` and `*.so`.

## Linking

`linkHexagonSharedObject` picks the linker in this order:

1. `--iree-hexagon-linker-path`
2. `IREE_HEXAGON_LINKER_PATH`
3. `hexagon-clang`, `lld` or `ld.lld` found by `findTool`: the compiler's
   install and build directories, then `PATH`.

A bare tool name is resolved through `PATH`. The linker is executed directly
with `llvm::sys::ExecuteAndWait`, not through a shell. The full command line is
printed on failure and under `--debug-only=iree-hexagon-linker`.

- `hexagon-clang` only receives `-shared`, because the SDK driver adds the
  rest.
- For `lld`, the flags reproduce what the SDK linker emits. Each flag is
  annotated in `HexagonLinker.cpp` as either required by the DSP loader or
  kept to match the SDK output.
- Variants tagged for native runtime linking
  (`codegen::kNativeRuntimeLinkVariantAttrName`) keep undefined symbols. The
  DSP loader binds them against `libhexagon_dsp_skel.so`.

## Possible upstream refactor

**Move `LibraryBuilder` to a common location.** For example
`compiler/src/iree/compiler/Dialect/HAL/Utils/`, next to `LLVMLinkerUtils`, with
its own build target depending only on LLVM Core, Support and TargetParser, and
MLIR Support. Its compiler-side counterpart, `Codegen/LLVMCPU/DispatchABI`, is
likewise copied into `CodeGen/Conversion/HexagonDispatchABI.*`, and should be
considered at the same time.

The copy is kept as close to upstream as possible, so it can be diffed and
eventually deleted. Only these things differ:

- the namespace, `hexagon::target`, which avoids ODR clashes with the LLVMCPU
  copy linked into the same `iree-compile`;
- the header guard;
- the stale includes (`LLVMTargetOptions.h`, `UtilTypes.h`), replaced by
  `mlir/Support/LLVM.h`.

The header of `LibraryBuilder.h` records the IREE revision and ABI version it
was copied at. `HexagonExecutableSerialization.cpp` has a `static_assert`
comparing `LibraryBuilder::Version::LATEST` with the runtime's
`IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST`. An IREE bump that changes the
executable library ABI therefore breaks the build instead of failing on the
DSP. When that happens, re-sync the copy from upstream and update the header.
