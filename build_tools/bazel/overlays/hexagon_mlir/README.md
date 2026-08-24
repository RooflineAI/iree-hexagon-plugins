# Hexagon-MLIR Bazel Overlay

This folder defines the Bazel overlay used to expose `third-party/hexagon-mlir`
as a first-class external repository (`@hexagon-mlir`).

## Why this exists

- Upstream `hexagon-mlir` is CMake-first while this repo is Bazel-first.
- Ideally want Bazel targets without forking/editing upstream sources directly.
- An overlay allows keeping build logic here while reusing upstream code.

## Repository flow

1. `MODULE.bazel` creates `@hexagon-mlir-raw` as a
   `new_local_repository` pointing at `third-party/hexagon-mlir`.
2. `MODULE.bazel` calls `hexagon_mlir_configure(name = "hexagon-mlir")`.
3. `build_tools/bazel/hexagon_mlir_configure.bzl` creates a generated external
   repo named `@hexagon-mlir`.
4. That generated repo is a symlinked view of:
   - upstream source tree (`@hexagon-mlir-raw`)
   - this overlay tree (`build_tools/bazel/overlays/hexagon_mlir`)

Overlay path precedence is "overlay wins":
- If a file exists in overlay and upstream at the same relative path, the
  overlay file is used.
- If an overlay directory exists, upstream entries under that directory are only
  linked when they are not overridden by overlay entries.

## Compatibility notes

- Overlaid `CopyOpInterface` implementation:
  - The LLVM snapshot used here does not provide
    `mlir/Interfaces/CopyOpInterface.td` and
    `mlir/Interfaces/CopyOpInterface.h`.
    These were deprecated and removed long ago.
  - Overlay provides a copy of the deprecated 
    code at:
    `qcom_hexagon_backend/include/mlir/Interfaces/`
- Upstream warning suppressions carried in the overlay:
  - Current LLVM emits deprecation warnings for several
    `hexagon-mlir` libraries because upstream uses deprecated
    `IRRewriter::create` APIs.
  - Overlay silences `-Wdeprecated-declarations` locally in:
    `qcom_hexagon_backend/lib/Common`,
    `qcom_hexagon_backend/lib/Transforms`, and the
    `qcom_hexagon_backend/lib/Conversion/*ToLLVM` libraries.
  - Overlay also silences `-Wunused-but-set-variable` locally in
    `qcom_hexagon_backend/lib/Transforms`.

## Dependency model

- Source-bearing overlay libraries use
  `//build_tools/bazel:hexagon_mlir_overlay_library`.
- That macro separates:
  - `deps`: real implementation dependencies that should participate in linking
  - `hdr_deps`: compile-time-only LLVM/MLIR/IREE dependencies wrapped with
    `cc_headers_only`
- The Hexagon compiler plugin should reuse common
  LLVM/MLIR/IREE implementation already provided by
  `libIREECompilerUnshielded.so` instead of re-linking those libraries into the
  plugin `.so`.
- When extending the overlay, prefer putting generic LLVM/MLIR libraries in
  `hdr_deps` and reserve `deps` for:
  - other `@hexagon-mlir//qcom_hexagon_backend/...` implementation libraries
  - Hexagon-specific implementation that must actually link
  - generated code targets such as `gentbl_cc_library` outputs when they are
    consumed as sources/providers and not just as forwarded headers

## Platform split

- Most overlay targets under `qcom_hexagon_backend/include` and
  `qcom_hexagon_backend/lib` are compiler-side MLIR/LLVM libraries. They are
  expected to build for the host toolchain, not for the Hexagon/QURT device
  toolchain.
- Only device/runtime pieces such as
  `@hexagon-mlir//qcom_hexagon_backend/bin/runtime:runtime_lib` are meant to be
  built with `--platforms=//platform:hexagon`.

## How to extend this overlay

1. Add mirrored `BUILD.bazel` files under this directory for new upstream
   subpaths.
2. Export aggregated targets from `qcom_hexagon_backend/BUILD.bazel` when you
   need stable top-level labels.
