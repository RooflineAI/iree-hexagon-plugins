# Hexagon-MLIR Bazel Overlay

This folder defines the Bazel overlay used to expose `third-party/hexagon-mlir`
as a first-class external repository (`@hexagon-mlir`) in roof-mlir.

## Why this exists

- Upstream `hexagon-mlir` is CMake-first while roof-mlir is Bazel-first.
- Ideally want Bazel targets without forking/editing upstream sources directly.
- An overlay allows to keep build logic in roof-mlir while reusing upstream code.

## Repository flow

1. `build_tools/bazel/extensions.bzl` creates `@hexagon-mlir-raw` as a
   `new_local_repository` pointing at `third-party/hexagon-mlir`.
2. `MODULE.bazel` calls `hexagon_mlir_configure(name = "hexagon-mlir")`.
3. `build_tools/bazel/hexagon_mlir_configure.bzl` creates a generated external
   repo named `@hexagon-mlir`.
4. That generated repo is a symlinked view of:
   - upstream source tree (`@hexagon-mlir-raw`)
   - this overlay tree
     (`plugins/external/cellar-hexagon/build_tools/bazel/overlays/hexagon_mlir`)

Overlay path precedence is "overlay wins":
- If a file exists in overlay and upstream at the same relative path, the
  overlay file is used.
- If an overlay directory exists, upstream entries under that directory are only
  linked when they are not overridden by overlay entries.

## Compatibility notes

- Overlaid `CopyOpInterface` implementation:
  - LLVM snapshot used in roof-mlir does not provide
    `mlir/Interfaces/CopyOpInterface.td` and
    `mlir/Interfaces/CopyOpInterface.h`.
    These were deprecated and removed long ago.
  - Overlay provides a copy of the deprecated 
    code at:
    `qcom_hexagon_backend/include/mlir/Interfaces/`
- Deprecated API warnings in upstream `DMATransferUtil.cpp`:
  - Current snapshot emits deprecation warnings that may be promoted to errors.
  - Overlay uses a local `copts` relaxation on `HexagonCommon` to keep the
    integration buildable without patching upstream source.

## How to extend this overlay

1. Add mirrored `BUILD.bazel` files under this directory for new upstream
   subpaths.
2. Export aggregated targets from `qcom_hexagon_backend/BUILD.bazel` when you
   need stable top-level labels.
