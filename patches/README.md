# Dependency patches

Submodule patches are applied to working trees by
[`apply_submodule_patches.sh`](../build_tools/apply_submodule_patches.sh), in
filename order within each folder. Keep changes here rather than committing
inside `third-party/`.

- [hexagon-mlir](hexagon-mlir/README.md): LLVM compatibility, conversion APIs
  reused by the plugin, and runtime/compiler fixes.
- [IREE](iree/README.md): embedded Bazel builds, Hexagon integration and
  APIs needed by the plugin.
- [LLVM](llvm-project/README.md): no active source patches after the upgrade.

Patches directly in this folder are applied by Bazel module overrides in
[`MODULE.bazel`](../MODULE.bazel), rather than the submodule patch helper.

| Patch | Category | Reason |
| --- | --- | --- |
| [hedron_skip_pch.patch](hedron_skip_pch.patch) | Tooling compatibility | Skip header-only `-xc++-header` actions that otherwise make the compilation-database extractor fail its source-file assertion. |
| [llvm_libstdcxx_no_static_runtime.patch](llvm_libstdcxx_no_static_runtime.patch) | Host toolchain / ABI compatibility | Support dynamic libstdc++ without requiring a static archive. Stage linker names and a link-time unwinder stand-in so built libraries use the system `libgcc_s.so.1`, avoiding competing unwinders with PyTorch/ONNX. |

The obsolete `llvm_rules_cc_use_libtool_rename.patch` was removed because the
updated hermetic LLVM toolchain already uses the current rules_cc setting.
