# Dependency patches

Submodule patches are applied to working trees by
[`apply_submodule_patches.sh`](../build_tools/apply_submodule_patches.sh), in
filename order within each folder. Keep changes here rather than committing
inside `third-party/`.

- [IREE](iree/README.md): embedded Bazel builds, Hexagon integration and
  APIs needed by the plugin.
- [LLVM](llvm-project/README.md): no active source patches after the upgrade.

hexagon-mlir is not patched here. Its changes (LLVM compatibility, conversion
APIs reused by the plugin, and runtime/compiler fixes) are carried as commits on
a branch of the [iree-org/hexagon-mlir](https://github.com/iree-org/hexagon-mlir)
fork; see [hexagon-mlir fork](#hexagon-mlir-fork) below.

Patches directly in this folder are applied by Bazel module overrides in
[`MODULE.bazel`](../MODULE.bazel), rather than the submodule patch helper.

| Patch | Category | Reason |
| --- | --- | --- |
| [hedron_skip_pch.patch](hedron_skip_pch.patch) | Tooling compatibility | Skip header-only `-xc++-header` actions that otherwise make the compilation-database extractor fail its source-file assertion. |
| [llvm_libstdcxx_no_static_runtime.patch](llvm_libstdcxx_no_static_runtime.patch) | Host toolchain / ABI compatibility | Support dynamic libstdc++ without requiring a static archive. Stage linker names and a link-time unwinder stand-in so built libraries use the system `libgcc_s.so.1`, avoiding competing unwinders with PyTorch/ONNX. |

The obsolete `llvm_rules_cc_use_libtool_rename.patch` was removed because the
updated hermetic LLVM toolchain already uses the current rules_cc setting.

## hexagon-mlir fork

`third-party/hexagon-mlir` is pinned to a commit on a
`sm-iree-integrates/hexagon-mlir-YYYYMMDD` branch of
[iree-org/hexagon-mlir](https://github.com/iree-org/hexagon-mlir), following
the scheme IREE uses for its LLVM fork. Each branch is an upstream
[qualcomm/hexagon-mlir](https://github.com/qualcomm/hexagon-mlir) commit plus
the commits this repo needs on top of it; the fork's `main` is not used.

- List the carried commits of the current pin with
  `git log $(git merge-base HEAD upstream/main)..HEAD` inside the submodule
  (with `upstream` pointing at qualcomm/hexagon-mlir).
- To change them or move to a newer upstream commit, create a new dated branch
  from the current pin, `git rebase` it onto the new upstream commit, and drop
  carried commits that became empty or obsolete.
- Never force-push or delete a pushed `sm-*` branch: the pinned commit is only
  reachable through it, and older revisions of this repo would no longer be able
  to fetch their submodule.
- The fork branch is not reviewed separately. Review happens on the pull request
  here that bumps the submodule; list the carried commits that were added,
  dropped or changed in its description and link the compare view
  (`qualcomm/hexagon-mlir/compare/<upstream base>...iree-org:hexagon-mlir:<branch>`).
