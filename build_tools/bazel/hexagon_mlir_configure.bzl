# Copyright 2026 RooflineAI GmbH
# SPDX-License-Identifier: Apache-2.0

"""Repository rule for materializing the `@hexagon-mlir` Bazel overlay.

This rule creates a generated external repository whose root contains symlinks
to:
1) files from `third-party/hexagon-mlir` (`@hexagon-mlir-raw`), and
2) overlay files from `build_tools/bazel/overlays/hexagon_mlir`.

Overlay files/directories take precedence over source files with the same path.

The overlay behavior tries to follow the pattern from LLVM/torch-mlir's own
Bazel overlays. This file closely mimics llvm's utils/bazel/configure.bzl.

This file intentionally remains root-owned rather than loaded via
`use_repo_rule()` from another repository: doing so would create a circular
repository definition during MODULE evaluation, since both a local-repository
alias and the repository rule would participate in the same `_repo_rules`
bootstrap flow.
"""

MAX_TRAVERSAL_STEPS = 1000000  # upper bound on visited dirs.

def _overlay_directories(repository_ctx, src_root, overlay_root):
    """Symlinks source + overlay trees into the generated repository.

    All symlinks are created under `target_root` (the root of the repository
    produced by this rule, i.e. `@hexagon-mlir`), not in the workspace.
    """
    target_root = repository_ctx.path(".")

    # Objective here:
    # - symlink all overlay files
    # - for overlaid directories, symlink non-overridden source entries
    stack = ["."]
    for _ in range(MAX_TRAVERSAL_STEPS):
        rel_dir = stack.pop()
        overlay_dirs = {}
        overlay_files = {}

        for entry in overlay_root.get_child(rel_dir).readdir():
            name = entry.basename
            full_rel_path = rel_dir + "/" + name
            if entry.is_dir:
                stack.append(full_rel_path)
                overlay_dirs[name] = None
            else:
                overlay_files[name] = None
                repository_ctx.symlink(
                    overlay_root.get_child(full_rel_path),
                    target_root.get_child(full_rel_path),
                )

        src_dir = src_root.get_child(rel_dir)

        # Tolerate directories that only exist in the overlay.
        # This is useful to add compatibility files if needed.
        if src_dir.exists and src_dir.is_dir:
            for src_entry in src_dir.readdir():
                name = src_entry.basename
                if name in overlay_dirs.keys() or name in overlay_files.keys():
                    continue
                repository_ctx.symlink(
                    src_entry,
                    target_root.get_child(rel_dir + "/" + name),
                )

        if not stack:
            return

    fail("overlay_directories: exceeded MAX_TRAVERSAL_STEPS ({}).".format(MAX_TRAVERSAL_STEPS))

def _hexagon_mlir_configure_impl(repository_ctx):
    """Materializes the configured `@hexagon-mlir` repository."""
    src_root = repository_ctx.path(repository_ctx.attr.src_root_marker).dirname

    # Intentionally do NOT resolve overlay_root from a label marker under the
    # overlay directory itself.
    #
    # The overlay tree's own BUILD.bazel files are meant to be consumed via
    # @hexagon-mlir only (not built directly as part of //...), so if we used
    # a label there, label resolution could fail depending on ignore/exclude
    # configuration.
    workspace_root = repository_ctx.path(repository_ctx.attr.workspace_root_marker).dirname
    overlay_root = workspace_root.get_child(repository_ctx.attr.overlay_relpath)
    if not overlay_root.exists:
        fail("hexagon_mlir_configure: overlay root '{}' does not exist. If you modified the overlay path, please update this file.".format(overlay_root))
    _overlay_directories(repository_ctx, src_root, overlay_root)

hexagon_mlir_configure = repository_rule(
    implementation = _hexagon_mlir_configure_impl,
    local = True,
    configure = True,
    attrs = {
        "src_root_marker": attr.label(
            default = Label("@hexagon-mlir-raw//:README.md"),
            doc = "Label pointing to a file in hexagon-mlir source root",
        ),
        "workspace_root_marker": attr.label(
            default = Label("@iree_hexagon_plugins//:MODULE.bazel"),
            doc = "Label pointing to a file in this repo's own root",
        ),
        "overlay_relpath": attr.string(
            default = "build_tools/bazel/overlays/hexagon_mlir",
            doc = "Overlay directory path relative to this repo's own root",
        ),
    },
)
