# Copyright 2025 RooflineAI GmbH
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lit config for the cellar-hexagon compiler plugin tests.

iree_lit_test's `tools` list symlinks all declared tool binaries into a single
lit_bin/ directory and passes --path lit_bin to lit. That makes bare command
names in RUN: lines work. For --iree-load-plugin=cellar_hexagon=<absolute-path>
the tests need the full path to the .so, which cannot be expressed as a bare
name. This file computes that path and exports it as CELLAR_HEXAGON_COMPILER_PLUGIN.

The plugin and iree-compile are both in the tools list of every test suite, so
Bazel stages them in the same lit_bin/ directory. CELLAR_HEXAGON_COMPILER_PLUGIN
is therefore always adjacent to iree-compile, and LD_LIBRARY_PATH is extended to
that directory so the dynamic loader can find libIREECompilerUnshielded.so when
the plugin is loaded.
"""
# Lint for undefined variables is disabled as config is not defined inside this
# file, instead config is injected by way of evaluating runlit.cfg.py from
# runlit.site.cfg.py which in turn is evaluated by lit.py.
# pylint: disable=undefined-variable

import os
import tempfile

import lit.formats
import lit.util

config.name = "IREE"
config.suffixes = [".mlir", ".txt"]
config.test_format = lit.formats.ShTest(execute_external=True)
# Forward all IREE environment variables
passthrough_env_vars = ["VK_ICD_FILENAMES"]


def _tool_dir() -> str:
    """Return the lit_bin directory by locating iree-compile on the lit path."""
    extra_paths = lit_config.path
    env_path = os.environ.get("PATH", "")
    path = os.pathsep.join(extra_paths + [env_path])
    iree_compile = lit.util.which("iree-compile", path)
    return os.path.dirname(iree_compile) if iree_compile else ""


IREE_COMPILER_TOOL_DIR = _tool_dir()

# Both iree-compile and cellar_hexagon_compiler_plugin are in the same tools
# list, so Bazel symlinks them into the same lit_bin/ directory.
CELLAR_HEXAGON_COMPILER_PLUGIN = (
    os.path.join(IREE_COMPILER_TOOL_DIR, "libcellar_hexagon_compiler_plugin.so")
    if IREE_COMPILER_TOOL_DIR
    else "CELLAR_HEXAGON_COMPILER_PLUGIN_NOT_FOUND"
)

config.environment.update(
    {
        k: v
        for k, v in os.environ.items()
        if k.startswith("IREE_") or k in passthrough_env_vars
    },
)
config.environment["CELLAR_HEXAGON_COMPILER_PLUGIN"] = CELLAR_HEXAGON_COMPILER_PLUGIN
if IREE_COMPILER_TOOL_DIR:
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    config.environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [p for p in [IREE_COMPILER_TOOL_DIR, ld_library_path] if p]
    )

# Use the most preferred temp directory.
config.test_exec_root = (
    os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    or os.environ.get("TEST_TMPDIR")
    or os.path.join(tempfile.gettempdir(), "lit")
)
