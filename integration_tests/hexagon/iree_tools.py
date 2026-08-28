# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Locating the Bazel-built artifacts these tests drive.

Under `bazel test` the paths arrive as command-line options filled in by
`$(location ...)` in the BUILD file, and this module does nothing. Run under a
bare `pytest` - which is the fast local loop - nothing fills them in, so the
paths are resolved with `bazel cquery` instead. That fallback is deliberately
skipped when running inside Bazel (TEST_TMPDIR set), because a nested Bazel
invocation inside a test action deadlocks on the output-base lock.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

IREE_COMPILE_TARGET = "@iree//tools:iree-compile"
RUNTIME_ZIP_TARGET = "//plugins/runtime/hexagon:hexagon_runtime_aarch64_android"
LLD_TARGET = "@llvm-project//lld:ld.lld"

_REPO_ROOT = Path(__file__).resolve().parents[2]


class ToolResolutionError(RuntimeError):
    pass


def running_under_bazel() -> bool:
    return "TEST_TMPDIR" in os.environ


def resolve_bazel_artifact(target: str, explicit: str | None) -> Path:
    """Return the built file for `target`, preferring an explicitly given path."""
    if explicit:
        path = Path(explicit).resolve()
        if not path.exists():
            raise ToolResolutionError(f"{target}: given path does not exist: {path}")
        return path
    if running_under_bazel():
        raise ToolResolutionError(
            f"{target} was not passed on the command line. Under Bazel the path "
            f"must come from $(location {target}) in the BUILD file; querying "
            "Bazel from inside a test action would deadlock."
        )
    if shutil.which("bazel") is None:
        raise ToolResolutionError(
            f"{target}: no path given and bazel is not on PATH. Either build the "
            "target and pass its path, or run these tests through Bazel."
        )
    query = subprocess.run(
        ["bazel", "cquery", "--output=files", target],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if query.returncode != 0:
        raise ToolResolutionError(
            f"'bazel cquery --output=files {target}' failed:\n{query.stderr}"
        )
    files = [line for line in query.stdout.splitlines() if line.strip()]
    if not files:
        raise ToolResolutionError(
            f"'bazel cquery --output=files {target}' produced no output. For the "
            "Hexagon runtime this usually means the Android NDK was not found, "
            "in which case the target is @platforms//:incompatible - see "
            "ANDROID_NDK_PATH.md."
        )
    path = (_REPO_ROOT / files[0]).resolve()
    if not path.exists():
        raise ToolResolutionError(
            f"{target} resolves to {path}, which does not exist. Run "
            f"'bazel build {target}' first."
        )
    return path
