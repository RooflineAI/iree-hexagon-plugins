# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Locating the four artifacts these tests drive.

The device suite runs under plain pytest, so the normal path is that each of
`iree-compile`, the runtime zip, the device-tools zip and `ld.lld` arrives as a
command-line option and this module only checks that it exists.

When an option is omitted the path is resolved with
`bazel cquery` instead.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

IREE_COMPILE_TARGET = "@iree//tools:iree-compile"
RUNTIME_ZIP_TARGET = "//plugins/runtime/hexagon:hexagon_runtime_aarch64_android"
LLD_TARGET = "@llvm-project//lld:ld.lld"
DEVICE_TOOLS_TARGET = "//integration_tests/device/tools:device_tools_aarch64_android"

_REPO_ROOT = Path(__file__).resolve().parents[1]


class ToolResolutionError(RuntimeError):
    pass


def resolve_bazel_artifact(target: str, explicit: str | None) -> Path:
    """Return the built file for `target`, preferring an explicitly given path."""
    if explicit:
        path = Path(explicit).resolve()
        if not path.exists():
            raise ToolResolutionError(f"{target}: given path does not exist: {path}")
        return path
    if shutil.which("bazel") is None:
        raise ToolResolutionError(
            f"{target}: no path given and bazel is not on PATH. Build the "
            "target and pass its path, or put bazel on PATH so it can be "
            "queried for one."
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
