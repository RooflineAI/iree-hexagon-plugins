# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared helpers for submodule initialization.

IREE vendors its own copies of llvm-project, stablehlo, and torch-mlir. This
repo supersedes all three with its own top-level third-party/ copies:

  - llvm-raw and torch-mlir-raw are created in MODULE.bazel from
    third-party/{llvm-project,torch-mlir} and fed to IREE's module extension
    via inject_repo().
  - stablehlo is repointed at third-party/stablehlo by
    patches/iree/iree_extensions_embedded_paths.patch.

IREE's *other* submodules (googletest, flatcc, benchmark, spirv_cross,
vulkan_headers, webgpu-headers, tracy, musl, hip-build-deps,
hsa-runtime-headers) are still resolved out of IREE's own checkout, so those do
need to be initialized.
"""

import re
import subprocess
from pathlib import Path

SM_EXCLUDE_PATTERN = re.compile(r"third_party/(llvm-project|torch-mlir|stablehlo)")


def iree_submodule_paths(iree_dir: Path) -> list[str]:
    """List paths of IREE's submodules (recursive), minus the excluded three.

    `git submodule status --recursive` lines look like
    "<flag><sha> <path> [(<describe>)]"; the second whitespace-delimited field
    is the submodule path.
    """
    result = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=iree_dir,
        check=True,
        text=True,
        capture_output=True,
    )
    paths = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        path = parts[1]
        if SM_EXCLUDE_PATTERN.search(path):
            continue
        paths.append(path)
    return paths
