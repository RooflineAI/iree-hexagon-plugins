#!/usr/bin/env python3
# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Initialize this repo's submodules and apply the required patches.

Run this after cloning, and any time a submodule pin moves:

    build_tools/update_submodules.py

Deliberately *not* a plain `git submodule update --init --recursive`:

  - IREE's nested third_party/{llvm-project,stablehlo,torch-mlir} are
    superseded by this repo's own top-level copies (see _submodule_utils.py),
    so initializing them would clone a second, unused copy of the LLVM
    monorepo - several GB that Bazel never reads.
  - third-party/hexagon-mlir carries triton/triton_shared submodules that
    nothing in this repo's build references.

Everything is therefore initialized explicitly rather than recursively.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from _submodule_utils import iree_submodule_paths

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
IREE_DIR = REPO_ROOT / "third-party" / "iree"
APPLY_PATCHES = SCRIPT_DIR / "apply_submodule_patches.sh"


def run(cmd: list[str], *, cwd: Path) -> None:
    """Run a command, streaming its output so long clones show progress."""
    subprocess.run(cmd, cwd=cwd, check=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-patches",
        action="store_true",
        help="Do not run apply_submodule_patches.sh afterwards. The Bazel build"
        " needs the patches applied, so only use this if you apply them"
        " separately.",
    )
    args = parser.parse_args()

    print("==> Initializing top-level submodules", flush=True)
    run(["git", "submodule", "update", "--init"], cwd=REPO_ROOT)

    print("==> Initializing IREE's own submodules", flush=True)
    for path in iree_submodule_paths(IREE_DIR):
        run(["git", "submodule", "update", "--init", path], cwd=IREE_DIR)

    if args.skip_patches:
        print(
            "==> Skipping patches; run"
            f" {APPLY_PATCHES.relative_to(REPO_ROOT)} before building",
            flush=True,
        )
        return 0

    # Submodule updates reset each checkout to its pinned commit, wiping any
    # previously applied patches - so this always has to run afterwards.
    print("==> Applying submodule patches", flush=True)
    run([str(APPLY_PATCHES)], cwd=REPO_ROOT)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as e:
        # Output already streamed to the terminal above; just name the command.
        print(f"ERROR: command failed: {' '.join(e.cmd)}", file=sys.stderr)
        sys.exit(e.returncode)
