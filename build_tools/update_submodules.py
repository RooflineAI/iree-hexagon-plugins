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
    superseded by this repo's own top-level copies (see _submodule_utils.py).
    Bazel consumes those copies directly, while CMake consumes them through
    symlinks at the nested paths IREE expects. This avoids cloning a second
    copy of the LLVM monorepo.
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
SUPERSEDED_IREE_SUBMODULES = ("llvm-project", "stablehlo", "torch-mlir")


def run(cmd: list[str], *, cwd: Path) -> None:
    """Run a command, streaming its output so long clones show progress."""
    subprocess.run(cmd, cwd=cwd, check=True, text=True)


def git_output(args: list[str], *, cwd: Path) -> str:
    """Run git and return its stripped stdout."""
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True
    ).strip()


def set_up_iree_dependency_links() -> None:
    """Point IREE's superseded nested submodules at the top-level copies."""
    print("==> Linking IREE dependencies to top-level submodules", flush=True)
    for name in SUPERSEDED_IREE_SUBMODULES:
        top_level_path = REPO_ROOT / "third-party" / name
        nested_path = IREE_DIR / "third_party" / name

        top_level_pin = git_output(
            ["rev-parse", f"HEAD:third-party/{name}"], cwd=REPO_ROOT
        )
        nested_pin = git_output(
            ["rev-parse", f"HEAD:third_party/{name}"], cwd=IREE_DIR
        )
        if top_level_pin != nested_pin:
            raise RuntimeError(
                f"cannot share {name}: top-level pin {top_level_pin} does not "
                f"match IREE's nested pin {nested_pin}"
            )

        checked_out_pin = git_output(["rev-parse", "HEAD"], cwd=top_level_path)
        if checked_out_pin != top_level_pin:
            raise RuntimeError(
                f"top-level submodule third-party/{name} is at "
                f"{checked_out_pin}, expected {top_level_pin}"
            )

        if nested_path.is_symlink():
            if nested_path.resolve() != top_level_path.resolve():
                raise RuntimeError(
                    f"{nested_path.relative_to(REPO_ROOT)} points to "
                    f"{nested_path.resolve()}, expected {top_level_path.resolve()}"
                )
            print(f"already linked: {nested_path.relative_to(REPO_ROOT)}")
            continue

        if nested_path.exists():
            try:
                nested_checkout_pin = git_output(
                    ["rev-parse", "HEAD"], cwd=nested_path
                )
            except subprocess.CalledProcessError:
                nested_checkout_pin = ""
            if nested_checkout_pin == nested_pin:
                print(
                    f"keep existing checkout: {nested_path.relative_to(REPO_ROOT)}"
                )
                continue
            try:
                nested_path.rmdir()
            except OSError as exc:
                raise RuntimeError(
                    f"cannot replace non-empty {nested_path.relative_to(REPO_ROOT)} "
                    "with a link to the top-level submodule"
                ) from exc

        relative_target = Path("../..") / name
        nested_path.symlink_to(relative_target, target_is_directory=True)
        print(
            f"linked: {nested_path.relative_to(REPO_ROOT)} -> {relative_target}"
        )


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

    set_up_iree_dependency_links()

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
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
