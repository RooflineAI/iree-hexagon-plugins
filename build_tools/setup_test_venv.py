#!/usr/bin/env python3
# Copyright 2026 RooflineAI GmbH
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Install the Bazel compiler and locked external packages into a test venv."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPILER_PROVENANCE = "iree/compiler/_build_provenance.json"


def external_requirements(requirements: str) -> str:
    """Retain the Bazel lock, but provide its compiler locally in this venv.

    Install with --no-deps: resolving Turbine's transitive dependencies would
    otherwise restore the compiler wheel that this workflow replaces.
    """
    return "".join(
        line
        for line in requirements.splitlines(keepends=True)
        if not re.match(r"^iree[-_]base[-_]compiler\s*==", line, flags=re.IGNORECASE)
    )


def read_provenance(package: Path | None, wheel: Path | None) -> dict:
    if package is not None:
        content = (package / COMPILER_PROVENANCE).read_text()
    else:
        with zipfile.ZipFile(wheel) as archive:
            content = archive.read(COMPILER_PROVENANCE).decode()
    provenance = json.loads(content)
    if provenance.get("schema_version") != 1 or not re.fullmatch(
        r"[0-9a-f]{64}", provenance.get("build_id", "")
    ):
        raise ValueError("compiler artifact has invalid build provenance")
    return provenance


def prepare_editable(venv: Path, package: Path) -> Path:
    """Keep editable build metadata writable without modifying Bazel outputs."""
    editable = venv / ".iree-compiler-source"
    editable.mkdir(exist_ok=True)
    shutil.copyfile(
        REPO_ROOT / "build_tools/python_compiler/pyproject.toml",
        editable / "pyproject.toml",
    )
    source_link = editable / "iree"
    if source_link.is_symlink():
        source_link.unlink()
    elif source_link.exists():
        raise ValueError(f"unexpected directory at compiler source link: {source_link}")
    source_link.symlink_to(package / "iree", target_is_directory=True)
    return editable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    artifact = parser.add_mutually_exclusive_group(required=True)
    artifact.add_argument("--compiler-package", type=Path)
    artifact.add_argument("--compiler-wheel", type=Path)
    args = parser.parse_args()
    venv = args.venv.resolve()
    package = args.compiler_package.resolve() if args.compiler_package else None
    wheel = args.compiler_wheel.resolve() if args.compiler_wheel else None
    provenance = read_provenance(package, wheel)
    python = venv / "bin/python"
    if not python.is_file():
        subprocess.run(["uv", "venv", "--python", "3.11", str(venv)], check=True)
    subprocess.run(
        [
            str(python),
            "-c",
            "import sys; assert sys.version_info[:2] == (3, 11), "
            "'the compiler package requires Python 3.11; use a Python 3.11 venv'",
        ],
        check=True,
    )

    def run_uv(command: str, *options: str) -> None:
        subprocess.run(
            ["uv", "pip", command, "--python", str(python), *options], check=True
        )

    requirements = external_requirements(
        (REPO_ROOT / "build_tools/bazel/requirements.txt").read_text()
    )
    with tempfile.TemporaryDirectory(prefix="iree-test-dependencies-") as directory:
        lock = Path(directory) / "requirements.txt"
        lock.write_text(requirements)
        run_uv(
            "install",
            "--no-deps",
            "--index-strategy",
            "unsafe-best-match",
            "--requirements",
            str(lock),
        )
    install_options = ["--no-deps", "--reinstall-package", "iree-base-compiler"]
    if package:
        editable = prepare_editable(venv, package)
        install_options += ["--no-build-isolation", "--editable", str(editable)]
    else:
        install_options += [str(wheel)]
    run_uv("install", *install_options)
    run_uv("check")
    verification = [
        str(python),
        str(REPO_ROOT / "build_tools/verify_compiler_package.py"),
        "--build-id",
        provenance["build_id"],
    ]
    if package:
        verification += ["--package", str(package)]
    subprocess.run(verification, check=True)


if __name__ == "__main__":
    main()
