#!/usr/bin/env python3
# Copyright 2026 RooflineAI GmbH
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check the installed compiler's Python modules and native library together."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path


def verify(build_id: str, package: Path | None = None) -> dict:
    import iree.compiler
    from iree.compiler.api import ctypes_dl
    from iree.compiler.extras import fx_as_strided, fx_importer

    compiler_root = Path(iree.compiler.__file__).resolve().parent
    if package is not None and compiler_root != package.resolve() / "iree/compiler":
        raise ValueError(
            f"compiler was imported from a different build: {compiler_root}"
        )
    provenance = json.loads((compiler_root / "_build_provenance.json").read_text())
    if provenance["build_id"] != build_id:
        raise ValueError("installed compiler provenance differs from selected artifact")
    for module in (fx_importer, fx_as_strided):
        if Path(module.__file__).resolve().parent != compiler_root / "extras":
            raise ValueError(
                f"importer helper was loaded from another package: {module}"
            )
    library = Path(ctypes_dl._dylib._name).resolve()
    if library.parent != compiler_root / "_mlir_libs":
        raise ValueError(f"compiler API loaded an unexpected library: {library}")
    # Check that all packaged native bindings and dialect modules can load.
    for extension in sorted((compiler_root / "_mlir_libs").glob("_*.so")):
        importlib.import_module("iree.compiler._mlir_libs." + extension.stem)
    for source in sorted((compiler_root / "dialects").rglob("*.py")):
        parts = source.relative_to(compiler_root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        importlib.import_module("iree.compiler." + ".".join(parts))
    providers = [
        d
        for d in importlib.metadata.distributions()
        if d.metadata["Name"].lower().replace("_", "-") == "iree-base-compiler"
    ]
    if len(providers) != 1:
        raise ValueError(f"expected one compiler distribution, found {len(providers)}")
    result = {
        "distribution": "iree-base-compiler",
        "version": providers[0].version,
        "build_id": build_id,
        "compiler": str(compiler_root),
        "fx_importer": fx_importer.__file__,
        "library": str(library),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--package", type=Path)
    args = parser.parse_args()
    verify(args.build_id, args.package)


if __name__ == "__main__":
    main()
