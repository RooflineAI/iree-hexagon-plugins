# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Finding models in the tree."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from integration_tests.modeltree.spec import ModelSpec, load_model_spec

# The data tree lives beside this package.
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def model_directories(root: Path | None = None) -> list[Path]:
    """Every directory in the tree holding a `model.yaml`, sorted."""
    root = MODELS_DIR if root is None else root
    if not root.exists():
        return []
    return sorted(path.parent for path in root.rglob("model.yaml"))


def load_all(root: Path | None = None) -> list[ModelSpec]:
    root = MODELS_DIR if root is None else root
    return [load_model_spec(directory, root) for directory in model_directories(root)]


def select(
    names: Sequence[str] | None = None, root: Path | None = None
) -> list[ModelSpec]:
    """The models to run: named explicitly, or the whole tree."""
    specs = load_all(root)
    if not names:
        return specs
    wanted = list(names)
    by_name = {spec.name: spec for spec in specs}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise LookupError(
            f"no model named {missing} in {MODELS_DIR if root is None else root}; "
            f"known: {sorted(by_name)}"
        )
    return [by_name[name] for name in wanted]
