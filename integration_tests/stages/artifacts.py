# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""What one import produces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModuleCall:
    """One call of an exported function: its inputs and torch reference."""

    function: str
    input_files: list[Path]
    output_files: list[Path]

    @property
    def output_names(self) -> list[str]:
        return [path.name for path in self.output_files]


@dataclass(frozen=True)
class ImportedModel:
    """An imported model and the data needed to run and check it."""

    model_mlir: Path
    # One call per exported function. A plain model has one, `main`
    # (iree-turbine's default entry point name).
    calls: list[ModuleCall]
    # Present only when the model carries a config.id2label, i.e. classifiers.
    labels_file: Path | None = None
    # A `KvCacheLM`: `calls` are `prefill` then `decode`.
    prefill_decode: bool = False

    def labels(self) -> dict[int, str] | None:
        if self.labels_file is None or not self.labels_file.exists():
            return None
        return {
            int(key): str(value)
            for key, value in json.loads(self.labels_file.read_text()).items()
        }
