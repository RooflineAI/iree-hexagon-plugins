# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""What one import produces, described without importing torch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportedModel:
    """An imported model and the data needed to run and check it."""

    model_mlir: Path
    input_files: list[Path]
    output_files: list[Path]
    function: str = "main"
    # Present only when the model spec asked for externalized weights.
    parameter_file: Path | None = None
    # Present only when the model carries a config.id2label, i.e. classifiers.
    labels_file: Path | None = None

    @property
    def output_names(self) -> list[str]:
        return [path.name for path in self.output_files]

    def labels(self) -> dict[int, str] | None:
        if self.labels_file is None or not self.labels_file.exists():
            return None
        return {
            int(key): str(value)
            for key, value in json.loads(self.labels_file.read_text()).items()
        }

    @classmethod
    def from_directory(cls, directory: Path, function: str = "main") -> ImportedModel:
        """Re-open an already-imported model, without needing torch."""
        parameters = directory / "weights.irpa"
        labels = directory / "labels.json"
        return cls(
            model_mlir=directory / "model.mlir",
            input_files=sorted(directory.glob("input*.npy")),
            output_files=sorted(directory.glob("output*.npy")),
            function=function,
            parameter_file=parameters if parameters.exists() else None,
            labels_file=labels if labels.exists() else None,
        )
