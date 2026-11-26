# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""What one import produces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParameterArchive:
    """The weights a module compiled with external weights loads at runtime."""

    path: Path
    # Names the archive on the device: a changed archive is a different file.
    sha256: str
    # The scope the module looks its parameters up in, as in
    # `--parameters=<scope>=<file>`.
    scope: str

    @classmethod
    def of(cls, path: Path, scope: str) -> ParameterArchive:
        """The archive at `path`, hashed (~5 s for 9 GB)."""
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1 << 24), b""):
                digest.update(chunk)
        return cls(path, digest.hexdigest(), scope)


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
class GeneratedTokenInfo:
    """One step of `KvCacheLM.generate_greedy`."""

    # The token generated: the argmax of the step's logits.
    selected_id: int
    # The second most likely token, which a near-tie could have tipped to.
    runner_up_id: int
    # How far the selected token's logit leads the runner-up's, in float32.
    margin: float


@dataclass(frozen=True)
class GenerationReference:
    """What iree-run-llm gets, and the eager tokens it has to generate."""

    prompt_file: Path
    tokenizer_json: Path
    max_new_tokens: int
    # KvCacheLM's shape parameters, which iree-run-llm cannot read from the
    # module and has to be told.
    max_seq_len: int
    block_seq_stride: int
    min_prefill_blocks: int
    pad_token_id: int
    prompt_ids: tuple[int, ...]
    # One per generated token, in order.
    generated: tuple[GeneratedTokenInfo, ...]
    # Prefill's logits on the padded prompt, [1, padded length, vocab].
    prefill_logits_file: Path

    @property
    def tokens(self) -> tuple[int, ...]:
        """The generated token ids."""
        return tuple(info.selected_id for info in self.generated)


@dataclass(frozen=True)
class ImportedModel:
    """An imported model and the data needed to run and check it."""

    model_mlir: Path
    # One call per exported function. A plain model has one, `main`
    # (iree-turbine's default entry point name). A `KvCacheLM` has none: its
    # functions are checked by `generation` instead.
    calls: list[ModuleCall]
    # Present only when the model carries a config.id2label, i.e. classifiers.
    labels_file: Path | None = None
    # Only, and always, for a `KvCacheLM`.
    generation: GenerationReference | None = None
    # The weights, when the import kept them out of the MLIR (external
    # weights): the MLIR names them in this archive.
    weights_archive: Path | None = None

    def labels(self) -> dict[int, str] | None:
        if self.labels_file is None or not self.labels_file.exists():
            return None
        return {
            int(key): str(value)
            for key, value in json.loads(self.labels_file.read_text()).items()
        }
