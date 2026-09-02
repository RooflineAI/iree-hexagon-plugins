# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The model tree's schema: what a `model.yaml` says and how it is validated.

A model in this suite is two files and no test code:

    models/<org>/<name>/model.yaml   metadata - this file's schema
    models/<org>/<name>/model.py     `get_model() -> torch.nn.Module`
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import dacite
import yaml

from integration_tests.modeltree.outcomes import ExpectedOutcome, Status, parse_status

# numpy/torch dtype names accepted in `inputs[].dtype`. Closed
# set: a typo here would otherwise produce a silently different reference.
_DTYPES = ("float32", "float16", "bfloat16", "int64", "int32", "bool")

_GENERATORS = ("normal", "uniform", "ones", "zeros", "image")


class SpecError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class InputSpec:
    """One input tensor: its shape, dtype, and how it is filled."""

    shape: tuple[int, ...]
    dtype: str
    generator: str = "normal"
    seed: int = 0
    # Inclusive bounds for the integral generators. For token ids, leaving
    # `high` unset means "the model's vocab_size", resolved at import time.
    low: int = 0
    high: int | None = None
    # generator: image only - a file beside model.yaml, decoded to NCHW and
    # scaled to 0..1. A real photograph is a stronger check than any tolerance,
    # because it makes the *label* assertable without a reference tensor.
    image: str | None = None

    def __post_init__(self) -> None:
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise SpecError(f"input shape must be positive and non-empty: {self.shape}")
        if self.dtype not in _DTYPES:
            raise SpecError(f"unknown input dtype {self.dtype!r}; known: {_DTYPES}")
        if self.generator not in _GENERATORS:
            raise SpecError(
                f"unknown generator {self.generator!r}; known: {_GENERATORS}"
            )
        if (self.generator == "image") != (self.image is not None):
            raise SpecError(
                "generator 'image' needs an 'image:' filename, and 'image:' "
                "needs generator 'image'"
            )


@dataclasses.dataclass(frozen=True)
class Tolerances:
    """How close the device has to be to the torch reference."""

    atol: float = 1e-2
    rtol: float = 1e-2
    relative_l2_percent: float | None = 0.1


@dataclasses.dataclass(frozen=True)
class SemanticCheck:
    """An assertion about what the output *means*, not just its numbers.

    For a classifier: the label the device is expected to predict. This needs no
    reference tensor at all.

    The label mapping comes from the model's own `config.id2label`, recorded
    into `labels.json` at import time.
    """

    expected_label: str
    output_index: int = 0


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """One entry of the model tree."""

    # The path under models/ is the name, and the directory is
    # where model.py and any image live.
    name: str
    directory: Path
    inputs: tuple[InputSpec, ...]
    dtype: str = "float32"
    tolerances: Tolerances = dataclasses.field(default_factory=Tolerances)
    expected_outcomes: tuple[ExpectedOutcome, ...] = ()
    # Which compile cases to run, by name. Empty means all of them. This is how
    # a model opts out of a configuration that is not interesting for it.
    compile_cases: tuple[str, ...] = ()
    semantic_check: SemanticCheck | None = None
    # Move the weights out of the MLIR and into a .irpa parameter archive. Off
    # for every model today: it does not compile for the hexagon target.
    externalize: bool = False
    extra_compile_flags: tuple[str, ...] = ()
    function: str = "main"

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise SpecError(f"{self.name}: unknown dtype {self.dtype!r}")
        if not self.inputs:
            raise SpecError(f"{self.name}: at least one input is required")

    @property
    def model_source(self) -> Path:
        return self.directory / "model.py"

    def runs_case(self, case_name: str) -> bool:
        """Whether this model asks for `case_name`. No list means all cases."""
        return not self.compile_cases or case_name in self.compile_cases

    def expected_outcome_for(self, case_name: str) -> ExpectedOutcome | None:
        """The recorded outcome for `case_name`, if the model declares one.

        An entry naming the case explicitly wins over a `case: "*"` entry, so a
        model that fails everywhere except in one configuration can say so.
        """
        wildcard = None
        for outcome in self.expected_outcomes:
            if outcome.case == case_name:
                return outcome
            if outcome.case == "*":
                wildcard = outcome
        return wildcard


_DACITE_CONFIG = dacite.Config(
    cast=[tuple, float],
    type_hooks={Status: parse_status},
    strict=True,
)


def load_model_spec(directory: Path, root: Path) -> ModelSpec:
    """Read `directory/model.yaml` into a validated `ModelSpec`."""
    yaml_path = directory / "model.yaml"
    if not yaml_path.exists():
        raise SpecError(f"{directory}: no model.yaml")
    if not (directory / "model.py").exists():
        raise SpecError(f"{directory}: no model.py next to model.yaml")

    data = yaml.safe_load(yaml_path.read_text())
    if not isinstance(data, dict):
        raise SpecError(f"{yaml_path}: expected a mapping at the top level")
    data = {key: value for key, value in data.items() if value is not None}

    # There two filed are deduced from the path in the tree, not read from the yaml.
    data["name"] = directory.relative_to(root).as_posix()
    data["directory"] = directory

    try:
        return dacite.from_dict(ModelSpec, data, _DACITE_CONFIG)
    except (dacite.DaciteError, TypeError, ValueError) as err:
        raise SpecError(f"{yaml_path}: {err}") from err
