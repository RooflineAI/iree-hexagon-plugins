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
from typing import Any

import yaml

from integration_tests.modeltree.outcomes import ExpectedOutcome

# numpy/torch dtype names accepted in `inputs[].dtype`. Closed
# set: a typo here would otherwise produce a silently different reference.
_DTYPES = ("float32", "float16", "bfloat16", "int64", "int32", "bool")

_GENERATORS = ("normal", "uniform", "ones", "zeros", "image")


class SpecError(ValueError):
    pass


def _require_keys(where: str, data: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise SpecError(
            f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}"
        )


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

    @classmethod
    def from_yaml(cls, where: str, data: dict[str, Any]) -> InputSpec:
        _require_keys(
            where,
            data,
            {"shape", "dtype", "generator", "seed", "low", "high", "image"},
        )
        if "shape" not in data or "dtype" not in data:
            raise SpecError(f"{where}: 'shape' and 'dtype' are required")
        return cls(
            shape=tuple(int(dim) for dim in data["shape"]),
            dtype=str(data["dtype"]),
            generator=str(data.get("generator", "normal")),
            seed=int(data.get("seed", 0)),
            low=int(data.get("low", 0)),
            high=None if data.get("high") is None else int(data["high"]),
            image=None if data.get("image") is None else str(data["image"]),
        )


@dataclasses.dataclass(frozen=True)
class Tolerances:
    """How close the device has to be to the torch reference.

    Two criteria, because they fail differently. `atol`/`rtol` is element-wise
    and catches a single wrong lane; relative L2 is aggregate and is the one
    that stays meaningful on a large output, where a handful of elements over
    the element-wise bound says nothing.
    """

    atol: float = 1e-2
    rtol: float = 1e-2
    relative_l2_percent: float | None = 0.1

    @classmethod
    def from_yaml(cls, where: str, data: dict[str, Any]) -> Tolerances:
        _require_keys(where, data, {"atol", "rtol", "relative_l2_percent"})
        raw = data.get("relative_l2_percent", 0.1)
        return cls(
            atol=float(data.get("atol", 1e-2)),
            rtol=float(data.get("rtol", 1e-2)),
            relative_l2_percent=None if raw is None else float(raw),
        )


@dataclasses.dataclass(frozen=True)
class SemanticCheck:
    """An assertion about what the output *means*, not just its numbers.

    For a classifier: the label the device is expected to predict. This needs no
    reference tensor at all, which makes it the one check that cannot be
    satisfied by a tolerance chosen after the fact - the argument for having a
    real photograph in the tree rather than only random inputs.

    The label mapping comes from the model's own `config.id2label`, recorded
    into `labels.json` at import time so the check stays torch-free.
    """

    expected_label: str
    output_index: int = 0

    @classmethod
    def from_yaml(cls, where: str, data: dict[str, Any]) -> SemanticCheck:
        _require_keys(where, data, {"expected_label", "output_index"})
        if "expected_label" not in data:
            raise SpecError(f"{where}: 'expected_label' is required")
        return cls(
            expected_label=str(data["expected_label"]),
            output_index=int(data.get("output_index", 0)),
        )


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """One entry of the model tree."""

    name: str
    directory: Path
    model_id: str
    dtype: str
    inputs: tuple[InputSpec, ...]
    tolerances: Tolerances
    tags: tuple[str, ...]
    expected_outcomes: tuple[ExpectedOutcome, ...]
    revision: str | None = None
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


_ALLOWED_TOP_LEVEL = {
    "model",
    "revision",
    "dtype",
    "inputs",
    "tolerances",
    "tags",
    "expected_outcomes",
    "compile_cases",
    "semantic_check",
    "externalize",
    "extra_compile_flags",
    "function",
}


def load_model_spec(directory: Path) -> ModelSpec:
    """Read `directory/model.yaml` into a validated `ModelSpec`."""
    yaml_path = directory / "model.yaml"
    if not yaml_path.exists():
        raise SpecError(f"{directory}: no model.yaml")
    if not (directory / "model.py").exists():
        raise SpecError(f"{directory}: no model.py next to model.yaml")

    data = yaml.safe_load(yaml_path.read_text())
    if not isinstance(data, dict):
        raise SpecError(f"{yaml_path}: expected a mapping at the top level")
    _require_keys(str(yaml_path), data, _ALLOWED_TOP_LEVEL)
    if "model" not in data:
        raise SpecError(f"{yaml_path}: 'model' (the hub id) is required")

    model_id = str(data["model"])
    inputs = tuple(
        InputSpec.from_yaml(f"{yaml_path} inputs[{index}]", entry)
        for index, entry in enumerate(data.get("inputs") or ())
    )
    outcomes = tuple(
        ExpectedOutcome.from_yaml(f"{yaml_path} expected_outcomes[{index}]", entry)
        for index, entry in enumerate(data.get("expected_outcomes") or ())
    )
    return ModelSpec(
        # The tree layout mirrors the hub id, and the id is what people search
        # for, so it is also the test name.
        name=model_id,
        directory=directory,
        model_id=model_id,
        revision=None if data.get("revision") is None else str(data["revision"]),
        dtype=str(data.get("dtype", "float32")),
        inputs=inputs,
        tolerances=Tolerances.from_yaml(
            f"{yaml_path} tolerances", data.get("tolerances") or {}
        ),
        tags=tuple(str(tag) for tag in data.get("tags") or ()),
        expected_outcomes=outcomes,
        compile_cases=tuple(str(name) for name in data.get("compile_cases") or ()),
        semantic_check=(
            None
            if data.get("semantic_check") is None
            else SemanticCheck.from_yaml(
                f"{yaml_path} semantic_check", data["semantic_check"]
            )
        ),
        externalize=bool(data.get("externalize", False)),
        extra_compile_flags=tuple(
            str(flag) for flag in data.get("extra_compile_flags") or ()
        ),
        function=str(data.get("function", "main")),
    )
