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
import typing
from pathlib import Path

import dacite
import yaml

from integration_tests.modeltree.outcomes import ExpectedOutcome, Status
from integration_tests.stages.compiling import (
    COMPILE_CASE_NAMES,
    DEFAULT_COMPILE_CASE_NAMES,
)
from integration_tests.stages.profile_checks import (
    PROFILE_CHECKS,
    validate_check_args,
)
from integration_tests.stages.running import (
    DEFAULT_RUN_CASE_NAMES,
    RUN_CASE_NAMES,
    RUN_CASES_BY_NAME,
)

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


# eq=False keeps it hashable (by identity) despite the dict, so ModelSpec is too.
@dataclasses.dataclass(frozen=True, eq=False)
class ProfileCheckSpec:
    """One `profile_checks` entry: a check from `stages/profile_checks.py`,
    the profile run cases and compile cases it applies to, and its arguments.
    """

    check: str
    # "*" means every profile run case, or every compile case, of the model.
    run_case: str = "*"
    compile_case: str = "*"
    args: dict[str, typing.Any] = dataclasses.field(default_factory=dict)

    def applies_to(self, compile_case: str, run_case: str) -> bool:
        return (
            RUN_CASES_BY_NAME[run_case].profile
            and self.compile_case in ("*", compile_case)
            and self.run_case in ("*", run_case)
        )


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
    # Which compile cases to run. Most models exercise both custom Hexagon
    # configurations; other configurations are requested explicitly.
    compile_cases: tuple[str, ...] = DEFAULT_COMPILE_CASE_NAMES
    # Which runtime configurations to run each compile case with. Each one is
    # a separate test of the same compiled module.
    run_cases: tuple[str, ...] = DEFAULT_RUN_CASE_NAMES
    # Checks on the profile of the profile run cases (e.g. parallel-profile).
    profile_checks: tuple[ProfileCheckSpec, ...] = ()
    semantic_check: SemanticCheck | None = None
    extra_compile_flags: tuple[str, ...] = ()
    # Seconds the module push, and separately iree-run-module on the device,
    # may take before they are killed.
    run_timeout: float = 600.0
    # Why the model is not run by default, e.g. an import too expensive to
    # repeat while a known failure stands. Its tests are reported as skipped
    # with this reason, unless pytest is given --run-skipped.
    skip: str | None = None

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise SpecError(f"{self.name}: unknown dtype {self.dtype!r}")
        if not self.inputs:
            raise SpecError(f"{self.name}: at least one input is required")
        unknown = [c for c in self.compile_cases if c not in COMPILE_CASE_NAMES]
        if unknown:
            raise SpecError(
                f"{self.name}: unknown compile cases {unknown}; known: {COMPILE_CASE_NAMES}"
            )
        if not self.compile_cases:
            raise SpecError(f"{self.name}: at least one compile case is required")
        unknown = [c for c in self.run_cases if c not in RUN_CASE_NAMES]
        if unknown:
            raise SpecError(
                f"{self.name}: unknown run cases {unknown}; known: {RUN_CASE_NAMES}"
            )
        if not self.run_cases:
            raise SpecError(f"{self.name}: at least one run case is required")
        if self.skip is not None and not self.skip.strip():
            raise SpecError(f"{self.name}: 'skip' must say why the model is skipped")
        for check in self.profile_checks:
            self._validate_profile_check(check)

    def _validate_profile_check(self, check: ProfileCheckSpec) -> None:
        where = f"{self.name}: profile check {check.check!r}"
        if check.check not in PROFILE_CHECKS:
            raise SpecError(f"{where} is unknown; known: {sorted(PROFILE_CHECKS)}")
        try:
            validate_check_args(check.check, check.args)
        except TypeError as err:
            raise SpecError(f"{where}: bad args {check.args}: {err}") from err
        profile_run_cases = [
            name for name in self.run_cases if RUN_CASES_BY_NAME[name].profile
        ]
        if check.run_case == "*":
            if not profile_run_cases:
                raise SpecError(
                    f"{where}: the model has no profile run case in 'run_cases'"
                )
        elif check.run_case not in profile_run_cases:
            raise SpecError(
                f"{where}: run case {check.run_case!r} is not one of the model's "
                f"profile run cases {profile_run_cases}"
            )
        if check.compile_case != "*" and check.compile_case not in self.compile_cases:
            raise SpecError(
                f"{where}: compile case {check.compile_case!r} is not one of the "
                f"model's compile cases {list(self.compile_cases)}"
            )

    def profile_checks_for(
        self, compile_case: str, run_case: str
    ) -> list[ProfileCheckSpec]:
        return [c for c in self.profile_checks if c.applies_to(compile_case, run_case)]

    @property
    def model_source(self) -> Path:
        return self.directory / "model.py"

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
    cast=[tuple, float, Status],
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
