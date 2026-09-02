# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""One test for every (model, compile case) pair in the model tree.

There is no per-model test code. A model is `models/<org>/<name>/model.yaml`
plus a `model.py` with `get_model()`; `conftest.pytest_generate_tests`
parametrizes this file over whatever the tree holds and the tag filters select.

The four stages live in `stages/`, in pipeline order: import -> compile -> run
-> check. Each stage's failure has a name a `model.yaml` can record as expected,
together with a reason that has to appear in the log (`modeltree/outcomes.py`).
"""

from __future__ import annotations

import pathlib

import pytest

from integration_tests.device.deploy import Deployment
from integration_tests.modeltree.outcomes import Outcome, Status, check_outcome
from integration_tests.modeltree.spec import ModelSpec
from integration_tests.stages.artifacts import ImportedModel
from integration_tests.stages.checking import (
    AccuracyError,
    check_output_files,
    check_predicted_label,
)
from integration_tests.stages.compiling import (
    HEXAGON_DEFAULT_COMPILE_CASES,
    CompilationError,
    CompileCase,
    compile_model,
)
from integration_tests.stages.running import run_module_on_device


@pytest.fixture(scope="session")
def imported_model(
    model_spec: ModelSpec, tmp_path_factory: pytest.TempPathFactory
) -> ImportedModel:
    """The model's MLIR, inputs and torch reference, imported once per session."""
    from integration_tests.stages.importing import import_model

    output_dir = tmp_path_factory.mktemp(model_spec.name.replace("/", "_"))
    imported = import_model(model_spec, output_dir)
    print(
        f"{model_spec.name}: imported to {imported.model_mlir} "
        f"({imported.model_mlir.stat().st_size} bytes of MLIR)"
    )
    return imported


def _run_pipeline(
    model_spec: ModelSpec,
    compile_case: CompileCase,
    imported_model: ImportedModel,
    iree_compile: pathlib.Path,
    lld: pathlib.Path | None,
    deployment: Deployment,
    work_dir: pathlib.Path,
) -> Outcome:
    """Compile, run and check, returning where it got to and the evidence."""
    try:
        vmfb = compile_model(
            iree_compile=iree_compile,
            model_mlir=imported_model.model_mlir,
            output_vmfb=work_dir / "model.vmfb",
            compile_case=compile_case,
            log_file=work_dir / "iree-compile.log",
            linker=lld,
            model_flags=model_spec.extra_compile_flags,
        )
    except CompilationError as err:
        return Outcome(Status.COMPILE_FAILURE, f"{err}\n{err.log}")
    print(f"compiled {compile_case.name} -> {vmfb} ({vmfb.stat().st_size} bytes)")

    case_dir = f"{model_spec.name.replace('/', '_')}_{compile_case.name}"
    result = run_module_on_device(
        deployment=deployment,
        case_dir_name=case_dir,
        module_vmfb=vmfb,
        function=imported_model.function,
        input_files=imported_model.input_files,
        output_names=imported_model.output_names,
        local_output_dir=work_dir / "device_outputs",
    )
    if result.exit_code != 0:
        return Outcome(Status.RUNTIME_FAILURE, result.describe())
    print(result.log)

    try:
        check_output_files(
            reference_files=imported_model.output_files,
            actual_files=result.output_files,
            atol=model_spec.tolerances.atol,
            rtol=model_spec.tolerances.rtol,
            max_relative_l2_percent=model_spec.tolerances.relative_l2_percent,
        )
        if model_spec.semantic_check is not None:
            check = model_spec.semantic_check
            check_predicted_label(
                output_file=result.output_files[check.output_index],
                labels=imported_model.labels(),
                expected_label=check.expected_label,
            )
    except AccuracyError as err:
        return Outcome(Status.ACCURACY_FAILURE, f"{err}\n{result.describe()}")
    return Outcome(Status.PASSED, result.log)


@pytest.mark.parametrize(
    "compile_case",
    HEXAGON_DEFAULT_COMPILE_CASES,
    ids=[case.name for case in HEXAGON_DEFAULT_COMPILE_CASES],
)
def test_model_on_device(
    model_spec: ModelSpec,
    compile_case: CompileCase,
    imported_model: ImportedModel,
    iree_compile: pathlib.Path,
    lld: pathlib.Path | None,
    deployment: Deployment,
    tmp_path: pathlib.Path,
) -> None:
    if not model_spec.runs_case(compile_case.name):
        pytest.skip(f"{model_spec.name} asks only for {list(model_spec.compile_cases)}")

    outcome = _run_pipeline(
        model_spec=model_spec,
        compile_case=compile_case,
        imported_model=imported_model,
        iree_compile=iree_compile,
        lld=lld,
        deployment=deployment,
        work_dir=tmp_path,
    )

    expected = model_spec.expected_outcome_for(compile_case.name)
    if expected is None:
        if not outcome.passed:
            pytest.fail(
                f"{model_spec.name} [{compile_case.name}] failed as "
                f"{outcome.status}:\n{outcome.log}",
                pytrace=False,
            )
        return

    mismatch = check_outcome(expected, outcome)
    if mismatch is not None:
        pytest.fail(
            f"{model_spec.name} [{compile_case.name}]: {mismatch.kind.title()}\n"
            f"{mismatch.message}",
            pytrace=False,
        )
    # Reported as XFAIL, which is what it is - but only after the failure was
    # confirmed to be the recorded one, so the strictness that
    # xfail(strict=True) gave is preserved and the reason is now evidence-backed.
    pytest.xfail(f"{expected.status}: {expected.reason}")
