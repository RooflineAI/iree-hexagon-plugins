# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""One test for every (model, compile case) pair in the model tree.

There is no per-model test code. A model is `models/<org>/<name>/model.yaml`
plus a `model.py` with `get_model()`; `conftest.pytest_generate_tests`
parametrizes this file over the model/case pairs selected by the manifests and
CLI filters.

The four stages live in `stages/`, in pipeline order: import -> compile -> run
-> check. Each stage's failure has a name a `model.yaml` can record as expected,
together with a reason that has to appear in the log (`modeltree/outcomes.py`).
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator

import pytest

from integration_tests.device.deploy import Deployment, Deployments
from integration_tests.modeltree.outcomes import Outcome, Status, check_outcome
from integration_tests.modeltree.spec import ModelSpec
from integration_tests.prefetch import Prefetcher
from integration_tests.stages.artifacts import ImportedModel, ParameterArchive
from integration_tests.stages.checking import (
    AccuracyError,
    check_generation,
    check_output_files,
    check_predicted_label,
)
from integration_tests.stages.compiling import (
    CompilationError,
    CompileCase,
)
from integration_tests.stages.profile_checks import Profile, ProfileError, run_check
from integration_tests.stages.running import (
    RunCase,
    run_llm_on_device,
    run_module_on_device,
)


@pytest.fixture(scope="session")
def prefetcher(
    request: pytest.FixtureRequest,
    iree_compile: pathlib.Path,
    iree_encode_parameters: pathlib.Path,
    lld: pathlib.Path | None,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Prefetcher]:
    """Imports and compiles all tests' models in the background, in test order."""
    active = Prefetcher(
        request.session.items,
        iree_compile,
        iree_encode_parameters,
        lld,
        tmp_path_factory,
    )
    yield active
    active.close()  # cleanup


def _run_pipeline(
    model_spec: ModelSpec,
    compile_case: CompileCase,
    run_case: RunCase,
    weights: str,
    prefetcher: Prefetcher,
    deployment: Deployment,
    work_dir: pathlib.Path,
    tracy_port: int | None,
) -> Outcome:
    """fetch compile result, run and check, returning where it got to and the
    evidence."""
    try:
        # waits for compilation to finish
        compiled = prefetcher.compiled(model_spec, compile_case, weights)
    except CompilationError as err:
        return Outcome(Status.COMPILE_FAILURE, f"{err}\n{err.log}")
    # A ModelImportError (the import failed) is not caught: it fails the test,
    # and its message carries the import's output. It is not an Outcome, as
    # model.yaml cannot record an import failure.
    imported_model, vmfb = compiled.imported, compiled.vmfb
    print(
        f"{model_spec.name}: imported to {imported_model.model_mlir} "
        f"({imported_model.model_mlir.stat().st_size} bytes of MLIR)"
    )
    print(
        f"compiled {compile_case.name} in {compiled.seconds:.2f}s -> {vmfb} ({vmfb.stat().st_size} bytes)"
    )
    parameters = compiled.parameters
    if parameters is not None:
        print(
            f"weights in {parameters.path} (scope {parameters.scope}, "
            f"{parameters.path.stat().st_size} bytes, sha256 {parameters.sha256}), "
            f"encoded and hashed in {compiled.encode_seconds:.2f}s"
        )
    elif weights == "external":
        print("external weights requested, but the model has no weights")

    case_dir = (
        f"{model_spec.name.replace('/', '_')}_{compile_case.name}_{run_case.name}"
    ) + ("_external" if weights == "external" else "")
    local_output_dir = work_dir / "device_outputs"
    if imported_model.generation is not None:
        return _run_generation(
            model_spec=model_spec,
            run_case=run_case,
            imported_model=imported_model,
            vmfb=vmfb,
            deployment=deployment,
            case_dir=case_dir,
            local_output_dir=local_output_dir,
            tracy_port=tracy_port,
            parameters=parameters,
        )

    result = run_module_on_device(
        deployment=deployment,
        case_dir_name=case_dir,
        module_vmfb=vmfb,
        calls=imported_model.calls,
        local_output_dir=local_output_dir,
        extra_flags=run_case.flags,
        timeout=model_spec.run_timeout,
        tracy_port=tracy_port,
        debug_malloc=run_case.debug_malloc,
        parameters=parameters,
    )
    print(result.log)
    print("timings:")
    for key, value in result.timings.items():
        print(f"  {key}: {value:.3f}s")
    if result.exit_code != 0:
        return Outcome(Status.RUNTIME_FAILURE, result.describe())

    try:
        for call in imported_model.calls:
            actual_files = [local_output_dir / name for name in call.output_names]
            check_output_files(
                reference_files=call.output_files,
                actual_files=actual_files,
                atol=model_spec.tolerances.atol,
                rtol=model_spec.tolerances.rtol,
                max_relative_l2_percent=model_spec.tolerances.relative_l2_percent,
            )
            if model_spec.semantic_check is not None:
                check = model_spec.semantic_check
                check_predicted_label(
                    output_file=actual_files[check.output_index],
                    labels=imported_model.labels(),
                    expected_label=check.expected_label,
                )
    except AccuracyError as err:
        return Outcome(Status.ACCURACY_FAILURE, f"{err}\n{result.describe()}")

    if tracy_port is not None:
        if result.profile_error is not None:
            return Outcome(Status.CAPTURE_PROFILE_FAILURE, result.profile_error)
        print(f"profile csv: {result.profile_csv}")
        profile = Profile.from_csv(result.profile_csv)
        failures = []
        for check in model_spec.profile_checks_for(compile_case.name, run_case.name):
            try:
                run_check(check.check, profile, check.args)
            except ProfileError as err:
                failures.append(f"profile check {check.check} {check.args}: {err}")
            else:
                print(f"profile check {check.check} {check.args}: passed")
        if failures:
            # The trace stays on the device, so point at how to get one.
            failures.append(
                "the trace is not kept; to look at one, run "
                "dev_tools/profile_on_device.sh [--serial] "
                f"{model_spec.directory}"
            )
            return Outcome(Status.PROFILE_FAILURE, "\n".join(failures))
    return Outcome(Status.PASSED, result.log)


def _run_generation(
    model_spec: ModelSpec,
    run_case: RunCase,
    imported_model: ImportedModel,
    vmfb: pathlib.Path,
    deployment: Deployment,
    case_dir: str,
    local_output_dir: pathlib.Path,
    tracy_port: int | None,
    parameters: ParameterArchive | None,
) -> Outcome:
    """Run and check a `KvCacheLM` module: token generation with iree-run-llm."""
    generation_reference = imported_model.generation
    assert generation_reference is not None
    if tracy_port is not None:
        raise ValueError("profiling does not support token generation")
    generation = run_llm_on_device(
        deployment=deployment,
        case_dir_name=case_dir,
        module_vmfb=vmfb,
        generation=generation_reference,
        local_output_dir=local_output_dir,
        extra_flags=run_case.flags,
        timeout=model_spec.run_timeout,
        parameters=parameters,
    )
    print(generation.log)
    # perf_compare.py compares prefill and decode_per_token from here,
    # instead of the `run:` under "timings:" a plain run has.
    print("generation timings:")
    for key, value in generation.timings.items():
        print(f"  {key}: {value:.3f}s")
    if generation.exit_code != 0:
        return Outcome(Status.RUNTIME_FAILURE, generation.describe())
    try:
        check_generation(
            reference=generation_reference,
            prompt_ids=generation.prompt_ids,
            tokens=generation.tokens,
            prefill_logits_file=generation.prefill_logits_file,
            atol=model_spec.tolerances.atol,
            rtol=model_spec.tolerances.rtol,
            max_relative_l2_percent=model_spec.tolerances.relative_l2_percent,
        )
    except AccuracyError as err:
        return Outcome(Status.ACCURACY_FAILURE, f"{err}\n{generation.describe()}")
    return Outcome(Status.PASSED, generation.log)


def test_model_on_device(
    model_spec: ModelSpec,
    compile_case: CompileCase,
    run_case: RunCase,
    weights: str,
    prefetcher: Prefetcher,
    deployments: Deployments,
    tracy_port: int,
    tmp_path: pathlib.Path,
) -> None:
    outcome = _run_pipeline(
        model_spec=model_spec,
        compile_case=compile_case,
        run_case=run_case,
        weights=weights,
        prefetcher=prefetcher,
        deployment=deployments.tracy if run_case.profile else deployments.plain,
        work_dir=tmp_path,
        tracy_port=tracy_port if run_case.profile else None,
    )

    expected = model_spec.expected_outcome_for(compile_case.name)
    if expected is None:
        if not outcome.passed:
            pytest.fail(
                f"{model_spec.name} [{compile_case.name}/{run_case.name}/{weights} weights] failed as "
                f"{outcome.status}:\n{outcome.log}",
                pytrace=False,
            )
        return

    mismatch = check_outcome(expected, outcome)
    if mismatch is not None:
        pytest.fail(
            f"{model_spec.name} [{compile_case.name}/{run_case.name}/{weights} weights]: {mismatch.kind.title()}\n"
            f"{mismatch.message}",
            pytrace=False,
        )
    # Reported as XFAIL, which is what it is - but only after the failure was
    # confirmed to be the recorded one, so the strictness that
    # xfail(strict=True) gave is preserved and the reason is now evidence-backed.
    pytest.xfail(f"{expected.status}: {expected.reason}")
