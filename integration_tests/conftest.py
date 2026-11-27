# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Options and fixtures."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from integration_tests.device import adb
from integration_tests.device.deploy import (
    DEFAULT_DEVICE_ROOT,
    Deployments,
    cleanup,
    deploy,
)
from integration_tests.modeltree import discover
from integration_tests.modeltree.spec import MARKS, ModelSpec
from integration_tests.stages.compiling import (
    COMPILE_CASE_NAMES,
    COMPILE_CASES_BY_NAME,
    WEIGHTS_MODES,
    CompileCase,
)
from integration_tests.stages.running import (
    RUN_CASE_NAMES,
    RUN_CASES_BY_NAME,
    RunCase,
)
from integration_tests.tool_paths import (
    DEVICE_TOOLS_TARGET,
    IREE_COMPILE_TARGET,
    IREE_ENCODE_PARAMETERS_TARGET,
    LLD_TARGET,
    RUNTIME_ZIP_TARGET,
    RUNTIME_ZIP_TRACY_TARGET,
    resolve_bazel_artifact,
    resolve_host_tool,
    resolve_sibling_tool,
)


def pytest_configure(config: pytest.Config) -> None:
    # torch.export logs its whole tracing session at DEBUG. pytest captures it
    # and replays it into the report of any failing test, where it buries the
    # actual assertion under thousands of lines.
    logging.getLogger("torch").setLevel(logging.WARNING)
    for name, description in MARKS.items():
        config.addinivalue_line("markers", f"{name}: {description}")


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("hexagon")
    group.addoption(
        "--iree-compile",
        default=None,
        help=(
            f"path to iree-compile ({IREE_COMPILE_TARGET}). Resolved with "
            "'bazel cquery' when omitted."
        ),
    )
    group.addoption(
        "--iree-encode-parameters",
        default=None,
        help=(
            f"path to iree-encode-parameters ({IREE_ENCODE_PARAMETERS_TARGET}), "
            "which encodes external weights. Taken from beside iree-compile "
            "when omitted."
        ),
    )
    group.addoption(
        "--runtime-zip",
        default=None,
        help=(
            f"path to the Hexagon runtime package ({RUNTIME_ZIP_TARGET}). "
            "Resolved with 'bazel cquery' when omitted."
        ),
    )
    group.addoption(
        "--device-tools-zip",
        default=None,
        help=(
            f"path to the on-device test helpers ({DEVICE_TOOLS_TARGET}): "
            "limit_lifetime and rsync. Resolved with 'bazel cquery' when omitted."
        ),
    )
    group.addoption(
        "--lld",
        default=None,
        help=(
            f"path to ld.lld ({LLD_TARGET}), passed to iree-compile as the "
            "Hexagon and llvm-cpu linker. Omit to let iree-compile find one on "
            "PATH."
        ),
    )
    group.addoption(
        "--rsync",
        default=None,
        help=(
            "path to the host's rsync, which sends modules to the device as "
            "deltas. Taken from PATH when omitted."
        ),
    )
    group.addoption(
        "--runtime-zip-tracy",
        default=None,
        help=(
            f"path to the Hexagon runtime package with Tracy "
            f"({RUNTIME_ZIP_TRACY_TARGET}), for the profile run cases. Resolved "
            "with 'bazel cquery' when omitted."
        ),
    )
    group.addoption(
        "--tracy-port",
        type=int,
        default=8086,
        help="device port of the Tracy connection (default: 8086).",
    )
    group.addoption(
        "--device-root",
        default=DEFAULT_DEVICE_ROOT,
        help=(
            "root directory on the device for the per-run trees "
            f"(default: {DEFAULT_DEVICE_ROOT})."
        ),
    )
    group.addoption(
        "--keep-device-dir",
        action="store_true",
        default=False,
        help=(
            "leave this run's directory on the device, to inspect a failure. "
            "Don't remove earlier run's results in the same device directory, "
            "either."
        ),
    )
    group.addoption(
        "--model",
        action="append",
        default=[],
        metavar="ORG/NAME",
        help="run only this model from models/ (repeatable).",
    )
    group.addoption(
        "--run-skipped",
        action="store_true",
        default=False,
        help="also run the models whose model.yaml sets 'skip'.",
    )
    group.addoption(
        "--compile-case",
        action="append",
        default=[],
        choices=COMPILE_CASE_NAMES,
        metavar="NAME",
        help=(
            "run only this compile case from each selected model's configured "
            "cases (repeatable)."
        ),
    )
    group.addoption(
        "--weights",
        action="append",
        default=[],
        choices=WEIGHTS_MODES,
        metavar="MODE",
        help=(
            "run only this weights mode (embedded or external) from each "
            "selected model's configured modes (repeatable)."
        ),
    )
    group.addoption(
        "--run-case",
        action="append",
        default=[],
        choices=RUN_CASE_NAMES,
        metavar="NAME",
        help=(
            "run only this run case from each selected model's configured "
            "cases (repeatable)."
        ),
    )


def _selected_specs(config: pytest.Config) -> list[ModelSpec]:
    return discover.select(names=config.getoption("--model") or None)


def _param(config: pytest.Config, spec: ModelSpec, *values: object, id: str):
    """One test parameter, with the model's marks from its yaml, and marked
    skipped if the yaml says so.

    A skip mark is evaluated before any fixture is set up, so a skipped model
    is never imported.
    """
    marks = [getattr(pytest.mark, name) for name in spec.marks]
    if spec.skip is not None and not config.getoption("--run-skipped"):
        marks.append(pytest.mark.skip(reason=f"model.yaml skip: {spec.skip.strip()}"))
    return pytest.param(*values, marks=marks, id=id)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize over the model tree, so adding a model needs no test code."""
    if "model_spec" not in metafunc.fixturenames:
        return
    specs = _selected_specs(metafunc.config)
    if "compile_case" in metafunc.fixturenames:
        requested_compile = set(metafunc.config.getoption("--compile-case"))
        requested_run = set(metafunc.config.getoption("--run-case"))
        requested_weights = set(metafunc.config.getoption("--weights"))
        combinations: list[tuple[ModelSpec, CompileCase, RunCase, str]] = []
        for spec in specs:
            for compile_name in spec.compile_cases:
                if requested_compile and compile_name not in requested_compile:
                    continue
                for weights in spec.weights:
                    if requested_weights and weights not in requested_weights:
                        continue
                    for run_name in spec.run_cases:
                        if not requested_run or run_name in requested_run:
                            combinations.append(
                                (
                                    spec,
                                    COMPILE_CASES_BY_NAME[compile_name],
                                    RUN_CASES_BY_NAME[run_name],
                                    weights,
                                )
                            )
        if not combinations:
            raise pytest.UsageError(
                "no model/compile-case/run-case/weights combinations match "
                "--model, --compile-case, --run-case and --weights"
            )
        metafunc.parametrize(
            ("model_spec", "compile_case", "run_case", "weights"),
            [
                _param(
                    metafunc.config,
                    spec,
                    spec,
                    case,
                    run,
                    weights,
                    # Embedded weights keep the ids tests had before the
                    # weights modes existed, so their history carries on.
                    id=f"{spec.name}-{case.name}-{run.name}"
                    + ("-external" if weights == "external" else ""),
                )
                for spec, case, run, weights in combinations
            ],
            scope="session",
        )
        return
    metafunc.parametrize(
        "model_spec",
        [_param(metafunc.config, spec, spec, id=spec.name) for spec in specs],
        scope="session",
    )


def _run_order(item: pytest.Item) -> int:
    """micro/ models first"""
    callspec = getattr(item, "callspec", None)
    spec = callspec.params.get("model_spec") if callspec is not None else None
    return 0 if spec is not None and spec.name.startswith("micro/") else 1


# trylast: after pytest's own reordering, which groups the tests of a model.
# The sort is stable, so it keeps that grouping and the order within a group.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    items.sort(key=_run_order)


@pytest.fixture(scope="session")
def iree_compile(pytestconfig: pytest.Config) -> Path:
    return resolve_bazel_artifact(
        IREE_COMPILE_TARGET, pytestconfig.getoption("--iree-compile")
    )


@pytest.fixture(scope="session")
def iree_encode_parameters(pytestconfig: pytest.Config, iree_compile: Path) -> Path:
    return resolve_sibling_tool(
        "iree-encode-parameters",
        pytestconfig.getoption("--iree-encode-parameters"),
        iree_compile,
    )


@pytest.fixture(scope="session")
def runtime_zip(pytestconfig: pytest.Config) -> Path:
    return resolve_bazel_artifact(
        RUNTIME_ZIP_TARGET, pytestconfig.getoption("--runtime-zip")
    )


@pytest.fixture(scope="session")
def device_tools_zip(pytestconfig: pytest.Config) -> Path:
    return resolve_bazel_artifact(
        DEVICE_TOOLS_TARGET, pytestconfig.getoption("--device-tools-zip")
    )


@pytest.fixture(scope="session")
def runtime_zip_tracy(pytestconfig: pytest.Config) -> Path:
    return resolve_bazel_artifact(
        RUNTIME_ZIP_TRACY_TARGET, pytestconfig.getoption("--runtime-zip-tracy")
    )


@pytest.fixture(scope="session")
def lld(pytestconfig: pytest.Config) -> Path | None:
    """The linker to hand iree-compile, or None to let it search PATH."""
    given = pytestconfig.getoption("--lld")
    return Path(given).resolve() if given else None


@pytest.fixture(scope="session")
def host_rsync(pytestconfig: pytest.Config) -> Path:
    return resolve_host_tool("rsync", pytestconfig.getoption("--rsync"))


@pytest.fixture(scope="session")
def require_device() -> None:
    """Fail every device test if there is no usable device."""
    ok, detail = adb.available()
    if ok:
        return
    pytest.fail(
        f"no usable adb device ({detail}). The adb server is assumed to be up "
        "already - a restarted server silently drops a TCP device. Check "
        "'adb devices'; an 'unauthorized' phone needs 'Always allow' tapped on "
        "the handset, and $ANDROID_SERIAL picks between several.",
        pytrace=False,
    )


@pytest.fixture(scope="session")
def deployments(
    require_device: None,
    runtime_zip: Path,
    runtime_zip_tracy: Path,
    device_tools_zip: Path,
    pytestconfig: pytest.Config,
) -> Iterator[Deployments]:
    """Unpack both runtimes and the helper once per session, each in a fresh
    directory. The profile run cases use the Tracy one, the others the plain one.
    """
    root = pytestconfig.getoption("--device-root")
    keep_device_dir = pytestconfig.getoption("--keep-device-dir")
    active = Deployments(
        plain=deploy(
            runtime_zip=runtime_zip,
            tools_zip=device_tools_zip,
            root=root,
            keep_device_dir=keep_device_dir,
        ),
        tracy=deploy(
            runtime_zip=runtime_zip_tracy,
            tools_zip=device_tools_zip,
            root=root,
            has_tracy=True,
            keep_device_dir=keep_device_dir,
        ),
    )
    print(
        f"deployed to {active.plain.run_dir} (plain) and {active.tracy.run_dir} "
        f"(tracy) on {adb.describe_device()}"
    )
    yield active
    for deployment in (active.plain, active.tracy):
        if keep_device_dir:
            print(f"leaving {deployment.run_dir} on the device as requested")
        else:
            cleanup(deployment)


@pytest.fixture(scope="session")
def tracy_port(pytestconfig: pytest.Config) -> int:
    return pytestconfig.getoption("--tracy-port")
