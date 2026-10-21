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
from integration_tests.modeltree.spec import ModelSpec
from integration_tests.stages.compiling import (
    COMPILE_CASE_NAMES,
    COMPILE_CASES_BY_NAME,
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
    LLD_TARGET,
    RUNTIME_ZIP_TARGET,
    RUNTIME_ZIP_TRACY_TARGET,
    resolve_bazel_artifact,
    resolve_host_tool,
)


def pytest_configure(config: pytest.Config) -> None:
    # torch.export logs its whole tracing session at DEBUG. pytest captures it
    # and replays it into the report of any failing test, where it buries the
    # actual assertion under thousands of lines.
    logging.getLogger("torch").setLevel(logging.WARNING)


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
        help="leave this run's directory on the device, to inspect a failure.",
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
    """One test parameter, marked skipped if the model's yaml says so.

    A skip mark is evaluated before any fixture is set up, so a skipped model
    is never imported.
    """
    marks = []
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
        triples: list[tuple[ModelSpec, CompileCase, RunCase]] = []
        for spec in specs:
            for compile_name in spec.compile_cases:
                if requested_compile and compile_name not in requested_compile:
                    continue
                for run_name in spec.run_cases:
                    if not requested_run or run_name in requested_run:
                        triples.append(
                            (
                                spec,
                                COMPILE_CASES_BY_NAME[compile_name],
                                RUN_CASES_BY_NAME[run_name],
                            )
                        )
        if not triples:
            raise pytest.UsageError(
                "no model/compile-case/run-case combinations match --model, "
                "--compile-case and --run-case"
            )
        metafunc.parametrize(
            ("model_spec", "compile_case", "run_case"),
            [
                _param(
                    metafunc.config,
                    spec,
                    spec,
                    case,
                    run,
                    id=f"{spec.name}-{case.name}-{run.name}",
                )
                for spec, case, run in triples
            ],
            scope="session",
        )
        return
    metafunc.parametrize(
        "model_spec",
        [_param(metafunc.config, spec, spec, id=spec.name) for spec in specs],
        scope="session",
    )


@pytest.fixture(scope="session")
def iree_compile(pytestconfig: pytest.Config) -> Path:
    return resolve_bazel_artifact(
        IREE_COMPILE_TARGET, pytestconfig.getoption("--iree-compile")
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
    active = Deployments(
        plain=deploy(runtime_zip=runtime_zip, tools_zip=device_tools_zip, root=root),
        tracy=deploy(
            runtime_zip=runtime_zip_tracy, tools_zip=device_tools_zip, root=root
        ),
    )
    print(
        f"deployed to {active.plain.run_dir} (plain) and {active.tracy.run_dir} "
        f"(tracy) on {adb.describe_device()}"
    )
    yield active
    for deployment in (active.plain, active.tracy):
        if pytestconfig.getoption("--keep-device-dir"):
            print(f"leaving {deployment.run_dir} on the device as requested")
        else:
            cleanup(deployment)


@pytest.fixture(scope="session")
def tracy_port(pytestconfig: pytest.Config) -> int:
    return pytestconfig.getoption("--tracy-port")
