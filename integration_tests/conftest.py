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
    Deployment,
    cleanup,
    deploy,
)
from integration_tests.modeltree import discover
from integration_tests.modeltree.spec import ModelSpec
from integration_tests.tool_paths import (
    DEVICE_TOOLS_TARGET,
    IREE_COMPILE_TARGET,
    LLD_TARGET,
    RUNTIME_ZIP_TARGET,
    resolve_bazel_artifact,
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
            f"path to the on-device test helper ({DEVICE_TOOLS_TARGET}): "
            "limit_lifetime. Resolved with 'bazel cquery' when omitted."
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


def _selected_specs(config: pytest.Config) -> list[ModelSpec]:
    return discover.select(names=config.getoption("--model") or None)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize over the model tree, so adding a model needs no test code."""
    if "model_spec" not in metafunc.fixturenames:
        return
    specs = _selected_specs(metafunc.config)
    metafunc.parametrize(
        "model_spec", specs, ids=[spec.name for spec in specs], scope="session"
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
def lld(pytestconfig: pytest.Config) -> Path | None:
    """The linker to hand iree-compile, or None to let it search PATH."""
    given = pytestconfig.getoption("--lld")
    return Path(given).resolve() if given else None


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
def deployment(
    require_device: None,
    runtime_zip: Path,
    device_tools_zip: Path,
    pytestconfig: pytest.Config,
) -> Iterator[Deployment]:
    """Unpack the runtime and the helper once per session, in a fresh directory."""
    active = deploy(
        runtime_zip=runtime_zip,
        tools_zip=device_tools_zip,
        root=pytestconfig.getoption("--device-root"),
    )
    print(f"deployed to {active.run_dir} on {adb.describe_device()}")
    yield active
    if pytestconfig.getoption("--keep-device-dir"):
        print(f"leaving {active.run_dir} on the device as requested")
    else:
        cleanup(active)
