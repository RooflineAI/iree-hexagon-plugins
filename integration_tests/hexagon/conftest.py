# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fixtures for the Hexagon on-device integration tests."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from integration_tests.hexagon.adb_device import AdbDevice, attached_devices
from integration_tests.hexagon.hexagon_runtime import DeployedRuntime, deploy_runtime
from integration_tests.hexagon.iree_tools import (
    IREE_COMPILE_TARGET,
    LLD_TARGET,
    RUNTIME_ZIP_TARGET,
    resolve_bazel_artifact,
)

_DEFAULT_REMOTE_DIR = "/data/local/tmp/hexagon_integration_tests"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "hexagon: test that runs on a physical Hexagon device."
    )
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
        "--lld",
        default=None,
        help=(
            f"path to ld.lld ({LLD_TARGET}), passed to iree-compile as the "
            "Hexagon and llvm-cpu linker. Omit to let iree-compile find one on "
            "PATH, which works in a shell but not under 'bazel test'."
        ),
    )
    group.addoption(
        "--android-serial",
        default=None,
        help=(
            "adb serial of the device to use. Defaults to $ANDROID_SERIAL, or "
            "to the only attached device."
        ),
    )
    group.addoption(
        "--device-dir",
        default=_DEFAULT_REMOTE_DIR,
        help=f"working directory on the device (default: {_DEFAULT_REMOTE_DIR}).",
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
def lld(pytestconfig: pytest.Config) -> Path | None:
    """The linker to hand iree-compile, or None to let it search PATH."""
    given = pytestconfig.getoption("--lld")
    return Path(given).resolve() if given else None


@pytest.fixture(scope="session")
def hexagon_device(pytestconfig: pytest.Config) -> AdbDevice:
    """The device under test, or a skip if there is none.

    Skipping rather than failing is the point: a machine with no phone should
    still be able to run the rest of the suite, and CI without device access
    should not go red for a reason it cannot fix. The corollary is that a green
    run does not prove anything ran on hardware - check the skip count.
    """
    requested = pytestconfig.getoption("--android-serial") or os.environ.get(
        "ANDROID_SERIAL"
    )
    serials = attached_devices()
    if not serials:
        pytest.skip(
            "no adb device in state 'device' (check 'adb devices'; an "
            "'unauthorized' phone needs 'Always allow' tapped on the handset)"
        )
    if requested:
        if requested not in serials:
            pytest.skip(
                f"requested device {requested!r} is not attached; attached: {serials}"
            )
        return AdbDevice(serial=requested)
    if len(serials) > 1:
        pytest.skip(
            f"{len(serials)} devices attached ({serials}); pick one with "
            "--android-serial or $ANDROID_SERIAL"
        )
    return AdbDevice(serial=serials[0])


@pytest.fixture(scope="session")
def hexagon_runtime(
    hexagon_device: AdbDevice, runtime_zip: Path, pytestconfig: pytest.Config
) -> DeployedRuntime:
    """Deploy the runtime package once per session."""
    remote_dir = pytestconfig.getoption("--device-dir")
    print(
        f"deploying {runtime_zip.name} to {hexagon_device.serial}:{remote_dir} "
        f"(model {hexagon_device.property('ro.product.model')}, "
        f"platform {hexagon_device.property('ro.board.platform')})"
    )
    return deploy_runtime(hexagon_device, runtime_zip, remote_dir)
