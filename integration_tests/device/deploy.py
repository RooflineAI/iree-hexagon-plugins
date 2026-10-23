# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""What gets put on the device, and where.

Two packages are unpacked into one directory per run:

  * the runtime, `//plugins/runtime/hexagon:hexagon_runtime_aarch64_android`,
    whose layout is:

        bin/iree-run-module          ARM host side, Hexagon HAL driver linked in
        bin/iree-benchmark-module    same, for benchmarking
        bin/tracy-capture            only in the Tracy runtime, records the
                                     trace on the device
        bin/tracy-csvexport          only in the Tracy runtime, exports it
        lib/hexagon/libhexagon_dsp_skel.so   the DSP side
        lib/libc++_shared.so         NDK C++ runtime

  * `//integration_tests/device/tools:device_tools_aarch64_android`,
    which adds `bin/limit_lifetime`.

Layout on the device, under one root (default `/data/local/tmp/hexagon_it`):

    runs/<run-id>/bin,lib/   the runtime and the helpers for this run
    runs/<run-id>/cases/     one directory per (model, compile case)

There is no lock: this suite is meant for one runner executing serially.
"""

from __future__ import annotations

import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path

from integration_tests.device import adb

# FARF mask enabling all five DSP debug flags. The file has to be named after
# the executable and live in the directory named by DSP_LIBRARY_PATH, which is
# also where the skel lives.
_FARF_MASK = "0x1f"

_ENTRY_BINARIES = ("iree-run-module", "iree-benchmark-module")

DEFAULT_DEVICE_ROOT = "/data/local/tmp/hexagon_it"


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class Deployment:
    """The runtime and helpers unpacked in this run's directory on the device."""

    root: str
    run_id: str
    has_tracy: bool = False

    @property
    def run_dir(self) -> str:
        return f"{self.root}/runs/{self.run_id}"

    @property
    def run_module(self) -> str:
        return f"{self.run_dir}/bin/iree-run-module"

    @property
    def tracy_capture(self) -> str:
        """Only present in a deployment of the Tracy runtime."""
        if not self.has_tracy:
            raise ValueError(f"{self.run_dir} is not a deployment of the Tracy runtime")
        return f"{self.run_dir}/bin/tracy-capture"

    @property
    def tracy_csvexport(self) -> str:
        """Only present in a deployment of the Tracy runtime."""
        if not self.has_tracy:
            raise ValueError(f"{self.run_dir} is not a deployment of the Tracy runtime")
        return f"{self.run_dir}/bin/tracy-csvexport"

    @property
    def limit_lifetime(self) -> str:
        return f"{self.run_dir}/bin/limit_lifetime"

    @property
    def dsp_library_path(self) -> str:
        return f"{self.run_dir}/lib/hexagon"

    def case_dir(self, case_dir_name: str) -> str:
        return f"{self.run_dir}/cases/{case_dir_name}"


@dataclass(frozen=True)
class Deployments:
    """The plain runtime and the Tracy runtime, each in its own run directory."""

    plain: Deployment
    tracy: Deployment


def _unpack(zip_path: Path, remote_dir: str) -> None:
    adb.shell(f"mkdir -p {shlex.quote(remote_dir)}")
    adb.push([zip_path], remote_dir + "/")
    adb.shell(
        f"cd {shlex.quote(remote_dir)} && unzip -oq {shlex.quote(zip_path.name)} && rm {shlex.quote(zip_path.name)}"
    )


def deploy(
    runtime_zip: Path,
    tools_zip: Path,
    root: str = DEFAULT_DEVICE_ROOT,
    run_id: str | None = None,
    has_tracy: bool = False,
) -> Deployment:
    """Unpack the runtime and the helpers into a fresh directory for this run.

    `has_tracy` says whether `runtime_zip` is the Tracy runtime.
    """
    deployment = Deployment(
        root=root, run_id=run_id or new_run_id(), has_tracy=has_tracy
    )
    _unpack(runtime_zip, deployment.run_dir)
    _unpack(tools_zip, deployment.run_dir)
    farf_writes = " && ".join(
        f"printf '{_FARF_MASK}\\n' > {shlex.quote(f'lib/hexagon/{binary}.farf')}"
        for binary in _ENTRY_BINARIES
    )
    # Which binaries there are depends on the runtime: only the Tracy one has
    # tracy-capture and tracy-csvexport.
    adb.shell(
        f"cd {shlex.quote(deployment.run_dir)} && chmod +x bin/* && {farf_writes}"
    )
    return deployment


def cleanup(deployment: Deployment) -> None:
    """Remove this run's directory, tolerating a device that has gone away."""
    try:
        adb.shell(f"rm -rf {shlex.quote(deployment.run_dir)}")
    except Exception as err:  # noqa: BLE001 - teardown must not mask a failure
        print(f"warning: could not remove {deployment.run_dir}: {err}")
