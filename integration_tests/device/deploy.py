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
        lib/hexagon/libhexagon_dsp_skel.so   the DSP side
        lib/libc++_shared.so         NDK C++ runtime

  * `//integration_tests/device/tools:device_tools_aarch64_android`,
    which adds `bin/limit_lifetime`.

Layout on the device, under one root (default `/data/local/tmp/hexagon_it`):

    runs/<run-id>/bin,lib/   the runtime and the helpers for this run
    runs/<run-id>/params/    parameter archives, one directory per model
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

    @property
    def run_dir(self) -> str:
        return f"{self.root}/runs/{self.run_id}"

    @property
    def run_module(self) -> str:
        return f"{self.run_dir}/bin/iree-run-module"

    @property
    def limit_lifetime(self) -> str:
        return f"{self.run_dir}/bin/limit_lifetime"

    @property
    def dsp_library_path(self) -> str:
        return f"{self.run_dir}/lib/hexagon"

    def case_dir(self, case_dir_name: str) -> str:
        return f"{self.run_dir}/cases/{case_dir_name}"

    def parameters_dir(self, model_name: str) -> str:
        """Where a model's parameter archive lives, shared by its cases."""
        return f"{self.run_dir}/params/{model_name.replace('/', '_')}"


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
) -> Deployment:
    """Unpack the runtime and the helpers into a fresh directory for this run."""
    deployment = Deployment(root=root, run_id=run_id or new_run_id())
    _unpack(runtime_zip, deployment.run_dir)
    _unpack(tools_zip, deployment.run_dir)
    farf_writes = " && ".join(
        f"printf '{_FARF_MASK}\\n' > lib/hexagon/{binary}.farf"
        for binary in _ENTRY_BINARIES
    )
    chmods = " ".join(
        f"bin/{binary}" for binary in (*_ENTRY_BINARIES, "limit_lifetime")
    )
    adb.shell(
        f"cd {shlex.quote(deployment.run_dir)} && chmod +x {shlex.quote(chmods)} && {shlex.quote(farf_writes)}"
    )
    return deployment


def push_parameters(deployment: Deployment, model_name: str, archive: Path) -> str:
    """Push a model's `.irpa` once, returning its path on the device.

    Once per *model*, not once per compile case: the archive is the weights, and
    the weights do not change between cases. For SmolLM2-135M at f16 that is one
    326 MB push per run instead of one per case.
    """
    remote_dir = deployment.parameters_dir(model_name)
    remote_path = f"{remote_dir}/{archive.name}"
    adb.shell(f"mkdir -p {shlex.quote(remote_dir)}")
    adb.push([archive], remote_dir + "/")
    return remote_path


def cleanup(deployment: Deployment) -> None:
    """Remove this run's directory, tolerating a device that has gone away."""
    try:
        adb.shell(f"rm -rf {shlex.quote(deployment.run_dir)}")
    except Exception as err:  # noqa: BLE001 - teardown must not mask a failure
        print(f"warning: could not remove {deployment.run_dir}: {err}")
