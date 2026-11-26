# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""What gets put on the device, and where.

Two packages are unpacked into one directory per host user and checkout:

  * the runtime, `//plugins/runtime/hexagon:hexagon_runtime_aarch64_android`,
    whose layout is:

        bin/iree-run-module          ARM host side, Hexagon HAL driver linked in
        bin/iree-benchmark-module    same, for benchmarking
        bin/iree-run-llm             same, an LLM's prefill/decode loop
        bin/tracy-capture            only in the Tracy runtime, records the
                                     trace on the device
        bin/tracy-csvexport          only in the Tracy runtime, exports it
        lib/hexagon/libhexagon_dsp_skel.so   the DSP side
        lib/libc++_shared.so         NDK C++ runtime

  * `//integration_tests/device/tools:device_tools_aarch64_android`,
    which adds `bin/limit_lifetime` and `bin/rsync`. A separate CI artifact, so
    that they stay out of the release.

Layout on the device, under one root (default `/data/local/tmp/hexagon_it`):

    runs/<id>/bin,lib/       the plain runtime and the helpers
    runs/<id>-tracy/         the same for the Tracy runtime
    runs/<id>*/cases/        one directory per (model, compile case)
    modules/<case>/module.vmfb   the last large module sent for each (model,
                             compile case), kept across runs

`<id>` is a hash of the user name on the host and the repository root, so each
checkout of each user has one directory per runtime. It is emptied when a run
starts, which also removes what an aborted earlier run from the same checkout
left behind, and removed at the end of the run. With `keep_device_dir`
(`--keep-device-dir`), it is neither emptied nor removed.

`modules/` is not removed: each module there is the basis that rsync diffs the
next build of the same case against, so a rebuilt module only sends the bytes
that changed. It holds one
module per (model, compile case) that ever ran with a module large enough to be
kept (`_KEEP_MODULE_MIN_BYTES` in `stages/running.py`), and can be deleted at
any time, at the cost of one full transfer per case.

There is no lock: this suite is meant for one runner executing serially.
"""

from __future__ import annotations

import getpass
import hashlib
import shlex
from dataclasses import dataclass
from pathlib import Path

from integration_tests.device import adb

# FARF mask enabling all five DSP debug flags. The file has to be named after
# the executable and live in the directory named by DSP_LIBRARY_PATH, which is
# also where the skel lives.
_FARF_MASK = "0x1f"

_ENTRY_BINARIES = ("iree-run-module", "iree-benchmark-module", "iree-run-llm")

DEFAULT_DEVICE_ROOT = "/data/local/tmp/hexagon_it"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def default_run_id(has_tracy: bool = False) -> str:
    """A hash of the host's user name and the repository root, plus `-tracy`
    for the Tracy runtime, so the two deployments of a session do not collide.
    """
    key = f"{getpass.getuser()}\0{_REPO_ROOT}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"{digest}-tracy" if has_tracy else digest


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
    def benchmark_module(self) -> str:
        return f"{self.run_dir}/bin/iree-benchmark-module"

    @property
    def run_llm(self) -> str:
        return f"{self.run_dir}/bin/iree-run-llm"

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
    def rsync(self) -> str:
        return f"{self.run_dir}/bin/rsync"

    @property
    def dsp_library_path(self) -> str:
        return f"{self.run_dir}/lib/hexagon"

    def case_dir(self, case_dir_name: str) -> str:
        return f"{self.run_dir}/cases/{case_dir_name}"

    def kept_module(self, case_dir_name: str) -> str:
        """Where the case's module lives across runs, see the module docstring."""
        return f"{self.root}/modules/{case_dir_name}/module.vmfb"


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
    keep_device_dir: bool = False,
) -> Deployment:
    """Unpack the runtime and the helpers into this run's directory, emptied
    first unless `keep_device_dir` is set.

    `has_tracy` says whether `runtime_zip` is the Tracy runtime.
    """
    deployment = Deployment(
        root=root,
        run_id=run_id or default_run_id(has_tracy),
        has_tracy=has_tracy,
    )
    run_dir = shlex.quote(deployment.run_dir)
    setup_dir_cmds: list[str] = []
    if not keep_device_dir:
        setup_dir_cmds.append(f"rm -rf {run_dir}")
    setup_dir_cmds.append(f"mkdir -p {run_dir}")
    adb.shell(" && ".join(setup_dir_cmds))
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
