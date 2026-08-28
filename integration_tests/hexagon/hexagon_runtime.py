# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Deployment of the Hexagon runtime package onto a device.

The package is the `pkg_zip` built by
`//plugins/runtime/hexagon:hexagon_runtime_aarch64_android`, whose layout is
documented in `examples/running.md`:

    bin/iree-run-module          ARM host side, Hexagon HAL driver linked in
    bin/iree-benchmark-module    same, for benchmarking
    lib/hexagon/libhexagon_dsp_skel.so   the DSP side
    lib/libc++_shared.so         NDK C++ runtime
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from integration_tests.hexagon.adb_device import AdbDevice

# FARF mask enabling all five DSP debug flags. The file has to be named after
# the executable and live in the directory named by DSP_LIBRARY_PATH, which is
# also where the skel lives.
_FARF_MASK = "0x1f"

_ENTRY_BINARIES = ("iree-run-module", "iree-benchmark-module")


@dataclass(frozen=True)
class DeployedRuntime:
    """A runtime package unpacked in a directory on the device."""

    device: AdbDevice
    remote_dir: str

    @property
    def run_module(self) -> str:
        return f"{self.remote_dir}/bin/iree-run-module"

    @property
    def dsp_library_path(self) -> str:
        return f"{self.remote_dir}/lib/hexagon"


def deploy_runtime(
    device: AdbDevice, runtime_zip: Path, remote_dir: str
) -> DeployedRuntime:
    """Push and unpack the runtime package, replacing whatever was there.

    The previous bin/ and lib/ are removed first rather than unpacked over: a
    stale binary from an earlier build silently answering for the new one is a
    failure mode that costs hours to spot.
    """
    device.shell(
        f"rm -rf {remote_dir}/bin {remote_dir}/lib {remote_dir}/{runtime_zip.name}"
    )
    device.shell(f"mkdir -p {remote_dir}")
    device.push([runtime_zip], remote_dir + "/")
    farf_writes = " && ".join(
        f"printf '{_FARF_MASK}\\n' > lib/hexagon/{binary}.farf"
        for binary in _ENTRY_BINARIES
    )
    chmods = " ".join(f"bin/{binary}" for binary in _ENTRY_BINARIES)
    device.shell(
        f"cd {remote_dir} && unzip -oq {runtime_zip.name} && "
        f"chmod +x {chmods} && {farf_writes}"
    )
    return DeployedRuntime(device=device, remote_dir=remote_dir)
