# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Minimal ADB wrapper for the Hexagon on-device tests.

Deliberately thin. The stock `adb` client already honours ADB_SERVER_SOCKET and
ANDROID_SERIAL, which is exactly how CI reaches the phone rig (see
`.github/workflows/build.yml`), so there is nothing here about ADB keys,
per-host adb servers or connection types: if a phone shows up in a plain
`adb devices`, it is usable by these tests.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


class AdbError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdbDevice:
    """One reachable device, addressed by its adb serial."""

    serial: str

    def _argv(self, args: list[str]) -> list[str]:
        return ["adb", "-s", self.serial, *args]

    def _run(self, args: list[str], timeout: float = 600.0) -> str:
        argv = self._argv(args)
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise AdbError(
                f"{' '.join(shlex.quote(a) for a in argv)} failed with exit code "
                f"{result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result.stdout

    def shell(self, script: str, timeout: float = 600.0) -> str:
        """Run a shell script on the device, raising if it fails.

        `adb shell` has propagated the remote exit status since platform-tools
        24, which is what CI pins (37.0.1) - so a failing script surfaces here
        as an AdbError rather than being swallowed. Where the exit code is the
        thing under test rather than a precondition, use `shell_exit_code`.
        """
        return self._run(["shell", f"set -eu; {script}"], timeout=timeout)

    def shell_exit_code(
        self, script: str, exit_code_file: str, timeout: float = 600.0
    ) -> tuple[int, str]:
        """Run a script whose failure is data, not an error.

        The exit code is written to a file on the device and read back, rather
        than relying on adb propagating it, because the script itself must be
        allowed to fail without turning into an exception here. Returns the
        remote exit code and the combined stdout/stderr.
        """
        output = self._run(
            [
                "shell",
                f"{{ {script} ; }} >{shlex.quote(exit_code_file)}.log 2>&1; "
                f"echo $? >{shlex.quote(exit_code_file)}; "
                f"cat {shlex.quote(exit_code_file)}.log",
            ],
            timeout=timeout,
        )
        code = self._run(["shell", f"cat {shlex.quote(exit_code_file)}"]).strip()
        return int(code), output

    def push(self, local_paths: list[Path], remote_dir: str) -> None:
        self._run(["push", *[str(p) for p in local_paths], remote_dir])

    def pull(self, remote_paths: list[str], local_dir: Path) -> None:
        self._run(["pull", *remote_paths, str(local_dir)])

    def exists(self, remote_path: str) -> bool:
        result = subprocess.run(
            self._argv(["shell", f"test -e {shlex.quote(remote_path)}"]),
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def logcat_clear(self) -> None:
        self._run(["logcat", "-c"])

    def logcat_dump(self, filters: list[str] | None = None) -> str:
        """Dump the log buffer collected since the last `logcat_clear`.

        Defaults to the adsprpc tag, which is where the DSP side of the runtime
        reports - on a failing dispatch it is frequently the only diagnostic
        that says anything at all.
        """
        args = ["logcat", "-d"]
        args += filters if filters is not None else ["-s", "adsprpc"]
        try:
            return self._run(args, timeout=120.0)
        except AdbError as err:  # a missing log must not mask the real failure
            return f"<could not read logcat: {err}>"

    def property(self, name: str) -> str:
        return self._run(["shell", f"getprop {shlex.quote(name)}"]).strip()


def attached_devices() -> list[str]:
    """Serials of all devices in the `device` state, or [] if adb is absent."""
    try:
        output = subprocess.run(
            ["adb", "devices"], capture_output=True, text=True, timeout=120.0
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if output.returncode != 0:
        return []
    serials = []
    for line in output.stdout.splitlines()[1:]:
        fields = line.split()
        # Anything other than "device" (unauthorized, offline, no permissions)
        # is not usable, and the caller reports it as "no device" rather than
        # failing halfway through a push.
        if len(fields) >= 2 and fields[1] == "device":
            serials.append(fields[0])
    return serials
