# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Minimal ADB wrapper for the Hexagon on-device tests.

The adb server is assumed to be up already, with one device.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


class AdbError(RuntimeError):
    pass


def _run(
    args: list[str], timeout: float = 600.0, check: bool = True
) -> subprocess.CompletedProcess[str]:
    argv = ["adb", *args]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as err:
        raise AdbError("adb is not on PATH") from err
    if check and result.returncode != 0:
        raise AdbError(
            f"{shlex.join(argv)} failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def available() -> tuple[bool, str]:
    """Whether a device is usable, and what adb said if it is not."""
    try:
        result = _run(["get-state"], timeout=120.0, check=False)
    except AdbError as err:
        return False, str(err)
    state = result.stdout.strip()
    if result.returncode == 0 and state == "device":
        return True, state
    return False, (result.stderr.strip() or state or "adb get-state said nothing")


def shell(script: str, timeout: float = 600.0) -> str:
    """Run a shell script on the device, raising if it fails."""
    return _run(["shell", f"set -eu; {script}"], timeout=timeout).stdout


def shell_exit_code(script: str, timeout: float = 600.0) -> tuple[int, str]:
    """Run a script whose failure is data, not an error."""
    result = _run(["shell", script], timeout=timeout, check=False)
    return result.returncode, result.stdout + result.stderr


def push(local_paths: list[Path], remote_dir: str) -> None:
    _run(["push", *[str(p) for p in local_paths], remote_dir])


def pull(remote_paths: list[str], local_dir: Path) -> None:
    _run(["pull", *remote_paths, str(local_dir)])


def logcat_clear() -> None:
    _run(["logcat", "-c"])


def logcat_dump(filters: list[str] | None = None) -> str:
    """Dump the log buffer collected since the last `logcat_clear`."""
    args = ["logcat", "-d"]
    args += filters if filters is not None else ["-s", "adsprpc"]
    try:
        return _run(args, timeout=120.0).stdout
    except AdbError as err:  # a missing log must not mask the real failure
        return f"<could not read logcat: {err}>"


def getprop(name: str) -> str:
    return _run(["shell", f"getprop {shlex.quote(name)}"]).stdout.strip()


def describe_device() -> str:
    """A one-line identification of whatever we are talking to."""
    return f"{getprop('ro.product.model')} (platform {getprop('ro.board.platform')})"
