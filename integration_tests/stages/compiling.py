# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The Hexagon compilation configurations exercised by these tests."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

BASE_FLAGS: tuple[str, ...] = (
    "--iree-hal-target-device=hexagon",
    "--iree-input-type=auto",
    "--iree-hexagon-v=79",
    "--iree-hexagon-features=+hvxv79,+hvx-length128b",
    "--iree-opt-data-tiling=false",
    "--iree-stream-resource-min-offset-alignment=128",
)


@dataclasses.dataclass(frozen=True)
class CompileCase:
    name: str
    extra_flags: tuple[str, ...] = ()

    def flags(self, model_flags: tuple[str, ...] = ()) -> list[str]:
        return [*BASE_FLAGS, *self.extra_flags, *model_flags]


HEXAGON_DEFAULT_COMPILE_CASES: tuple[CompileCase, ...] = (
    # No selector flag: routes through the generic upstream LLVMCPU
    # lowering-strategy selector, which is iree-compile's default.
    CompileCase(name="default"),
    CompileCase(
        name="new-tiling-heuristics",
        extra_flags=("--iree-hexagon-launch-config-selector=hexagon",),
    ),
    CompileCase(
        name="new-tiling-heuristics-vtcm",
        extra_flags=(
            "--iree-hexagon-launch-config-selector=hexagon",
            "--iree-hexagon-enable-vtcm-tiling",
        ),
    ),
)

CASE_NAMES = tuple(case.name for case in HEXAGON_DEFAULT_COMPILE_CASES)


class CompilationError(RuntimeError):
    """Raised when iree-compile fails. Carries the log, for reason matching."""

    def __init__(self, message: str, log: str = "") -> None:
        super().__init__(message)
        self.log = log


def linker_flags(linker: Path) -> list[str]:
    """Point both linkers at `linker` instead of letting them search PATH.

    Two are needed. The Hexagon target links the DSP shared object, and a
    test also emits an llvm-cpu variant which links its own embedded
    ELF. Both look for `lld`/`ld.lld` on PATH by default, which is fine in a
    shell and not fine under `bazel test`, where PATH is trimmed to
    /usr/local/bin:/usr/bin:/bin and an LLVM installed under /usr/lib/llvm-19
    is invisible. The symptom otherwise is
    "required embedded linker tool (typically `lld`) not found".
    """
    return [
        f"--iree-hexagon-linker-path={linker}",
        f"--iree-llvmcpu-embedded-linker-path={linker}",
    ]


def compile_model(
    iree_compile: Path,
    model_mlir: Path,
    output_vmfb: Path,
    compile_case: CompileCase,
    log_file: Path | None = None,
    linker: Path | None = None,
    model_flags: tuple[str, ...] = (),
) -> Path:
    """Compile `model_mlir` for Hexagon, returning the vmfb path."""
    output_vmfb.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        str(iree_compile),
        str(model_mlir),
        "-o",
        str(output_vmfb),
        *compile_case.flags(model_flags),
        *(linker_flags(linker) if linker is not None else []),
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    log = result.stdout + result.stderr
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(log)
    if result.returncode != 0:
        detail = f"see {log_file}" if log_file is not None else log[-4000:]
        raise CompilationError(
            f"iree-compile failed with exit code {result.returncode} for "
            f"{model_mlir} (case '{compile_case.name}'): {detail}",
            log=log,
        )
    if not output_vmfb.exists():
        raise CompilationError(
            f"iree-compile reported success but produced no {output_vmfb}", log=log
        )
    return output_vmfb
