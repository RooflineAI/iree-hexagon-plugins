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

import pytest

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

    def flags(self) -> list[str]:
        return [*BASE_FLAGS, *self.extra_flags]


HEXAGON_DEFAULT_COMPILE_CASES = [
    # No selector flag: routes through the generic upstream LLVMCPU
    # lowering-strategy selector, which is iree-compile's default.
    pytest.param(CompileCase(name="default"), id="default"),
    pytest.param(
        CompileCase(
            name="new_tiling_heuristics",
            extra_flags=("--iree-hexagon-launch-config-selector=hexagon",),
        ),
        id="new-tiling-heuristics",
    ),
    pytest.param(
        CompileCase(
            name="new-tiling-heuristics-vtcm",
            extra_flags=(
                "--iree-hexagon-launch-config-selector=hexagon",
                "--iree-hexagon-enable-vtcm-tiling",
            ),
        ),
        id="new-tiling-heuristics-vtcm",
    ),
]


class CompilationError(RuntimeError):
    pass


def linker_flags(linker: Path) -> list[str]:
    """Point both linkers at `linker` instead of letting them search PATH.

    Two are needed. The Hexagon target links the DSP shared object, and a
    Hexagon compile also emits an llvm-cpu variant which links its own embedded
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
) -> Path:
    """Compile `model_mlir` for Hexagon, returning the vmfb path.

    stdout/stderr go to `log_file` when given: an imported model's diagnostics
    quote the offending IR, and for a real model that is megabytes of text that
    should not land in the pytest report.
    """
    output_vmfb.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        str(iree_compile),
        str(model_mlir),
        "-o",
        str(output_vmfb),
        *compile_case.flags(),
        *(linker_flags(linker) if linker is not None else []),
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        detail = f"see {log_file}" if log_file is not None else result.stderr[-4000:]
        raise CompilationError(
            f"iree-compile failed with exit code {result.returncode} for "
            f"{model_mlir} (case '{compile_case.name}'): {detail}"
        )
    if not output_vmfb.exists():
        raise CompilationError(
            f"iree-compile reported success but produced no {output_vmfb}"
        )
    return output_vmfb


def cases_with_known_failures(
    known_failures: dict[str, str],
    raises: type[BaseException] | tuple[type[BaseException], ...] | None = None,
) -> list:
    """`HEXAGON_DEFAULT_COMPILE_CASES` with some cases marked as xfail.

    Keyed by `CompileCase.name`. Whether a case is expected to fail is a
    property of the *model*, not of the case, so this is applied per test
    rather than baked into the case list.

    `strict=True` on purpose: when the underlying problem is fixed, the
    unexpected pass fails the suite and forces this entry to be removed,
    instead of quietly staying an xfail forever.

    `raises` should always be given. Without it an xfail absorbs *any*
    exception, so a case expected to be numerically wrong would go on quietly
    "xfailing" after it stopped compiling at all - which is exactly what
    happened the first time this was run under Bazel.
    """
    marked = []
    for param in HEXAGON_DEFAULT_COMPILE_CASES:
        case = param.values[0]
        reason = known_failures.get(case.name)
        marks = list(param.marks)
        if reason is not None:
            marks.append(pytest.mark.xfail(reason=reason, strict=True, raises=raises))
        marked.append(pytest.param(case, id=param.id, marks=marks))
    return marked
