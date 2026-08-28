# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numeric comparison of a device output against a reference.

Reports both a relative L2 norm (which tolerates a few
badly wrong elements in a large tensor) and an element-wise atol/rtol check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class AccuracyError(AssertionError):
    pass


@dataclass(frozen=True)
class Comparison:
    name: str
    relative_l2_percent: float
    max_abs_diff: float
    elementwise_close: bool
    report: str


def _top_offenders(reference: np.ndarray, actual: np.ndarray, count: int) -> str:
    diff = np.abs(actual - reference)
    if diff.size == 0:
        return ""
    flat = np.argsort(diff, axis=None)[::-1][:count]
    lines = []
    for flat_index in flat:
        index = np.unravel_index(flat_index, diff.shape)
        lines.append(
            f"    {tuple(int(i) for i in index)}: "
            f"reference={reference[index]:.6g} actual={actual[index]:.6g} "
            f"absdiff={diff[index]:.6g}"
        )
    return "\n".join(lines)


def compare_arrays(
    name: str,
    reference: np.ndarray,
    actual: np.ndarray,
    atol: float,
    rtol: float,
    top_k: int = 5,
) -> Comparison:
    if reference.shape != actual.shape:
        raise AccuracyError(
            f"{name}: shape mismatch, reference {reference.shape} vs actual "
            f"{actual.shape}"
        )
    # Compared in float64 so that the metrics themselves cannot overflow or
    # lose precision for f16 outputs.
    ref = reference.astype(np.float64)
    act = actual.astype(np.float64)
    ref_norm = float(np.linalg.norm(ref))
    error_norm = float(np.linalg.norm(act - ref))
    relative = (error_norm / ref_norm * 100.0) if ref_norm > 0 else 100.0
    max_abs = float(np.abs(act - ref).max())
    close = bool(np.allclose(act, ref, atol=atol, rtol=rtol, equal_nan=False))
    report = (
        f"{name}: relative L2 error {relative:.4f}%, max abs diff {max_abs:.6g}, "
        f"allclose(atol={atol:g}, rtol={rtol:g}) = {close}"
    )
    if not close:
        offenders = _top_offenders(ref, act, top_k)
        if offenders:
            report += f"\n  worst {top_k} elements:\n{offenders}"
    return Comparison(
        name=name,
        relative_l2_percent=relative,
        max_abs_diff=max_abs,
        elementwise_close=close,
        report=report,
    )


def check_output_files(
    reference_files: list[Path],
    actual_files: list[Path],
    atol: float,
    rtol: float,
) -> None:
    """Compare every reference/actual pair, reporting all failures at once."""
    if len(reference_files) != len(actual_files):
        raise AccuracyError(
            f"expected {len(reference_files)} outputs, got {len(actual_files)}"
        )
    failures = []
    for reference_file, actual_file in zip(reference_files, actual_files):
        if not actual_file.exists():
            failures.append(f"{actual_file.name}: missing - the run produced no output")
            continue
        comparison = compare_arrays(
            actual_file.name,
            np.load(reference_file),
            np.load(actual_file),
            atol=atol,
            rtol=rtol,
        )
        print(comparison.report)
        if not comparison.elementwise_close:
            failures.append(comparison.report)
    if failures:
        raise AccuracyError(
            f"{len(failures)} of {len(reference_files)} outputs failed the "
            "accuracy check:\n" + "\n".join(failures)
        )
