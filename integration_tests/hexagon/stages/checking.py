# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numeric comparison of a device output against a reference."""

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
    max_relative_l2_percent: float | None = None,
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
        elif (
            max_relative_l2_percent is not None
            and comparison.relative_l2_percent > max_relative_l2_percent
        ):
            # Only reported separately when the element-wise check passed;
            # otherwise it would be the same failure counted twice.
            failures.append(
                f"{comparison.name}: relative L2 error "
                f"{comparison.relative_l2_percent:.4f}% exceeds the "
                f"{max_relative_l2_percent:g}% threshold, although every element "
                f"is within atol={atol:g}/rtol={rtol:g}"
            )
    if failures:
        raise AccuracyError(
            f"{len(failures)} of {len(reference_files)} outputs failed the "
            "accuracy check:\n" + "\n".join(failures)
        )


def check_predicted_label(
    output_file: Path,
    labels: dict[int, str] | None,
    expected_label: str,
) -> None:
    """Assert the device's argmax class is `expected_label`."""
    if labels is None:
        raise AccuracyError(
            "a semantic_check needs the model's config.id2label, and this "
            "import recorded none"
        )
    logits = np.load(output_file).astype(np.float64).reshape(-1)
    order = np.argsort(logits)[::-1]
    top, runner_up = int(order[0]), int(order[1]) if logits.size > 1 else None
    predicted = labels.get(top, f"<no label for class {top}>")
    margin = float(logits[top] - logits[runner_up]) if runner_up is not None else None
    print(
        f"{output_file.name}: predicted {predicted!r} (class {top})"
        + (
            f", runner-up {labels.get(runner_up, runner_up)!r} at a margin of "
            f"{margin:.4g}"
            if runner_up is not None
            else ""
        )
    )
    if predicted != expected_label:
        raise AccuracyError(
            f"{output_file.name}: expected the device to predict "
            f"{expected_label!r}, got {predicted!r} (class {top})"
        )
