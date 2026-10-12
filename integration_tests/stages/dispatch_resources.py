# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks of the per-dispatch resources the DSP runtime logs.

When it loads an executable, the DSP runtime logs the resources the compiler
recorded for each export (`HEXAGON-DISPATCH-RESOURCES`). After each dispatch it
logs a `HEXAGON-DISPATCH-RESOURCES-VIOLATION` if the dispatch allocated more
VTCM than declared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from integration_tests.modeltree.spec import DispatchResourcesCheck

_RESOURCES_RE = re.compile(
    r"HEXAGON-DISPATCH-RESOURCES name=(?P<name>\S+) valid=(?P<valid>\d) "
    r"hvx=(?P<hvx>\d) hmx=(?P<hmx>\d) vtcm_bytes=(?P<vtcm_bytes>\d+)"
)
_VIOLATION_RE = re.compile(
    r"HEXAGON-DISPATCH-RESOURCES-VIOLATION name=(?P<name>\S+) "
    r"declared=(?P<declared>\d+) peak=(?P<peak>\d+)"
)


class DispatchResourcesError(AssertionError):
    pass


@dataclass(frozen=True)
class DispatchResources:
    name: str
    valid: bool
    hvx: bool
    hmx: bool
    vtcm_bytes: int


@dataclass(frozen=True)
class VtcmViolation:
    name: str
    declared: int
    peak: int


def parse_dispatch_resources(logcat: str) -> list[DispatchResources]:
    """The resources logged for each export when its executable was loaded."""
    return [
        DispatchResources(
            name=match["name"],
            valid=match["valid"] == "1",
            hvx=match["hvx"] == "1",
            hmx=match["hmx"] == "1",
            vtcm_bytes=int(match["vtcm_bytes"]),
        )
        for match in _RESOURCES_RE.finditer(logcat)
    ]


def parse_vtcm_violations(logcat: str) -> list[VtcmViolation]:
    return [
        VtcmViolation(
            name=match["name"],
            declared=int(match["declared"]),
            peak=int(match["peak"]),
        )
        for match in _VIOLATION_RE.finditer(logcat)
    ]


def check_dispatch_resources(
    logcat: str,
    case_name: str,
    expectations: tuple[DispatchResourcesCheck, ...] = (),
) -> None:
    """Raise DispatchResourcesError if the logged resources are wrong.

    Every run is checked for:
    - at least one export logged,
    - every export carrying compiler-recorded resources (`valid=1`), unless
      the model's `allow_invalid` expectation accepts exports without them,
    - HMX implying HVX, and no HMX in the plain `hexagon` case,
    - no dispatch exceeding its declared VTCM.
    `expectations` adds the model's own checks for `case_name`.
    """
    resources = parse_dispatch_resources(logcat)
    expectations = tuple(e for e in expectations if e.case in ("*", case_name))
    allow_invalid = any(e.allow_invalid for e in expectations)
    problems: list[str] = []
    if not resources:
        problems.append("no HEXAGON-DISPATCH-RESOURCES line in the logcat")
    for entry in resources:
        if not entry.valid and not allow_invalid:
            problems.append(f"{entry.name}: no compiler-recorded resources")
        if entry.hmx and not entry.hvx:
            problems.append(f"{entry.name}: uses HMX but not HVX")
        if case_name == "hexagon" and entry.hmx:
            problems.append(f"{entry.name}: uses HMX in the plain 'hexagon' case")
    for violation in parse_vtcm_violations(logcat):
        problems.append(
            f"{violation.name}: allocated {violation.peak} bytes of VTCM but "
            f"declared {violation.declared}"
        )
    for expectation in expectations:
        if expectation.any_hmx is not None:
            any_hmx = any(entry.hmx for entry in resources)
            if any_hmx != expectation.any_hmx:
                problems.append(
                    f"expected any_hmx={expectation.any_hmx}, got {any_hmx}"
                )
    if problems:
        logged = "\n".join(
            f"  {entry.name}: valid={int(entry.valid)} hvx={int(entry.hvx)} "
            f"hmx={int(entry.hmx)} vtcm_bytes={entry.vtcm_bytes}"
            for entry in resources
        )
        raise DispatchResourcesError(
            "dispatch resources check failed:\n"
            + "\n".join(f"- {problem}" for problem in problems)
            + f"\nlogged resources:\n{logged or '  (none)'}"
        )
