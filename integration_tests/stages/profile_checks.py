# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks on a captured DSP profile, selected per model in `model.yaml`.

A check is a function registered with `@profile_check("<name>")`, taking the
`Profile` plus the `args` the `model.yaml` entry gives it, and raising
`ProfileError` if the profile does not show what it should.
"""

from __future__ import annotations

import csv
import dataclasses
import inspect
import re
from collections.abc import Callable
from pathlib import Path

# The names of the DSP zones other than kernels, see HEXAGON_PROFILER_ZONES in
# plugins/runtime/hexagon/arm_dsp/profiler.h.
_NON_KERNEL_ZONE_NAMES = frozenset(
    {
        "DSP execution",
        "Schedule",
        "Barrier",
        "Copy",
        "Fill",
        "Mem Management",
        "Marker",
        "Unknown",
    }
)

# The per-thread compute loop zones carry the dispatch name after this prefix,
# and the thread pool join zone has a fixed name, see
# plugins/compiler/hexagon/CodeGen/Transforms/InsertProfilerMarkers.cpp.
_INNER_LOOP_MARKER_PREFIX = "inner:"
_THREAD_JOIN_MARKER_NAME = "thread.join"


def _is_kernel_zone_name(name: str) -> bool:
    return (
        name not in _NON_KERNEL_ZONE_NAMES
        and name != _THREAD_JOIN_MARKER_NAME
        and not name.startswith(_INNER_LOOP_MARKER_PREFIX)
    )


class ProfileError(AssertionError):
    pass


@dataclasses.dataclass(frozen=True)
class Zone:
    name: str
    start_ns: int
    duration_ns: int

    @property
    def end_ns(self) -> int:
        return self.start_ns + self.duration_ns

    def overlaps(self, other: Zone) -> bool:
        """Whether both run at some instant. Touching ends do not overlap."""
        return self.start_ns < other.end_ns and other.start_ns < self.end_ns

    def __str__(self) -> str:
        return f"{self.name} [{self.start_ns}, {self.end_ns}) ns"


@dataclasses.dataclass(frozen=True)
class Profile:
    zones: tuple[Zone, ...]

    @classmethod
    def from_csv(cls, path: Path) -> Profile:
        """Read the output of `tracy-csvexport -g`."""
        with path.open(newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            raise ProfileError(f"{path}: empty profile export")
        # name, src_file, Time from start of program, GPU execution time
        zones = [
            Zone(name=row[0], start_ns=int(row[2]), duration_ns=int(row[3]))
            for row in rows[1:]
            if row
        ]
        return cls(tuple(sorted(zones, key=lambda z: (z.start_ns, z.end_ns))))

    def kernels(self) -> list[Zone]:
        return [z for z in self.zones if _is_kernel_zone_name(z.name)]

    def select(self, pattern: str) -> list[Zone]:
        """The kernels whose name matches `pattern`, at least one of them.

        Raising on no match keeps a stale or mistyped pattern from passing.
        """
        selected = [z for z in self.kernels() if re.search(pattern, z.name)]
        if not selected:
            names = sorted({z.name for z in self.kernels()})
            raise ProfileError(
                f"no kernel matches {pattern!r}; the kernels are: {names}"
            )
        return selected


def max_concurrency(zones: list[Zone]) -> int:
    """The most zones running at the same instant."""
    # At equal times the end sorts first, so touching zones do not overlap.
    events = sorted(
        # +1 when a new dispatch begins, -1 when it ends
        # -1 is sorted before +1, so a zone ending and a new one starting at the same time only count as 1 live zone
        [(z.start_ns, 1) for z in zones] + [(z.end_ns, -1) for z in zones]
    )
    running = peak = 0
    for _, delta in events:
        running += delta
        peak = max(peak, running)
    return peak


ProfileCheck = Callable[..., None]

PROFILE_CHECKS: dict[str, ProfileCheck] = {}


def profile_check(name: str) -> Callable[[ProfileCheck], ProfileCheck]:
    def register(fn: ProfileCheck) -> ProfileCheck:
        PROFILE_CHECKS[name] = fn
        return fn

    return register


def validate_check_args(name: str, args: dict[str, object]) -> None:
    """Raise TypeError unless `args` fit the parameters of check `name`."""
    inspect.signature(PROFILE_CHECKS[name]).bind(None, **args)


def run_check(name: str, profile: Profile, args: dict[str, object]) -> None:
    PROFILE_CHECKS[name](profile, **args)


def _overlapping(zones_a: list[Zone], zones_b: list[Zone]) -> list[tuple[Zone, Zone]]:
    """The pairs of distinct zones that overlap, the first from `zones_a`, the
    second from `zones_b`.

    One pass over the zones in start order, keeping those still running.
    """
    in_a = {id(z) for z in zones_a}
    in_b = {id(z) for z in zones_b}
    # A zone in both lists is visited once.
    zones = sorted(
        {id(z): z for z in zones_a + zones_b}.values(),
        key=lambda z: (z.start_ns, z.end_ns),
    )
    pairs = []
    running: list[Zone] = []
    for zone in zones:
        running = [r for r in running if r.end_ns > zone.start_ns]
        for r in running:
            if id(r) in in_a and id(zone) in in_b:
                pairs.append((r, zone))
            elif id(zone) in in_a and id(r) in in_b:
                pairs.append((zone, r))
        running.append(zone)
    return pairs


def _listing(zones: list[Zone]) -> str:
    return "\n".join(f"  {z}" for z in zones)


# The checks currently available
@profile_check("overlap")
def overlap(profile: Profile, a: str, b: str) -> None:
    """A kernel matching `a` ran at the same time as another matching `b`."""
    zones_a, zones_b = profile.select(a), profile.select(b)
    if not _overlapping(zones_a, zones_b):
        raise ProfileError(
            f"no kernel matching {a!r} overlaps one matching {b!r}:\n"
            + f"matching {a!r}:\n{_listing(zones_a)}\n"
            + f"matching {b!r}:\n{_listing(zones_b)}"
        )


@profile_check("no_overlap")
def no_overlap(profile: Profile, a: str, b: str) -> None:
    """No kernel matching `a` ran at the same time as any matching `b`."""
    pairs = _overlapping(profile.select(a), profile.select(b))
    if pairs:
        raise ProfileError(
            f"kernels matching {a!r} overlap kernels matching {b!r}:\n"
            + "\n".join(f"  {za}\n  overlaps {zb}" for za, zb in pairs)
        )


@profile_check("max_kernel_concurrency")
def max_kernel_concurrency(profile: Profile, max: int) -> None:
    """At no instant ran more than `max` kernels."""
    peak = max_concurrency(profile.kernels())
    if peak > max:
        raise ProfileError(
            f"up to {peak} kernels ran at the same time, expected at most {max}:\n"
            + _listing(profile.kernels())
        )


@profile_check("min_kernel_concurrency")
def min_kernel_concurrency(profile: Profile, min: int) -> None:
    """At some instant ran at least `min` kernels."""
    peak = max_concurrency(profile.kernels())
    if peak < min:
        raise ProfileError(
            f"at most {peak} kernels ran at the same time, expected at least {min}:\n"
            + _listing(profile.kernels())
        )


@profile_check("kernel_count")
def kernel_count(profile: Profile, count: int) -> None:
    """Exactly `count` kernel zones ran."""
    kernels = profile.kernels()
    if len(kernels) != count:
        raise ProfileError(
            f"{len(kernels)} kernels ran, expected {count}:\n" + _listing(kernels)
        )
