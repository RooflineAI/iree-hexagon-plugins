#!/usr/bin/env python3
"""Summarize per-region register pressure from a get_reg_pressure.sh log.

Parses the `-debug-only=machine-scheduler` trace (one or more dispatches,
each preceded by an "=== <file> ===" header from get_reg_pressure.sh) and
reports, per dispatch and per pressure set, the worst-case ("Max Pressure")
value MachineScheduler saw across all of its scheduling regions.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from collections.abc import Generator
from io import TextIOWrapper

FILE_RE = re.compile(r"^=== (?P<file>.+) ===$")
REGION_RE = re.compile(
    r"^\*+ MI Converging Scheduling VLIW (?P<bb>\S+)\s+in_func "
    r"(?P<func>\S+) at loop depth (?P<depth>\d+)"
)
MAX_PRESSURE_RE = re.compile(r"^Max Pressure:\s*(?P<sets>.*)$")
PRESSURE_SET_RE = re.compile(r"(?P<name>[A-Za-z0-9_]+)=(?P<value>\d+)")


def parse(
    lines: TextIOWrapper,
) -> Generator[tuple[str | None, str, str, int, dict[str, int]]]:
    """Yields one record per scheduling region: (file, func, bb, depth, {set: value})."""
    current_file = None
    pending_region = None
    for line in lines:
        line = line.rstrip("\n")
        m = FILE_RE.match(line)
        if m:
            current_file = m.group("file")
            continue
        m = REGION_RE.match(line)
        if m:
            pending_region = (m.group("bb"), m.group("func"), int(m.group("depth")))
            continue
        m = MAX_PRESSURE_RE.match(line)
        if m and pending_region is not None:
            pressures = {
                pm.group("name"): int(pm.group("value"))
                for pm in PRESSURE_SET_RE.finditer(m.group("sets"))
            }
            bb, func, depth = pending_region
            yield current_file, func, bb, depth, pressures
            pending_region = None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logfile", help="pressure log produced by get_reg_pressure.sh")
    ap.add_argument(
        "--per-region",
        action="store_true",
        help="also print every individual scheduling region, not just the peak",
    )
    ap.add_argument(
        "--csv",
        metavar="PATH",
        help="additionally write one row per (file, pressure set) peak to this CSV",
    )
    args = ap.parse_args(argv[1:])

    with open(args.logfile) as f:
        regions = list(parse(f))

    if not regions:
        print(
            "no 'Max Pressure:' regions found -- was this log produced with "
            "-debug-only=machine-scheduler on an assertions-enabled llc?",
            file=sys.stderr,
        )
        return 1

    # file -> pressure set -> (max value, bb, func, depth)
    peaks = defaultdict(dict)
    by_file = defaultdict(list)
    for file, func, bb, depth, pressures in regions:
        by_file[file].append((func, bb, depth, pressures))
        for name, value in pressures.items():
            best = peaks[file].get(name)
            if best is None or value > best[0]:
                peaks[file][name] = (value, bb, func, depth)

    for file, region_list in by_file.items():
        print(f"=== {file} ===")
        if args.per_region:
            for func, bb, depth, pressures in region_list:
                sets = " ".join(f"{k}={v}" for k, v in sorted(pressures.items()))
                print(f"  {func} {bb} (loop depth {depth}): {sets}")
        print(" peak pressure across all regions:")
        for name, (value, bb, func, depth) in sorted(
            peaks[file].items(), key=lambda kv: -kv[1][0]
        ):
            print(f"    {name:<16} {value:>3}   (in {func} {bb}, loop depth {depth})")
        print()

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pressure_set", "max_value", "func", "bb", "loop_depth"])
            for file, sets in peaks.items():
                for name, (value, bb, func, depth) in sets.items():
                    writer.writerow([name, value, func, bb, depth])
        print(f"wrote {args.csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
