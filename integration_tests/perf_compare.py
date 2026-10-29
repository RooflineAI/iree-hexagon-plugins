# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare the run times of a test session against the latest main runs.

Reads the pytest output of `pytest integration_tests/ -v -rA`.
The recent runs are the same job of the latest completed main runs, read from
their GitHub logs with `gh`. The newest of them is the baseline every case is
compared against; all of them give the min-max spread shown next to it.
The thresholds to be used are set as cmd line arguments.
Exits 1 when a case is slower than the fail threshold, 0 otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from tabulate import tabulate

from integration_tests.stages.running import RUN_CASES

# GitHub prefixes every line of a job log with "<time> ", and `gh run view
# --log` with "<job>\t<step>\t<time> "; the API's log starts with a BOM.
_GITHUB_PREFIX = re.compile(r"^﻿?(?:[^\t]*\t[^\t]*\t)?\d{4}-\d\d-\d\dT\S+Z ")
_SECTION = re.compile(r"^=+ (.+?) =+$")
_TEST_HEADER = re.compile(r"^_+ test_model_on_device\[(.+)\] _+$")
# Printed by test_models.py under "timings:".
_RUN_TIME = re.compile(r"^  run: ([0-9.]+)s$")
_PASSED = re.compile(r"^PASSED \S+::test_model_on_device\[(.+)\]$")


def _is_timed(case_id: str) -> bool:
    """Whether the run time of a test id is comparable: Tracy and malloc debug might distort it."""
    # Longest name first: "parallel-profile" also ends with "-profile".
    for run_case in sorted(RUN_CASES, key=lambda case: len(case.name), reverse=True):
        if case_id.endswith(f"-{run_case.name}"):
            return not (run_case.profile or run_case.debug_malloc)
    return False


class LogParseError(ValueError):
    """The log is not the output of a single pytest session."""


def parse_log(text: str) -> dict[str, float]:
    """The run times in seconds of the passed, timed tests in pytest's `-rA`
    output.

    A failed test prints its timings too, in FAILURES; only the PASSES section
    is read, and a test only counts when the short summary says PASSED.

    Raises LogParseError when a test id has a second run time, e.g. because
    the log holds more than one session, and when timed tests passed but none
    has a run time: then the output format changed under this parser.
    """
    section = None
    case_id = None
    in_timings = False
    run_times: dict[str, float] = {}
    passed: set[str] = set()
    for raw_line in text.splitlines():
        line = _GITHUB_PREFIX.sub("", raw_line, count=1).rstrip()
        if match := _SECTION.match(line):
            section = match.group(1)
            case_id = None
            continue
        if section == "PASSES":
            if match := _TEST_HEADER.match(line):
                case_id = match.group(1)
                in_timings = False
            elif line == "timings:":
                in_timings = case_id is not None
            elif in_timings and (match := _RUN_TIME.match(line)):
                if case_id in run_times:
                    raise LogParseError(f"second run time for {case_id}")
                run_times[case_id] = float(match.group(1))
                in_timings = False
        elif section == "short test summary info":
            if match := _PASSED.match(line):
                passed.add(match.group(1))
    timed = {case for case in passed if _is_timed(case)}
    if timed and not run_times:
        raise LogParseError(
            "no run time found for any passed test; did the 'timings:' output "
            "of test_models.py change?"
        )
    return {case: run_times[case] for case in timed if case in run_times}


def _parse_file(path: Path) -> dict[str, float]:
    try:
        return parse_log(path.read_text(errors="replace"))
    except LogParseError as err:
        raise LogParseError(f"{path}: {err}") from err


@dataclasses.dataclass(frozen=True)
class CIRun:
    # The GitHub run id, or the log file it was read from.
    label: str
    run_times: dict[str, float]


def _gh_api(endpoint: str, *args: str) -> str:
    """One GitHub REST request"""
    # using gh run view sends several requests, of which we are only interested in one
    return subprocess.run(
        ["gh", "api", f"repos/{{owner}}/{{repo}}/actions/{endpoint}", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def fetch_recent_runs(
    job: str, workflow: str, branch: str, count: int, runs_searched: int
) -> list[CIRun]:
    """fetch `job` in the latest `count` completed push runs of `branch` that have
    run times, newest first and parses their log. Only the latest `runs_searched` runs are looked
    at.

    The run executing this is still in progress, so it is never among them.
    """
    print(f"Fetching up to {runs_searched} runs of {job} on {branch} from GitHub")
    # fetch latest runs for branch
    runs = json.loads(
        _gh_api(
            f"workflows/{workflow}/runs?branch={branch}&event=push"
            f"&status=completed&per_page={runs_searched}"
        )
    )["workflow_runs"]
    recent_runs = []
    for run in runs:
        run_id = str(run["id"])
        # fetch jobs of the run
        jobs = json.loads(_gh_api(f"runs/{run_id}/jobs?per_page=100"))["jobs"]
        # look up the job id. There is only 1 job with matching name
        job_ids = [entry["id"] for entry in jobs if entry["name"] == job]
        if len(job_ids) != 1:
            raise LookupError(
                f"{branch} run {run_id} has {len(job_ids)} jobs named {job}"
            )
        (job_id,) = job_ids
        try:
            # and fetch the logs
            # The log holds the colour codes of the output.
            log = _gh_api(f"jobs/{job_id}/logs", "--allow-escape-sequences")
        except subprocess.CalledProcessError:
            # Skipped jobs and expired logs have nothing to download.
            continue
        try:
            run_times = parse_log(log)
        except LogParseError as err:
            raise LogParseError(f"{branch} run {run_id}: {err}") from err
        if run_times:
            recent_runs.append(CIRun(run_id, run_times))
        if len(recent_runs) == count:
            break
    return recent_runs


def read_recent_runs_from_files(paths: Sequence[Path], count: int) -> list[CIRun]:
    """The logs at `paths`, newest first, as recent runs; the ones without run
    times are skipped, like in `fetch_recent_runs`."""
    recent_runs = [CIRun(str(path), _parse_file(path)) for path in paths[:count]]
    return [run for run in recent_runs if run.run_times]


@dataclasses.dataclass(frozen=True)
class Thresholds:
    warn_percent: float
    fail_percent: float
    min_delta_s: float  # everything change below is considered noise


class Verdict(enum.StrEnum):
    FAIL = "FAIL"
    WARN = "WARN"
    OK = "ok"
    FASTER = "faster"
    NEW = "new"
    # Timed in the baseline but not in this session: removed, renamed, failed
    # or no longer comparable.
    MISSING = "MISSING"


@dataclasses.dataclass(frozen=True)
class Row:
    case_id: str
    # None for a case this session has no time for.
    current: float | None
    # None for a case the baseline does not have.
    baseline: float | None
    # Min and max over the recent runs that have the case.
    spread: tuple[float, float] | None
    verdict: Verdict

    @property
    def percent(self) -> float | None:
        if self.baseline is None or self.current is None:
            return None
        return 100.0 * (self.current - self.baseline) / self.baseline


def _verdict(
    current: float | None, baseline: float | None, thresholds: Thresholds
) -> Verdict:
    if current is None:
        return Verdict.MISSING
    if baseline is None:
        return Verdict.NEW
    delta = current - baseline
    percent = 100.0 * delta / baseline
    if abs(delta) < thresholds.min_delta_s:
        return Verdict.OK
    if percent >= thresholds.fail_percent:
        return Verdict.FAIL
    if percent >= thresholds.warn_percent:
        return Verdict.WARN
    if percent <= -thresholds.warn_percent:
        return Verdict.FASTER
    return Verdict.OK


# build comparison table
def compare(
    current: dict[str, float],
    recent_runs: Sequence[CIRun],
    thresholds: Thresholds,
) -> list[Row]:
    """`recent_runs` first the baseline, others are used to measure the usual spread."""
    baseline_times = recent_runs[0].run_times
    rows = []
    for case_id in sorted(current.keys() | baseline_times.keys()):
        history = [
            run.run_times[case_id] for run in recent_runs if case_id in run.run_times
        ]
        seconds = current.get(case_id)
        baseline = baseline_times.get(case_id)
        rows.append(
            Row(
                case_id=case_id,
                current=seconds,
                baseline=baseline,
                spread=(min(history), max(history)) if history else None,
                verdict=_verdict(seconds, baseline, thresholds),
            )
        )
    return rows


_TABLE_HEADER = ("case", "baseline", "spread", "current", "change", "verdict")
_TABLE_ALIGN = ("left", "right", "right", "right", "right", "left")


def _format_table(rows: Sequence[Row], recent_runs: Sequence[CIRun]) -> str:
    """A plain-text table with aligned columns, for the terminal and log."""
    cells = [
        (
            row.case_id,
            "-" if row.baseline is None else f"{row.baseline:.2f}s",
            "-" if row.spread is None else f"{row.spread[0]:.2f}-{row.spread[1]:.2f}s",
            "-" if row.current is None else f"{row.current:.2f}s",
            "-" if row.percent is None else f"{row.percent:+.1f}%",
            str(row.verdict),
        )
        for row in rows
    ]
    title = (
        f"Performance against the baseline, main run {recent_runs[0].label} "
        f"(host wall-clock of iree-run-module; spread over the last "
        f"{len(recent_runs)} main run(s))"
    )
    # The cells are formatted already; tabulate must not reparse them.
    table = tabulate(cells, _TABLE_HEADER, colalign=_TABLE_ALIGN, disable_numparse=True)
    return f"{title}\n\n{table}\n"


def _annotate(rows: Sequence[Row], baseline: CIRun) -> None:
    """GitHub annotations for the rows that changed or are missing."""
    for row in rows:
        if row.verdict is Verdict.MISSING:
            print(
                f"::notice title=performance case missing::{row.case_id}: timed in "
                f"the baseline {baseline.label} ({row.baseline:.2f}s), but not "
                "in this run (removed, renamed or failed?)"
            )
        elif row.verdict in (Verdict.FAIL, Verdict.WARN):
            level = "error" if row.verdict is Verdict.FAIL else "warning"
            print(
                f"::{level} title=performance regression::{row.case_id}: "
                f"{row.baseline:.2f}s -> {row.current:.2f}s ({row.percent:+.1f}%)"
            )
        elif row.verdict is Verdict.FASTER:
            print(
                f"::notice title=performance improvement::{row.case_id}: "
                f"{row.baseline:.2f}s -> {row.current:.2f}s ({row.percent:+.1f}%)"
            )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--current", type=Path, required=True, help="pytest output of this session"
    )
    parser.add_argument(
        "--baseline-log",
        type=Path,
        action="append",
        default=[],
        help="pytest output or GitHub job log to compare against (repeatable): "
        "the first is the baseline, all give the spread;"
        " skips fetching main runs with gh",
    )
    parser.add_argument(
        "--job",
        default=os.environ.get("GITHUB_JOB"),
        help="job to compare against (test-bazel or test-cmake) (default: $GITHUB_JOB) ",
    )
    parser.add_argument("--workflow", default="build.yml")
    parser.add_argument("--branch", default="main")
    parser.add_argument(
        "--num-recent-runs",
        type=int,
        required=True,
        help="how many main runs give the spread; the newest is the baseline",
    )
    parser.add_argument(
        "--max-runs-searched",
        type=int,
        required=True,
        help="how many runs to look through for --num-recent-runs to find completed runs",
    )
    parser.add_argument("--warn-percent", type=float, required=True)
    parser.add_argument("--fail-percent", type=float, required=True)
    parser.add_argument(
        "--min-delta-s",
        type=float,
        required=True,
        help="a case is only flagged when it is slower by both the percentage "
        "and this many seconds",
    )
    args = parser.parse_args(argv)
    if not args.baseline_log and args.job is None:
        parser.error("--job (or $GITHUB_JOB) is required to fetch main runs")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        current = _parse_file(args.current)
        if not current:
            print(
                f"::warn title=performance check::{args.current}: no timed test found"
            )
            return 0
        if args.baseline_log:
            recent_runs = read_recent_runs_from_files(
                args.baseline_log, args.num_recent_runs
            )
        else:
            recent_runs = fetch_recent_runs(
                args.job,
                args.workflow,
                args.branch,
                args.num_recent_runs,
                args.max_runs_searched,
            )
    except (LogParseError, LookupError) as err:
        print(f"::error title=performance check::{err}")
        return 1
    except subprocess.CalledProcessError as err:
        print(f"::error title=performance check::gh failed: {err.stderr.strip()}")
        return 1
    if not recent_runs:
        print(
            f"::error title=performance check::no {args.branch} run with run times found"
        )
        return 1

    thresholds = Thresholds(
        warn_percent=args.warn_percent,
        fail_percent=args.fail_percent,
        min_delta_s=args.min_delta_s,
    )
    rows = compare(current, recent_runs, thresholds)
    print(_format_table(rows, recent_runs))
    _annotate(rows, recent_runs[0])
    return 1 if any(row.verdict is Verdict.FAIL for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
