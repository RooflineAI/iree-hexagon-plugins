# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare the run times of a test session against the latest main runs.

Reads the pytest output of `pytest integration_tests/ -v -rA`.
The baseline is the same job of the latest completed main run, read from its GitHub log with `gh`;
the main runs before it give the min-max spread shown next to it.
The thresholds to be used are set as cmd line arguments
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

from integration_tests.stages.running import RUN_CASES

# GitHub prefixes every line of a downloaded log with "<job>\t<step>\t<time> ".
_GITHUB_PREFIX = re.compile(r"^[^\t]*\t[^\t]*\t\d{4}-\d\d-\d\dT\S+Z ")
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


@dataclasses.dataclass(frozen=True)
class ParsedLog:
    # Run time in seconds of every passed, comparable test id.
    run_times: dict[str, float]
    # Every passed test id.
    passed: frozenset[str]


def parse_log(text: str) -> ParsedLog:
    """The run times of the passed tests in pytest's `-rA` output.

    A failed test prints its timings too, in FAILURES; only the PASSES section
    is read, and a test only counts when the short summary says PASSED.

    Raises LogParseError when a test id has a second run time, e.g. because
    the log holds more than one session.
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
    return ParsedLog(
        run_times={
            case: seconds
            for case, seconds in run_times.items()
            if case in passed and _is_timed(case)
        },
        passed=frozenset(passed),
    )


@dataclasses.dataclass(frozen=True)
class Thresholds:
    # Main runs shown as the spread; the newest is the baseline.
    baseline_runs: int
    warn_percent: float
    fail_percent: float
    # A case is only flagged when it is slower by both the percentage and this.
    min_delta_s: float


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
    spread: tuple[float, float] | None
    verdict: Verdict

    @property
    def percent(self) -> float | None:
        if self.baseline is None or self.current is None:
            return None
        return 100.0 * (self.current - self.baseline) / self.baseline


def _verdict(current: float, baseline: float, thresholds: Thresholds) -> Verdict:
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


def compare(
    current: dict[str, float],
    baselines: Sequence[dict[str, float]],
    thresholds: Thresholds,
) -> list[Row]:
    """`baselines` newest first; the newest is the baseline, all give the
    spread."""
    rows = []
    for case_id in sorted(current.keys() | baselines[0].keys()):
        history = [run[case_id] for run in baselines if case_id in run]
        seconds = current.get(case_id)
        baseline = baselines[0].get(case_id)
        if seconds is None:
            verdict = Verdict.MISSING
        elif baseline is None:
            verdict = Verdict.NEW
        else:
            verdict = _verdict(seconds, baseline, thresholds)
        rows.append(
            Row(
                case_id=case_id,
                current=seconds,
                baseline=baseline,
                spread=(min(history), max(history)) if history else None,
                verdict=verdict,
            )
        )
    return rows


def _gh(*args: str) -> str:
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout


def fetch_baselines(
    job: str, workflow: str, branch: str, count: int, runs_searched: int
) -> list[tuple[str, dict[str, float]]]:
    """The run times of `job` in the latest `count` completed push runs of
    `branch` that have any, newest first, labelled by run id. Only the latest
    `runs_searched` runs are looked at.

    The run executing this is still in progress, so it is never among them.
    """
    print(f"Fetch {runs_searched} Baseline Jobs from GitHub")
    runs = json.loads(
        _gh(
            "run", "list",
            "--workflow", workflow,
            "--branch", branch,
            "--event", "push",
            "--status", "completed",
            "--limit", str(runs_searched),
            "--json", "databaseId",
        )
    )  # fmt: skip
    baselines = []
    for run in runs:
        run_id = str(run["databaseId"])
        jobs = json.loads(_gh("run", "view", run_id, "--json", "jobs"))["jobs"]
        job_ids = [str(entry["databaseId"]) for entry in jobs if entry["name"] == job]
        if not job_ids:
            continue
        try:
            log = _gh("run", "view", "--job", job_ids[0], "--log")
        except subprocess.CalledProcessError:
            # Skipped jobs and expired logs have nothing to download.
            continue
        try:
            run_times = parse_log(log).run_times
        except LogParseError as err:
            raise LogParseError(f"{branch} run {run_id}: {err}") from err
        if run_times:
            baselines.append((run_id, run_times))
        if len(baselines) == count:
            break
    return baselines


_TABLE_HEADER = ("case", "baseline", "spread", "current", "change", "verdict")
# Columns aligned left; the others (the numbers) right.
_LEFT_ALIGNED = {"case", "verdict"}


def _format_table(rows: Sequence[Row], baseline_label: str, runs: int) -> str:
    """A plain-text table with aligned columns, for the terminal and log."""
    cells: list[Sequence[str]] = [_TABLE_HEADER]
    for row in rows:
        cells.append(
            (
                row.case_id,
                "-" if row.baseline is None else f"{row.baseline:.2f}s",
                "-"
                if row.spread is None
                else f"{row.spread[0]:.2f}-{row.spread[1]:.2f}s",
                "-" if row.current is None else f"{row.current:.2f}s",
                "-" if row.percent is None else f"{row.percent:+.1f}%",
                str(row.verdict),
            )
        )
    widths = [max(len(line[i]) for line in cells) for i in range(len(_TABLE_HEADER))]

    def format_line(line: Sequence[str]) -> str:
        return "  ".join(
            cell.ljust(width) if name in _LEFT_ALIGNED else cell.rjust(width)
            for name, cell, width in zip(_TABLE_HEADER, line, widths)
        ).rstrip()

    lines = [
        (
            f"Performance against main run {baseline_label} "
            f"(host wall-clock of iree-run-module; spread over the last {runs} "
            "main run(s))"
        ),
        "",
        format_line(cells[0]),
        "  ".join("-" * width for width in widths),
        *(format_line(line) for line in cells[1:]),
    ]
    return "\n".join(lines) + "\n"


def _report(table: str) -> None:
    print(table)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        # Markdown; a code block keeps the alignment.
        with open(summary, "a") as file:
            file.write(f"```text\n{table}```\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--current", type=Path, required=True, help="pytest output of this session"
    )
    parser.add_argument(
        "--baseline-log",
        type=Path,
        action="append",
        default=[],
        help="pytest output or GitHub job log to compare against, newest first "
        "(repeatable); skips fetching main runs with gh",
    )
    parser.add_argument(
        "--job",
        default=os.environ.get("GITHUB_JOB"),
        help="job whose main runs are the baseline (default: $GITHUB_JOB)",
    )
    parser.add_argument("--workflow", default="build.yml")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--baseline-runs", type=int, required=True)
    parser.add_argument(
        "--max-runs-searched",
        type=int,
        required=True,
        help="how many main runs to look through for --baseline-runs with run "
        "times; runs without them (cancelled, expired log) are skipped",
    )
    parser.add_argument("--warn-percent", type=float, required=True)
    parser.add_argument("--fail-percent", type=float, required=True)
    parser.add_argument("--min-delta-s", type=float, required=True)
    args = parser.parse_args(argv)

    thresholds = Thresholds(
        baseline_runs=args.baseline_runs,
        warn_percent=args.warn_percent,
        fail_percent=args.fail_percent,
        min_delta_s=args.min_delta_s,
    )

    try:
        return _check(args, parser, thresholds)
    except LogParseError as err:
        print(f"::error title=perf check::{err}")
        return 1


def _parse_file(path: Path) -> ParsedLog:
    try:
        return parse_log(path.read_text(errors="replace"))
    except LogParseError as err:
        raise LogParseError(f"{path}: {err}") from err


def _check(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    thresholds: Thresholds,
) -> int:
    current = _parse_file(args.current)
    if not current.run_times:
        if any(_is_timed(case) for case in current.passed):
            # Tests passed but no time was found: the output format changed
            # under this parser. Fail rather than silently check nothing.
            print(
                f"::error title=perf check::{args.current}: no run time found for "
                "any passed test; did the 'timings:' output of test_models.py "
                "change?"
            )
            return 1
        print(f"::notice title=perf check::{args.current}: no passed timed test")
        return 0

    if args.baseline_log:
        baselines = [
            (str(path), _parse_file(path).run_times)
            for path in args.baseline_log[: thresholds.baseline_runs]
        ]
        baselines = [(label, times) for label, times in baselines if times]
    elif args.job is None:
        parser.error("--job (or $GITHUB_JOB) is required to fetch main runs")
    else:
        try:
            baselines = fetch_baselines(
                args.job,
                args.workflow,
                args.branch,
                thresholds.baseline_runs,
                args.max_runs_searched,
            )
        except subprocess.CalledProcessError as err:
            # A perf check that cannot see main is not a reason to fail the PR.
            print(f"::warning title=perf check::gh failed: {err.stderr.strip()}")
            return 1
    if not baselines:
        print(f"::notice title=perf check::no {args.branch} run with run times found")
        return 1

    rows = compare(current.run_times, [times for _, times in baselines], thresholds)
    _report(_format_table(rows, baselines[0][0], len(baselines)))
    for row in rows:
        if row.verdict is Verdict.MISSING:
            print(
                f"::warning title=perf case missing::{row.case_id}: timed in "
                f"{args.branch} run {baselines[0][0]} ({row.baseline:.2f}s), but "
                "not in this run (removed, renamed or failed?)"
            )
        elif row.verdict in (Verdict.FAIL, Verdict.WARN):
            level = "error" if row.verdict is Verdict.FAIL else "warning"
            print(
                f"::{level} title=perf regression::{row.case_id}: "
                f"{row.baseline:.2f}s -> {row.current:.2f}s ({row.percent:+.1f}%)"
            )
    return 1 if any(row.verdict is Verdict.FAIL for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
