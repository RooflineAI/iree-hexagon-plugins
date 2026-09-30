"""Smoke test: does measure_all.py, measuring real Hexagon register pressure
actually runs end-to-end ?

Uses the Polynomial tests to try that:
Its a rather simple Op, where vectorization works rather straightforward, so
we dont expect lots of changes, when implementation of the vectorization evolves
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

MEASURE_ALL = Path(__file__).resolve().parent / "measure_all.py"
POLYNOMIAL_TESTFILE = (
    Path(__file__).resolve().parent.parent / "single_op" / "PolynomialTest.cpp"
)
EXPECTED_TESTS = {"IreeChosenTiling", "CoefficientsDoNotScaleWithTile"}


def run_measure_all(
    cpp_file: Path, out_csv: Path
) -> tuple[subprocess.CompletedProcess, list[dict]]:
    """Runs measure_all.py against one test .cpp file, returns (process, rows).
    rows is [] if measure_all.py failed and never wrote out_csv at all.
    """
    result = subprocess.run(
        [sys.executable, str(MEASURE_ALL), str(cpp_file), "-o", str(out_csv)],
        capture_output=True,
        text=True,
    )
    rows = []
    if out_csv.is_file():
        with open(out_csv, newline="") as f:
            rows = list(csv.DictReader(f))
    return result, rows


def llc_has_assertions() -> bool:
    """llc needs assertions enabled for `-debug-only=machine-scheduler` to
    produce anything"""
    llc = os.environ.get("LLC_WITH_DEBUG", "llc")
    result = subprocess.run([llc, "--version"], capture_output=True, text=True)
    return "assertions" in result.stdout.lower()


def test_measure_all_runs_end_to_end(tmp_path: Path) -> None:
    if not llc_has_assertions():
        pytest.skip(
            "this build's llc has no assertions, so register pressure can't be measured here"
        )

    result, rows = run_measure_all(POLYNOMIAL_TESTFILE, tmp_path / "results.csv")
    assert rows, (
        f"measure_all.py produced no results.csv rows\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    assert result.returncode == 0, result.stderr
    assert len(rows) == 2, rows
    # check that the tests are still the same
    assert {r["config"] for r in rows} == EXPECTED_TESTS

    bad = [r for r in rows if r["status"] != "match"]
    assert not bad, (
        "Polynomial's register pressure should be stable and optimal "
        f"regardless of unrelated codegen changes: {bad}"
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(pytest.main([__file__, "-vv"] + args))
