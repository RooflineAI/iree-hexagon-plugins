#!/usr/bin/env python3
"""Measures real HvxVR usage for every tiling a tests asks about and collect these into one csv file.
Orchestrates the other scripts (see README.md)
"""

import argparse
import csv
import re
import subprocess
import sys
import tempfile
import textwrap
from enum import StrEnum
from pathlib import Path

import extract_test_ir as eti

SCRIPT_DIR = Path(__file__).resolve().parent

NOTE_VALUE_RE = re.compile(r"(?i)measured\s*:?\s*(?:hvxvr\s*=\s*)?(\d+)")
EXPECTED_PEAK_RE = re.compile(
    r"/\*(?:expectedPeakRegisters|expectedVector)=\*/\s*(-?\d+)"
)
GENERIC_ITERATOR_TYPES_RE = re.compile(r"iterator_types\s*=\s*\[([^\]]*)\]")
VECTOR_COMMON_RE = re.compile(r"(vector_common_parallel = \[)[^\]]*(\])")
VECTOR_REDUCTION_RE = re.compile(r"(vector_reduction = \[)[^\]]*(\])")

# Reduction-dim positions for named linalg ops this suite uses, in their
# fixed iteration order. linalg.generic ops don't need an entry here: their
# iterator_types are read straight out of the snippet.
# (iterator_types are not shown in the ir for named ops)
NAMED_OP_REDUCTION_DIMS = {
    "linalg.matmul": {2},  # [M, N, K]
    "linalg.batch_matmul": {3},  # [B, M, N, K]
    "linalg.depthwise_conv_2d_nhwc_hwc": {4, 5},  # [N, OH, OW, C, KH, KW]
}


class Status(StrEnum):
    """How a config's real measured HvxVR compares to the estimate and the note.

    Only set once a real measurement was obtained; a failed compile or
    measurement leaves row["status"] as the plain string "error" instead -
    "error" is deliberately not a member of this enum, since it means no
    comparison was possible at all.

    MATCH: expected_peak_registers == real measured value
    KNOWN_MISMATCH: the estimate is wrong, but the note already states the
        real measured value, the mismatch is already documented
    NEW_MISMATCH: the estimate is wrong, and the note doesn't state the real
        value either: undocumented mismatch
    """

    MATCH = "match"
    KNOWN_MISMATCH = "known-mismatch"
    NEW_MISMATCH = "new-mismatch"


def reduction_dims(ir: str) -> set[int]:
    """The tile-size positions that are reduction dims for this snippet's op."""
    m = GENERIC_ITERATOR_TYPES_RE.search(ir)
    if m:
        types = [t.strip().strip('"') for t in m.group(1).split(",")]
        return {i for i, t in enumerate(types) if t == "reduction"}
    for op, dims in NAMED_OP_REDUCTION_DIMS.items():
        if re.search(rf"\b{re.escape(op)}\b", ir):
            return dims
    raise ValueError(
        "couldn't tell which tile-size positions are reduction dims: no "
        "iterator_types (not a linalg.generic) and the op isn't in "
        "NAMED_OP_REDUCTION_DIMS"
    )


def entries_for(source: str, start: int, end: int) -> list[tuple[str, str]]:
    """(name, full entry text) per TileConfig/DispatchConfig in a suite's call."""
    region = source[start:end]
    return [
        (m.group(2), eti.braced(region, m.start()))
        for m in eti.CONFIG_ENTRY.finditer(region)
    ]


def parse_tile_sizes(entry: str) -> list[int]:
    marker = "/*tileSizes=*/"
    at = entry.find(marker)
    if at == -1:
        raise ValueError("no /*tileSizes=*/ field in this config")
    group = eti.braced(entry, at + len(marker))
    return [int(v.strip()) for v in group.strip("{}").split(",") if v.strip()]


def set_vector_tile_sizes(
    config_text: str, tile_sizes: list[int], reduction: set[int]
) -> str:
    common = [t if i not in reduction else 0 for i, t in enumerate(tile_sizes)]
    common_str = ", ".join(str(v) for v in common)
    text, n = VECTOR_COMMON_RE.subn(
        lambda m: m.group(1) + common_str + m.group(2), config_text
    )
    if n != 1:
        raise ValueError(f"expected exactly one vector_common_parallel, found {n}")
    # A purely-parallel op (no reduction dims) has no vector_reduction field
    # at all in the auto-generated lowering_config - nothing to overwrite.
    reduce = [t if i in reduction else 0 for i, t in enumerate(tile_sizes)]
    if any(reduce):
        reduce_str = ", ".join(str(v) for v in reduce)
        text, n = VECTOR_REDUCTION_RE.subn(
            lambda m: m.group(1) + reduce_str + m.group(2), text
        )
        if n != 1:
            raise ValueError(f"expected exactly one vector_reduction, found {n}")
    return text


def run_shell_script(script: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(SCRIPT_DIR / script), *args],
        capture_output=True,
        text=True,
    )


def crash_summary(stderr: str) -> str:
    for line in stderr.splitlines():
        if "Assertion" in line or "error:" in line:
            return line.strip()
    lines = [line for line in stderr.splitlines() if line.strip()]
    return lines[-1].strip() if lines else "(no stderr)"


def measure_one(
    name: str,
    config: str,
    entry: str,
    reduction: set[int],
    base_config: str,
    scratch: Path,
) -> dict:
    row = {
        "name": name,
        "config": config,
        "tile_sizes": None,
        "note_value": None,
        "measured_hvxvr": None,
        "expected_peak_registers": None,
        "status": None,
        "detail": "",
    }
    # parsing the values for this config

    # The Expected Peak Registers the test tests against
    peak_m = EXPECTED_PEAK_RE.search(entry)
    row["expected_peak_registers"] = int(peak_m.group(1)) if peak_m else None

    # The Measured Value we find in the comment
    note_m = NOTE_VALUE_RE.search(entry)
    row["note_value"] = int(note_m.group(1)) if note_m else None

    tile_sizes = parse_tile_sizes(entry)
    row["tile_sizes"] = "[" + ",".join(str(t) for t in tile_sizes) + "]"

    # set the tilesizes in the mlir:
    config_text = set_vector_tile_sizes(base_config, tile_sizes, reduction)
    # write to file to be used as input for compile
    config_path = scratch / f"{name}__{config}.config.mlir"
    config_path.write_text(config_text)
    out_csv = scratch / f"{name}__{config}.pressure.csv"
    out_exe = scratch / f"{name}__{config}.vmfb"

    # run measurement
    result = run_shell_script(
        "get_reg_pressure.sh",
        [str(config_path), str(out_csv), str(out_exe)],
    )
    if result.returncode != 0:
        row["status"] = "error"
        row["detail"] = crash_summary(result.stderr)
        return row

    # extract Hvx register count, 0 if none are used
    measured = 0
    if out_csv.exists():
        with open(out_csv, newline="") as f:
            for csv_row in csv.DictReader(f):
                if csv_row["pressure_set"] == "HvxVR":
                    measured = int(csv_row["max_value"])
    row["measured_hvxvr"] = measured

    if row["expected_peak_registers"] == measured:
        row["status"] = Status.MATCH
    elif row["note_value"] == measured:
        row["status"] = Status.KNOWN_MISMATCH
    else:
        row["status"] = Status.NEW_MISMATCH
    return row


def measure_suite(
    name: str, ir: str, source: str, start: int, end: int, scratch: Path
) -> list[dict]:
    entries = entries_for(source, start, end)
    reduction = reduction_dims(ir)

    # generate base_config
    mlir_path = scratch / f"{name}.mlir"
    mlir_path.write_text(textwrap.dedent(ir).strip() + "\n")
    base_config_path = scratch / f"{name}.base_config.mlir"
    result = run_shell_script("to_config.sh", [str(mlir_path), str(base_config_path)])
    if result.returncode != 0:
        detail = crash_summary(result.stderr)
        return [
            {
                "name": name,
                "config": config,
                "tile_sizes": None,
                "note_value": None,
                "measured_hvxvr": None,
                "expected_peak_registers": None,
                "status": "error",
                "detail": f"to_config.sh: {detail}",
            }
            for config, _ in entries
        ]
    base_config = base_config_path.read_text()

    return [
        measure_one(name, config, entry, reduction, base_config, scratch)
        for config, entry in entries
    ]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("test", help="path to the test's .cpp file")
    ap.add_argument(
        "--name",
        nargs="+",
        help="only measure these tests (default: every test in the file)",
    )
    ap.add_argument("-o", "--out", default="results.csv", help="output CSV path")
    args = ap.parse_args(argv[1:])

    path = Path(args.test)
    if not path.is_file():
        print(f"no such file: {path}", file=sys.stderr)
        return 1
    source = path.read_text()

    found = eti.variants(source)
    if not found:
        print(f"{path.name} holds no MLIR literals.", file=sys.stderr)
        return 1
    if args.name:
        wanted = set(args.name)
        found = [v for v in found if v[0] in wanted]
        missing = wanted - {v[0] for v in found}
        if missing:
            print(f"no such test(s): {', '.join(sorted(missing))}", file=sys.stderr)
            return 1

    scratch = Path(tempfile.mkdtemp(prefix="measure_all_"))
    print(f"scratch directory: {scratch}")

    rows: list[dict] = []
    for name, ir, start, end in found:
        if name is None:
            print("skipping a test with no name", file=sys.stderr)
            continue
        print(f"=== {name} ===")
        rows.extend(measure_suite(name, ir, source, start, end, scratch))

    fieldnames = [
        "name",
        "config",
        "tile_sizes",
        "note_value",
        "measured_hvxvr",
        "expected_peak_registers",
        "status",
        "detail",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}")

    new_mismatches = [r for r in rows if r["status"] == Status.NEW_MISMATCH]
    print("\nnew mismatches (neither the estimate nor the note matches reality):")
    for r in new_mismatches:
        print(
            f"  {r['name']}/{r['config']}  tiles={r['tile_sizes']}  "
            f"expected={r['expected_peak_registers']}  note={r['note_value']}  "
            f"measured={r['measured_hvxvr']}"
        )
    if not new_mismatches:
        print("  (none)")

    counts = {status: 0 for status in (*Status, "error")}
    for r in rows:
        counts[r["status"]] += 1
    print(
        f"\n{len(rows)} configs: {counts[Status.MATCH]} match, "
        f"{counts[Status.KNOWN_MISMATCH]} known mismatch, "
        f"{counts[Status.NEW_MISMATCH]} new mismatch, {counts['error']} errored"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
