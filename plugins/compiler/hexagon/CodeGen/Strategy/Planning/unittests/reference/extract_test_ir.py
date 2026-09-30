#!/usr/bin/env python3
"""Extract a register-estimation test's MLIR into a file you can compile.

A test file embeds each op variant's IR in one R"mlir(...)mlir" literal, as
the second argument of a REGISTER_ESTIMATION_TEST_SUITE(name, ir, ...) call,
and states the tilings it asks the estimator about. This pulls both out: the
IR into example.mlir, and the tilings

A file with several variants needs --name to say which one to extract (the
suite's name, the macro's first argument); pass --list to see the available
names instead. Exactly one of --list/--name is required, even for a file
with a single literal.

Caveat: A test's IR is one dispatch cut out of a larger example, and recompiling it
runs dispatch formation again, which may not put the boundary back where it
was.
"""

import argparse
import re
import sys
import textwrap
from collections.abc import Generator
from pathlib import Path
from typing import Any

MLIR_LITERAL = re.compile(r'R"mlir\((.*?)\)mlir"', re.DOTALL)
CONFIG_ENTRY = re.compile(r"\b(DispatchConfig|TileConfig)\s*\{\s*\n?\s*\"([^\"]+)\"")
HVX = re.compile(r"measured HvxVR\s*=\s*(\d+)")
ROOT = re.compile(r"/\*rootIndex=\*/\s*([\w:]+)")
SUITE_MACRO = re.compile(r"\bREGISTER_ESTIMATION_TEST_SUITE\s*\(")


def grouped(text: str, start: int, open_chr: str, close_chr: str) -> str:
    """Returns the open_chr...close_chr group beginning at or after `start`,
    delimiters included."""
    open_at = text.index(open_chr, start)
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == open_chr:
            depth += 1
        elif text[i] == close_chr:
            depth -= 1
            if depth == 0:
                return text[open_at : i + 1]
    raise ValueError(f"unbalanced {open_chr!r}{close_chr!r}")


def braced(text: str, start: int) -> str:
    """Returns the {...} group beginning at or after `start`, braces included."""
    return grouped(text, start, "{", "}")


def parened(text: str, start: int) -> str:
    """Returns the (...) group beginning at or after `start`, parens included."""
    return grouped(text, start, "(", ")")


def variants(source: str) -> list[tuple[str | None, str, int, int]]:
    """returns a list of (name, ir, start, end) per MLIR literal in the file."""
    found = []
    for macro in SUITE_MACRO.finditer(source):
        start = macro.start()
        open_at = source.index("(", start)
        group = parened(source, start)
        end = open_at + len(group)
        literal = MLIR_LITERAL.search(group)
        if not literal:
            continue
        name = group.lstrip("(").split(",", 1)[0].strip() or None
        found.append((name, literal.group(1), start, end))
    return found


def configs_for(source: str, start: int, end: int) -> list:
    """The configs declared for one variant (within its own macro call)."""
    return list(configs(source[start:end]))


def tile_lists(entry: str) -> str:
    """Formats an entry's tile sizes, whichever of the two field names it uses."""
    for field in ("perOpTileSizes", "tileSizes"):
        marker = f"/*{field}=*/"
        at = entry.find(marker)
        if at == -1:
            continue
        group = braced(entry, at + len(marker))
        # One [..] per op for a dispatch config; a single [..] for a per-op one.
        inner = [g for g in re.findall(r"\{([^{}]*)\}", group)]
        if not inner:  # per-op: the outer braces are the only ones
            inner = [group.strip("{}")]
        return " ".join(
            "[" + ",".join(v.strip() for v in g.split(",") if v.strip()) + "]"
            for g in inner
        )
    return "?"


def configs(source: str) -> Generator[tuple[str | Any, str, str | Any, str | Any]]:
    """Yields (name, tiles, root, measured HvxVR) per config the test declares."""
    for match in CONFIG_ENTRY.finditer(source):
        entry = braced(source, match.start())
        root = ROOT.search(entry)
        hvx = HVX.search(entry)
        yield (
            match.group(2),
            tile_lists(entry),
            root.group(1) if root else "-",
            hvx.group(1) if hvx else "-",
        )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("test", help="path to the test's .cpp file")
    ap.add_argument("-o", "--out", default="example.mlir", help="output file name")
    picker = ap.add_mutually_exclusive_group(required=True)
    picker.add_argument(
        "--list", action="store_true", help="list the available tests and exit"
    )
    picker.add_argument("--name", help="which test's IR to extract")
    ap.add_argument(
        "--quiet", action="store_true", help="write the file, print nothing else"
    )
    args = ap.parse_args(argv[1:])

    path = Path(args.test)
    if not path.is_file():
        print(f"no such file: {path}")
        return 1
    source = path.read_text()

    found = variants(source)
    names = ", ".join(v[0] or "?" for v in found)
    if not found:
        print(f"{path.name} holds no MLIR literals.")
        return 1

    if args.list:
        for n, *_ in found:
            print(n or "?")
        return 0

    picked = [v for v in found if v[0] == args.name]
    if len(picked) != 1:
        print(
            f"{path.name} holds {len(picked)} literals for --name "
            f"{args.name}, expected exactly one. It has: {names}"
        )
        return 1
    chosen = picked[0]

    name, literal, start, end = chosen
    ir = textwrap.dedent(literal).strip() + "\n"
    Path(args.out).write_text(ir)

    if args.quiet:
        return 0

    ops = len(re.findall(r"=\s*linalg\.(?!yield)", ir))
    print(f"wrote {args.out} from {path.name}" + (f" ({name})" if name else ""))
    print(f"  {ops} linalg ops - check config.mlir still has this many")

    rows = configs_for(source, start, end)
    width = max([len("config")] + [len(r[0]) for r in rows])
    print()
    print("  the tilings this test asks about:")
    print(f"    {'config':<{width}}  {'root':<4}  {'HvxVR':<5}  tiles")
    for config_name, tiles, root, hvx in rows:
        print(f"    {config_name:<{width}}  {root:<4}  {hvx:<5}  {tiles}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
