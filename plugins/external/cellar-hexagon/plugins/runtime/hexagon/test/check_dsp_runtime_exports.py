#!/usr/bin/env python3

import sys
from pathlib import Path

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

SYMBOLS = [
    "hexagon_runtime_dma_start",
    "hexagon_runtime_dma2d_start",
    "hexagon_runtime_dma_wait",
    "hexagon_runtime_alloc_1d",
    "hexagon_runtime_free_1d",
    "hexagon_runtime_copy",
    "hexagon_runtime_malloc",
    "hexagon_runtime_free",
    "hexagon_runtime_memref_copy",
    "hexkl_matmul_f16f16_f32",
    "hexkl_micro_hmx_config_size",
    "hexkl_micro_hmx_mm_f16",
]


def exported_symbols(path: Path) -> set[str]:
    with path.open("rb") as file:
        elf = ELFFile(file)
        result = set()
        for section in elf.iter_sections():
            if not isinstance(section, SymbolTableSection):
                continue
            for symbol in section.iter_symbols():
                if not symbol.name:
                    continue
                if symbol.entry.st_shndx != "SHN_UNDEF" and symbol["st_info"][
                    "bind"
                ] in ("STB_GLOBAL", "STB_WEAK"):
                    result.add(symbol.name)
    return result


def main(args: list[str]) -> int:
    if len(args) != 1:
        print(f"usage: {sys.argv[0]} <runtime_so>", file=sys.stderr)
        return 2

    runtime_so = Path(args[0])
    exported = exported_symbols(runtime_so)
    missing = [symbol for symbol in SYMBOLS if symbol not in exported]
    if missing:
        for symbol in missing:
            print(f"missing exported DSP runtime symbol: {symbol}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
