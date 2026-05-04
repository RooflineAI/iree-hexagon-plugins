#!/usr/bin/env python3

import sys
from pathlib import Path

import pytest
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


def test_dsp_runtime_exports(runtime_so: Path) -> None:
    exported = exported_symbols(runtime_so)
    # ignore additional symbols
    exported &= set(SYMBOLS)
    assert sorted(exported) == sorted(SYMBOLS), "missing exported DSP runtime symbols"


if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(pytest.main([__file__, "-vv"] + args))
