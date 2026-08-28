# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""py_test entry point: runs pytest over this directory.

A plain `pytest integration_tests/hexagon` does the same thing; this exists so
Bazel has a single `main` to invoke and can hand the tool paths over as `args`.
"""

import sys
from pathlib import Path

import pytest

if __name__ == "__main__":
    # -rA so the reason for every skip and xfail is printed
    sys.exit(
        pytest.main([str(Path(__file__).resolve().parent), "-vv", "-rA", *sys.argv[1:]])
    )
