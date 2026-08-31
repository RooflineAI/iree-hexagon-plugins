# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Entry point for the one Bazel target here, the model-tree check."""

import sys
from pathlib import Path

import pytest

if __name__ == "__main__":
    given = sys.argv[1:]
    if any(not arg.startswith("-") for arg in given):
        targets = []
    else:
        targets = [str(Path(__file__).resolve().parent)]
    # -rA so the reason for every skip and xfail is printed
    sys.exit(pytest.main([*targets, "-vv", "-rA", *given]))
