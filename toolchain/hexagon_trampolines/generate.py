#! /usr/bin/env python3
# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

tools = [
    "clang",
    "hexagon-ar",
    "hexagon-nm",
    "hexagon-objdump",
    "hexagon-strip",
    "link",
]

# The canonical repo name for the hexagon SDK changes across Bazel major
# versions (e.g. _main~ext~repo in Bazel 7, +ext+repo in Bazel 8). Rather
# than hardcoding it, the trampoline resolves the path at execution time via
# the unquoted glob in exec's argument, which the shell expands to the unique
# external/<canonical>hexagon_sdk directory before invoking exec.
TRAMPOLINE_TEMPLATE = """\
#! /bin/sh
exec external/*hexagon_sdk/tools/HEXAGON_Tools/19.0.04/Tools/bin/{tool} "$@"
"""


def main() -> None:
    this_dir = os.path.dirname(__file__)
    for tool in tools:
        trampoline = os.path.join(this_dir, tool)
        with open(trampoline, "w", encoding="utf-8") as tf:
            tf.write(TRAMPOLINE_TEMPLATE.format(tool=tool))
        os.chmod(trampoline, 0o755)


if __name__ == "__main__":
    main()
