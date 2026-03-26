#! /usr/bin/env python3

import os

tools = [
    "clang",
    "hexagon-ar",
    "hexagon-nm",
    "hexagon-objdump",
    "hexagon-strip",
    "link",
]


def main() -> None:
    this_dir = os.path.dirname(__file__)
    for tool in tools:
        trampoline = os.path.join(this_dir, tool)
        with open(trampoline, "w", encoding="utf-8") as tf:
            print("#! /bin/sh", file=tf)
            print(
                f'exec external/_main~cellar_hexagon_repos~hexagon_sdk/tools/HEXAGON_Tools/19.0.04/Tools/bin/{tool} "$@"',
                file=tf,
            )
        os.chmod(trampoline, 0o755)


if __name__ == "__main__":
    main()
