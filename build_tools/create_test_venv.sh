#!/usr/bin/env bash
# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Creates (or updates) a virtualenv holding the Python dependencies the Hexagon
# integration tests need.
#
#   build_tools/create_test_venv.sh [venv-dir]      # default: ./.venv
#
# This exists because a bare `uv pip install -r build_tools/bazel/requirements.txt`
# fails, and the failure is opaque:
#
#   × No solution found when resolving dependencies:
#   ╰─▶ Because there is no version of torch==2.9.0+cpu ...
#
# torch is pinned to the "+cpu" build, which is published only on PyTorch's own
# index - the same version on PyPI drags in the whole nvidia-* CUDA stack. The
# lock file names that index itself, but uv still stops at the first index that
# carries a package unless --index-strategy says otherwise, and that setting
# cannot be written into a requirements file. So it lives here, once, where CI
# and a developer use the same copy.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-${REPO_ROOT}/.venv}"
REQUIREMENTS="${REPO_ROOT}/build_tools/bazel/requirements.txt"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is not on PATH (see https://docs.astral.sh/uv/)" >&2
    exit 1
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "creating ${VENV}"
    uv venv --python 3.11 "${VENV}"
fi

echo "installing ${REQUIREMENTS} into ${VENV}"
uv pip install \
    --python "${VENV}/bin/python" \
    --index-strategy unsafe-best-match \
    -r "${REQUIREMENTS}"

echo
echo "done. Run the integration tests with:"
echo "  ${VENV}/bin/pytest integration_tests/hexagon -v -rA"
