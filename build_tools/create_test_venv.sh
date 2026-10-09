#!/usr/bin/env bash
# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Creates (or updates) the integration-test venv with this build's compiler
# Python package and locked external dependencies.
#
#   build_tools/create_test_venv.sh [venv-dir] [--compiler-wheel wheel]
#
# Development builds/install the Bazel package editably. CI supplies the wheel
# from its build job, so test runners do not need Bazel or submodule checkouts.
#
# torch is pinned to the "+cpu" build, which is published only on PyTorch's own
# index - the same version on PyPI drags in the whole nvidia-* CUDA stack. The
# lock file names that index itself, but uv still stops at the first index that
# carries a package unless --index-strategy says otherwise, and that setting
# cannot be written into a requirements file. So it lives here, once, where CI
# and a developer use the same copy.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${repo_root}/.venv"
compiler_wheel=""
have_venv_dir=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --compiler-wheel)
            if [[ $# -lt 2 ]]; then
                echo "error: --compiler-wheel requires a wheel path" >&2
                exit 1
            fi
            compiler_wheel="$(realpath "$2")"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [venv-dir] [--compiler-wheel wheel]"
            exit 0
            ;;
        -*)
            echo "error: unknown option $1" >&2
            exit 1
            ;;
        *)
            if [[ "${have_venv_dir}" == true ]]; then
                echo "error: supply only one venv directory" >&2
                exit 1
            fi
            venv_dir="$(realpath -m "$1")"
            have_venv_dir=true
            shift
            ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is not on PATH (see https://docs.astral.sh/uv/)" >&2
    exit 1
fi

setup_args=(--venv "${venv_dir}")
if [[ -n "${compiler_wheel}" ]]; then
    setup_args+=(--compiler-wheel "${compiler_wheel}")
else
    cd "${repo_root}"
    package_target="//build_tools/python_compiler:package"
    bazel build "${package_target}"
    package_output="$(bazel cquery --output=files "config(${package_target}, target)")"
    if [[ -z "${package_output}" || "${package_output}" == *$'\n'* ]]; then
        echo "error: expected one compiler package output, got ${package_output}" >&2
        exit 1
    fi
    setup_args+=(--compiler-package "$(realpath "${package_output}")")
fi

python3 "${repo_root}/build_tools/setup_test_venv.py" "${setup_args[@]}"

echo
echo "done. Run the integration tests with:"
echo "  ${venv_dir}/bin/pytest integration_tests/ -v -rA"
