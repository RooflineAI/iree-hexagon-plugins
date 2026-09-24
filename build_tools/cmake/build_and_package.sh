#!/bin/bash
# Copyright 2026 RooflineAI GmbH
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Builds IREE compiler and related tools for host and IREE runtime for
# Android arm64, including Hexagon parts for Hexagon runtime.
#
# Usage: build_tools/cmake/build_and_package.sh [output-dir]
#
# Requires: a Ninja + a clang-19 toolchain on PATH, ANDROID_NDK_HOME set to
# the NDK described in README.md, and network access (the Hexagon SDK and
# HexKL are downloaded below).

set -xeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IREE_SRC="${REPO_ROOT}/third-party/iree"
LLVM_SRC="${REPO_ROOT}/third-party/llvm-project"
BUILD_ROOT="${REPO_ROOT}/build"
mkdir -p "${BUILD_ROOT}"
OUT_DIR="$(realpath "${1:-${REPO_ROOT}/build-artifacts-cmake}")"
mkdir -p "${OUT_DIR}"

export PATH="/usr/lib/llvm-19/bin:$PATH"

export IREE_HEXAGON_PLUGINS_ROOT="${REPO_ROOT}"

# --- Hexagon SDK: fetch and extract ----------------------------------------
export HEXAGON_SDK_ROOT="${BUILD_ROOT}/hexagon-sdk"
if [[ ! -d "${HEXAGON_SDK_ROOT}" ]]; then
  _hexagon_sdk_url="https://github.com/snapdragon-toolchain/hexagon-sdk/releases/download/v6.4.0.2/hexagon-sdk-v6.4.0.2-amd64-lnx.tar.xz"
  _hexagon_sdk_sha256="b4a57a774795cf12da19a777a5d306e970905bf9758a4c4765e5e4593428ae0b"
  _hexagon_sdk_tmp="$(mktemp -d)"
  curl -sL "${_hexagon_sdk_url}" -o "${_hexagon_sdk_tmp}/sdk.tar.xz"
  echo "${_hexagon_sdk_sha256}  ${_hexagon_sdk_tmp}/sdk.tar.xz" | sha256sum -c -
  mkdir -p "${_hexagon_sdk_tmp}/extracted"
  # The archive has a single top-level "6.4.0.2/" directory; strip it so
  # HEXAGON_SDK_ROOT ends up pointing directly at the SDK contents.
  tar -xf "${_hexagon_sdk_tmp}/sdk.tar.xz" -C "${_hexagon_sdk_tmp}/extracted" --strip-components=1
  mv "${_hexagon_sdk_tmp}/extracted" "${HEXAGON_SDK_ROOT}"
  rm -rf "${_hexagon_sdk_tmp}"
fi

# --- HexKL: fetch and extract -----------------------------------------------
export HEXKL_ROOT="${BUILD_ROOT}/hexkl"
if [[ ! -d "${HEXKL_ROOT}" ]]; then
  _hexkl_outer_zip="Hexagon_KL.Core.1.0.0.Linux-Any.zip"
  _hexkl_inner_zip="hexkl-1.0.0-beta1-6.4.0.0.zip"
  _hexkl_inner_zip_sha256="409add79ec895e8eb8062e3fc2bb9d74d548e5288fde729cc5d3376f0437dfd9"
  _hexkl_tmp="$(mktemp -d)"
  curl -sL "https://softwarecenter.qualcomm.com/api/download/software/tools/Hexagon_KL/Linux/1.0.0/${_hexkl_outer_zip}" \
    -o "${_hexkl_tmp}/outer.zip"
  mkdir -p "${_hexkl_tmp}/extracted"
  unzip -q "${_hexkl_tmp}/outer.zip" -d "${_hexkl_tmp}/extracted"
  if [[ -f "${_hexkl_tmp}/extracted/${_hexkl_inner_zip}" ]]; then
    _hexkl_inner_zip_path="${_hexkl_tmp}/extracted/${_hexkl_inner_zip}"
  else
    _hexkl_inner_zip_path="${_hexkl_tmp}/extracted/Hexagon_KL.Core.1.0.0.Linux-Any/${_hexkl_inner_zip}"
  fi
  echo "${_hexkl_inner_zip_sha256}  ${_hexkl_inner_zip_path}" | sha256sum -c -
  unzip -q "${_hexkl_inner_zip_path}" -d "${_hexkl_tmp}/extracted"
  if [[ ! -d "${_hexkl_tmp}/extracted/hexkl_addon" \
        && -d "${_hexkl_tmp}/extracted/hexkl-1.0.0-beta1-6.4.0.0/hexkl_addon" ]]; then
    ln -s "${_hexkl_tmp}/extracted/hexkl-1.0.0-beta1-6.4.0.0/hexkl_addon" \
      "${_hexkl_tmp}/extracted/hexkl_addon"
  fi
  mv "${_hexkl_tmp}/extracted" "${HEXKL_ROOT}"
  rm -rf "${_hexkl_tmp}"
fi

# --- Compiler settings ------------------------------------------------------
CC=/usr/lib/llvm-19/bin/clang
CXX=/usr/lib/llvm-19/bin/clang++

COMMON_FLAGS=(
  -G Ninja
  -DIREE_CMAKE_PLUGIN_PATHS="${REPO_ROOT}/plugins"
)

HOST_COMPILER_FLAGS=(
  -DCMAKE_C_COMPILER="$CC"
  -DCMAKE_CXX_COMPILER="$CXX"
)

# --- lld (its own, unrelated LLVM CMake build) -----------------------------
cmake -S "${LLVM_SRC}/llvm" -B "${BUILD_ROOT}/llvm" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=lld \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX"
cmake --build "${BUILD_ROOT}/llvm" --target lld -- -k 0
cp "${BUILD_ROOT}/llvm/bin/ld.lld" "${OUT_DIR}/"

# --- host tree: iree-compile, iree-opt, ... (+ the Hexagon compiler plugin)
# Explicitly turn on the Hexagon LLVM backend (off by default).
# IREE's nested LLVM, StableHLO, and torch-mlir paths are symlinks to this
# repo's matching top-level submodules, so git still reports those nested
# gitlinks as uninitialized even though their sources are available.
cmake -S "${IREE_SRC}" -B "${BUILD_ROOT}/host" \
  "${COMMON_FLAGS[@]}" \
  "${HOST_COMPILER_FLAGS[@]}" \
  -DLLVM_TARGETS_TO_BUILD=Hexagon \
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_BUILD_TESTS=ON \
  -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF \
  -DIREE_LINK_COMPILER_SHARED_LIBRARY=OFF
cmake --build "${BUILD_ROOT}/host" --target \
  iree-compile iree-opt iree-dump-module iree-dump-parameters iree-encode-parameters \
  iree_hexagon_plugins_plugins_runtime_hexagon_test_{bindings,command_buffer,cmd_{barrier,copy,dispatch,fill}}_serialize_test \
  -- -k 0
HOST_BIN_DIR="${BUILD_ROOT}/host/tools"
for TOOL in iree-compile iree-opt iree-dump-module iree-dump-parameters iree-encode-parameters; do
  cp "${HOST_BIN_DIR}/${TOOL}" "${OUT_DIR}/"
done

# --- Hexagon-DSP tree: the DSP-side skeleton .so ---------------------------
build_tree() {
  local build_dir="$1"
  local tracy="$2"
  local target="$3"
  local extra_flags=("${@:4}")
  if [[ "${tracy}" == "1" ]]; then
    extra_flags+=(-DIREE_ENABLE_RUNTIME_TRACING=ON -DIREE_TRACING_PROVIDER=tracy)
    # The Hexagon runtime uses IREE_TRACING_EXPERIMENTAL_CONTEXT_API=1. We
    # need it to be effective from the beginning, also in IREE's cmake
    # configure, so set it globally -- via CMAKE_PROJECT_INCLUDE rather than
    # CMAKE_C_FLAGS/CMAKE_CXX_FLAGS, since those would otherwise overwrite
    # (not add to) whatever flags the active toolchain file -- Hexagon's or
    # the Android NDK's -- already seeded there on this first configure.
    extra_flags+=(
      -DCMAKE_PROJECT_INCLUDE="${REPO_ROOT}/cmake/TracyExperimentalContextApi.cmake"
    )
  fi
  cmake -S "${IREE_SRC}" -B "${build_dir}" \
    "${COMMON_FLAGS[@]}" \
    "${extra_flags[@]}"
  cmake --build "${build_dir}" --target "${target}" -- -k 0
}
build_dsp_tree() {
  local build_dir="$1"
  local tracy="$2"
  # The toolchain sets CMAKE_SHARED_LINKER_FLAGS_INIT, which only seeds a new
  # cache. Clear its derived cache entry so existing build trees also pick up
  # toolchain changes instead of retaining stale global runtime libraries.
  build_tree "$build_dir" "$tracy" \
    iree_hexagon_plugins_plugins_runtime_hexagon_dsp_hexagon_dsp_skel \
    -U CMAKE_SHARED_LINKER_FLAGS \
    -DCMAKE_TOOLCHAIN_FILE="${REPO_ROOT}/cmake/HexagonToolchain.cmake" \
    -DIREE_BUILD_COMPILER=OFF \
    -DIREE_BUILD_TESTS=ON \
    -DIREE_BUILD_SAMPLES=OFF
  # Simulator tests are host-side CTest commands over shared modules built by
  # this DSP cross-build. Build all registered modules before invoking CTest;
  # they are TESTONLY and therefore not part of the default target.
  cmake --build "${build_dir}" --target iree_hexagon_sim_test_modules -- -k 0
  ctest --test-dir "${build_dir}" --output-on-failure -L test-type=hexagon-sim
}
build_dsp_tree "${BUILD_ROOT}/dsp" 0
build_dsp_tree "${BUILD_ROOT}/dsp-tracy" 1
DSP_SKEL_SO="${BUILD_ROOT}/dsp/runtime/plugins/iree_hexagon_plugins/plugins/runtime/hexagon/dsp/libhexagon_dsp_skel.so"
DSP_SKEL_SO_TRACY="${BUILD_ROOT}/dsp-tracy/runtime/plugins/iree_hexagon_plugins/plugins/runtime/hexagon/dsp/libhexagon_dsp_skel.so"
ls -l "$DSP_SKEL_SO" "$DSP_SKEL_SO_TRACY"

# --- Android tree(s): the ARM side of the Hexagon HAL driver ---------------
build_android_tree() {
  local build_dir="$1"
  local tracy="$2"
  # Build the default `all` target rather than naming iree-run-module/
  # iree-benchmark-module/limit_lifetime explicitly, for the same
  # name-mangling reason as the DSP tree above (limit_lifetime is one of our
  # own plugin targets); IREE_BUILD_COMPILER=OFF and IREE_BUILD_TESTS=OFF
  # already keep this configure's `all` scoped to just the runtime tools and
  # the Hexagon HAL driver / device tools.
  build_tree "$build_dir" "$tracy" \
    all \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DIREE_HOST_BIN_DIR="${HOST_BIN_DIR}" \
    -DIREE_BUILD_COMPILER=OFF \
    -DIREE_BUILD_TESTS=OFF \
    -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_HEXAGON_ANDROID_BUILD=ON \
    -DIREE_EXTERNAL_HAL_DRIVERS=hexagon
}
build_android_tree "${BUILD_ROOT}/android" 0
build_android_tree "${BUILD_ROOT}/android-tracy" 1

# --- Package the same zip layout as plugins/runtime/hexagon/BUILD.bazel's
# pkg_files/pkg_zip and integration_tests/device/tools/BUILD.bazel's --------
LIBCXX_SHARED="$(find "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt" \
  -path '*aarch64-linux-android/libc++_shared.so' | head -n1)"

package_zip() {
  local zip_path="$1"; shift
  local staging
  staging="$(mktemp -d)"
  while [[ $# -gt 0 ]]; do
    local dest="$1" src="$2"; shift 2
    mkdir -p "${staging}/$(dirname "${dest}")"
    cp "${src}" "${staging}/${dest}"
  done
  (cd "${staging}" && zip -r "${zip_path}" .)
  rm -rf "${staging}"
}

package_zip "${OUT_DIR}/hexagon_runtime_aarch64_android.zip" \
  "bin/iree-run-module" "${BUILD_ROOT}/android/tools/iree-run-module" \
  "bin/iree-benchmark-module" "${BUILD_ROOT}/android/tools/iree-benchmark-module" \
  "lib/libc++_shared.so" "${LIBCXX_SHARED}" \
  "lib/hexagon/libhexagon_dsp_skel.so" "${DSP_SKEL_SO}"

package_zip "${OUT_DIR}/hexagon_runtime_aarch64_android_tracy.zip" \
  "bin/iree-run-module" "${BUILD_ROOT}/android-tracy/tools/iree-run-module" \
  "bin/iree-benchmark-module" "${BUILD_ROOT}/android-tracy/tools/iree-benchmark-module" \
  "lib/libc++_shared.so" "${LIBCXX_SHARED}" \
  "lib/hexagon/libhexagon_dsp_skel.so" "${DSP_SKEL_SO_TRACY}"

package_zip "${OUT_DIR}/device_tools_aarch64_android.zip" \
  "bin/limit_lifetime" "$(find "${BUILD_ROOT}/android" -name limit_lifetime -type f | head -n1)"

# --- Unit Tests - not including integration tests, which need a device
ctest --test-dir "${BUILD_ROOT}/host" --output-on-failure -R iree_hexagon_plugins
