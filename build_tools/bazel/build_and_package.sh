#! /usr/bin/bash

set -eux -o pipefail

TARGETS=(
    @iree//tools:iree-compile
    @iree//tools:iree-dump-module
    @iree//tools:iree-dump-parameters
    @iree//tools:iree-encode-parameters
    @iree//tools:iree-opt
    @iree//tools:iree-run-module
    @iree//tools:iree-benchmark-module
    @llvm-project//lld:ld.lld
    //integration_tests/device/tools:device_tools_aarch64_android
    //plugins/runtime/hexagon:hexagon_runtime_aarch64_android
    //plugins/runtime/hexagon:hexagon_runtime_aarch64_android_tracy
)

cd "$(dirname "$0")/../.."

# --- Configure
cat - >configured.bazelrc <<EOF
build --action_env CC=/usr/lib/llvm-19/bin/clang
build --action_env CXX=/usr/lib/llvm-19/bin/clang++
build --config=generic_clang
build --incompatible_strict_action_env
EOF

# --- Build
bazel build --verbose_failures "${TARGETS[@]}"

# --- Collect build artifacts
mkdir -p build-artifacts-bazel
for TARGET in "${TARGETS[@]}"
do
  mapfile -t target_files < <(
    bazel cquery --output=files "config($TARGET, target)"
  )
  cp "${target_files[@]}" build-artifacts-bazel
done
cp third-party/iree/LICENSE build-artifacts-bazel/LICENSE.txt
hexkl_license="$(bazel info execution_root)/$(
  bazel cquery --output=files @hexkl//:license_txt
)"
cp "$hexkl_license" build-artifacts-bazel/HEXKL_LICENSE.txt
git rev-list -1 HEAD >build-artifacts-bazel/git-hash.txt
chmod -R u+w build-artifacts-bazel
chmod a-x build-artifacts-bazel/*.zip

# --- Unit Tests - not including integration tests, which need a device
bazel test //...
