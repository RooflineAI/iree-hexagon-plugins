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
# The hermetic clang toolchain (--config=mlinux) needs no system compiler and
# links against glibc 2.28, so the released host tools run on older
# distributions than the machine they are built on.
cat - >configured.bazelrc <<EOF
build --config=generic_clang
build --config=mlinux
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
