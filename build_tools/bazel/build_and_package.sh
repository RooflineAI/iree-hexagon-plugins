#! /usr/bin/bash

set -eux -o pipefail

# Host tools, packaged into hexagon_compiler_x86_64_linux.zip under bin/.
COMPILER_BIN_TARGETS=(
    @iree//tools:iree-compile
    @iree//tools:iree-dump-module
    @iree//tools:iree-dump-parameters
    @iree//tools:iree-encode-parameters
    @iree//tools:iree-opt
    @iree//tools:iree-run-module
    @iree//tools:iree-benchmark-module
    @llvm-project//lld:ld.lld
)
# Shared libraries the host tools load, packaged under lib/.
COMPILER_LIB_TARGETS=(
    @iree//lib:IREECompiler
)
# Already zips, passed on as they are.
ZIP_TARGETS=(
    //plugins/runtime/hexagon:hexagon_runtime_aarch64_android
    //plugins/runtime/hexagon:hexagon_runtime_aarch64_android_tracy
)
# Test helpers (limit_lifetime, rsync) that must not end up in the release,
# which publishes everything in build-artifacts-bazel: collected in their own
# directory and uploaded as a separate CI artifact.
TEST_ZIP_TARGETS=(
    //integration_tests/device/tools:device_tools_aarch64_android
)
TARGETS=(
    "${COMPILER_BIN_TARGETS[@]}"
    "${COMPILER_LIB_TARGETS[@]}"
    "${ZIP_TARGETS[@]}"
    "${TEST_ZIP_TARGETS[@]}"
)
COMPILER_PKG_NAME=hexagon_compiler_x86_64_linux

cd "$(dirname "$0")/../.."

# --- Configure
# The hermetic clang toolchain (--config=mlinux) needs no system compiler and
# links against glibc 2.28, so the released host tools run on older
# distributions than the machine they are built on.
# The host tools find libIREECompiler.so, which is packaged in a lib/ directory
# next to their bin/ directory, through the rpath $ORIGIN/../lib.
cat - >configured.bazelrc <<'EOF'
build --config=generic_clang
build --config=mlinux
build --linkopt=-Wl,-rpath,$ORIGIN/../lib
EOF

# --- Build
bazel build --verbose_failures "${TARGETS[@]}"

# --- Collect build artifacts
query_files() {
  bazel cquery --output=files "config($1, target)"
}
rm -rf build-artifacts-bazel test-artifacts-bazel
mkdir -p build-artifacts-bazel test-artifacts-bazel
COMPILER_PKG_DIR="$PWD/tmp/${COMPILER_PKG_NAME}"
rm -rf "${COMPILER_PKG_DIR}"
mkdir -p "${COMPILER_PKG_DIR}/bin" "${COMPILER_PKG_DIR}/lib"
for TARGET in "${COMPILER_BIN_TARGETS[@]}"
do
  mapfile -t target_files < <(query_files "$TARGET")
  cp "${target_files[@]}" "${COMPILER_PKG_DIR}/bin"
done
for TARGET in "${COMPILER_LIB_TARGETS[@]}"
do
  mapfile -t target_files < <(query_files "$TARGET")
  cp "${target_files[@]}" "${COMPILER_PKG_DIR}/lib"
done
chmod -R u+w "${COMPILER_PKG_DIR}"
# Fail loudly if a tool would not find the library after unpacking.
for TOOL in iree-compile iree-opt
do
  readelf -d "${COMPILER_PKG_DIR}/bin/${TOOL}" | grep -E 'Library (run)?path' \
    | grep -qF '$ORIGIN/../lib' \
    || { echo "${TOOL} lacks the \$ORIGIN/../lib rpath" >&2; exit 1; }
done
(cd "${COMPILER_PKG_DIR}" \
  && zip -r "$OLDPWD/build-artifacts-bazel/${COMPILER_PKG_NAME}.zip" bin lib)
rm -rf "${COMPILER_PKG_DIR}"
for TARGET in "${ZIP_TARGETS[@]}"
do
  mapfile -t target_files < <(query_files "$TARGET")
  cp "${target_files[@]}" build-artifacts-bazel
done
for TARGET in "${TEST_ZIP_TARGETS[@]}"
do
  mapfile -t target_files < <(query_files "$TARGET")
  cp "${target_files[@]}" test-artifacts-bazel
done
cp third-party/iree/LICENSE build-artifacts-bazel/LICENSE.txt
hexkl_license="$(bazel info execution_root)/$(
  bazel cquery --output=files @hexkl//:license_txt
)"
cp "$hexkl_license" build-artifacts-bazel/HEXKL_LICENSE.txt
git rev-list -1 HEAD >build-artifacts-bazel/git-hash.txt
chmod -R u+w build-artifacts-bazel test-artifacts-bazel
chmod a-x build-artifacts-bazel/*.zip test-artifacts-bazel/*.zip

# --- Unit Tests - not including integration tests, which need a device
bazel test //...
