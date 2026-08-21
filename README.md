# IREE Hexagon Plugin

This repo contains experimental IREE compiler and runtime plugins for targeting
the Qualcomm Hexagon NPU. It serves as a temporary staging area for the
prototype code that is intended to be upstreamed into the experimental folder
of the main IREE repo `https://github.com/iree-org/iree/`.

## Submodules

The `third-party/iree` and `third-party/hexagon-mlir` submodules each need a
handful of patches applied to their working tree for the Bazel build to work
(see `patches/iree/` and `patches/hexagon-mlir/` respectively - one `.patch`
file per topic/issue). The `iree` patches address current limitations of
upstream IREE being used as a submodule (those need to be fixed upstream); the
`hexagon-mlir` patches mostly bring its sources up to date with the LLVM/MLIR
version this repo builds against.

Run this after cloning and after any `git submodule update` (which resets a
submodule to its pinned commit and wipes these edits):

```sh
build_tools/apply_submodule_patches.sh
```

It's idempotent - safe to re-run any time, including when the patches are
already applied.

## Dependencies

### Runtime

The build of the runtime needs the Android ADK 28.2.13676358 to be installed:

```sh
NDK_VERSION=r28c
NDK_FOLDER=28.2.13676358
NDK_ZIP="android-ndk-${NDK_VERSION}-linux.zip"
NDK_SHA256="dfb20d396df28ca02a8c708314b814a4d961dc9074f9a161932746f815aa552f"
wget https://dl.google.com/android/repository/${NDK_ZIP} -O /tmp/ndk.zip
echo "${NDK_SHA256} /tmp/ndk.zip" | sha256sum -c -
sudo unzip -q /tmp/ndk.zip -d /opt
sudo mkdir -p /opt/android-sdk/ndk
sudo mv "/opt/android-ndk-${NDK_VERSION}" "/opt/android-sdk/ndk/${NDK_FOLDER}"
```

## Configuring

```sh
cat - >configured.bazelrc << EOF
build --action_env CC=/usr/lib/llvm-19/bin/clang
build --action_env CXX=/usr/lib/llvm-19/bin/clang++
build --config=generic_clang
build --incompatible_strict_action_env
EOF
```

## Building

### Compiler

The compiler part is built for the host:

```sh
bazel build @iree//tools:iree-compile
```

### Runtime

The runtime part is built for Android on aarch64 plus Hexagon.
It is produced in form a zip file to be deployed to the Android device:

```sh
bazel build //plugins/runtime/hexagon:hexagon_runtime_aarch64_android
```

A variant including tracing can be build using:

```sh
bazel build //plugins/runtime/hexagon:hexagon_runtime_aarch64_android_tracy
```
