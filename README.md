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

```sh
bazel build @iree//tools/...
```
