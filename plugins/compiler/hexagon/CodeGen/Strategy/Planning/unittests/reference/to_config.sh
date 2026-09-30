#!/bin/bash

set -e

IN=${1:-example.mlir}
OUT=${2:-config.mlir}

# The git root (roof-mlir)
# we run all bazel commands (to find the relevant targets) from the root
git_root() {
  if [ -z "${GIT_ROOT:-}" ]; then
    GIT_ROOT=$(git rev-parse --show-toplevel) || {
      echo "error: 'git rev-parse --show-toplevel' failed - run this from inside the roof-mlir checkout" >&2
      exit 1
    }
  fi
  printf '%s' "$GIT_ROOT"
}

# Resolves a Bazel target label to the fully qualified path of
# its built output
resolve_bazel_target() {
  local tgt="$1" root rel
  root=$(git_root)
  rel=$(cd "$root" && bazel cquery --output=files "$tgt" 2>/dev/null | head -n1)
  if [ -z "$rel" ]; then
    echo "error: bazel cquery for $tgt produced no output" >&2
    exit 1
  fi
  printf '%s/%s' "$root" "$rel"
}

IREE_COMPILE=${IREE_COMPILE:-$(resolve_bazel_target @iree//tools:iree-compile)}
TILING_PIPELINE=${TILING_PIPELINE:-iree}

if [ ! -f "$IREE_COMPILE" ]; then
  echo "error: IREE_COMPILE not found at $IREE_COMPILE" >&2
  exit 1
fi

$IREE_COMPILE \
--iree-hal-target-device=hexagon \
--iree-input-type=auto \
--iree-hexagon-v=79 \
--mhvx=v79 \
--iree-opt-data-tiling=false \
--iree-hexagon-features=+hvxv79,+hvx-length128b \
--iree-hexagon-launch-config-selector=hexagon \
--iree-stream-resource-min-offset-alignment=128 \
--iree-hexagon-enable-vtcm-tiling \
--iree-hexagon-launch-config-selector=hexagon \
--compile-to=executable-configurations \
--iree-llvmcpu-max-allowed-number-of-native-vectors=4906 \
${EXTRA_FLAGS:-} \
"$IN" -o "$OUT"
