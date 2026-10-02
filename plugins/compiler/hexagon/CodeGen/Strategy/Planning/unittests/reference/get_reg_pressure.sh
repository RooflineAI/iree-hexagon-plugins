#!/bin/bash

set -e

IN=${1:-config.mlir}
OUT=${2:-pressure.csv}
EXE=${3:-exe.vmfb}

TEMP_DIR=$(mktemp -d)
echo "$TEMP_DIR"

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
LLC_WITH_DEBUG=${LLC_WITH_DEBUG:-$(resolve_bazel_target @llvm-project//llvm:llc)}
LD_LLD=${LD_LLD:-$(resolve_bazel_target @llvm-project//lld:ld.lld)}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOG_ANALYZER="$SCRIPT_DIR/extract_reg_pressure.py"


if [ ! -f "$IREE_COMPILE" ]; then
  echo "error: IREE_COMPILE not found at $IREE_COMPILE" >&2
  exit 1
fi
if [ ! -f "$LLC_WITH_DEBUG" ]; then
  echo "error: LLC_WITH_DEBUG not found at $LLC_WITH_DEBUG" >&2
  exit 1
fi
if [ ! -f "$LD_LLD" ]; then
  echo "error: LD_LLD not found at $LD_LLD" >&2
  exit 1
fi
if [ ! -f "$LOG_ANALYZER" ]; then
  echo "error: LOG_ANALYZER script not found at $LOG_ANALYZER" >&2
  exit 1
fi

# iree-compile searches PATH for the hexagon linker (ld.lld) by that name.
PATH="$(dirname -- "$LD_LLD"):$PATH"


$IREE_COMPILE \
--iree-hal-target-device=hexagon \
--iree-input-type=auto \
--iree-hexagon-v=79 \
--mhvx=v79 \
--iree-hexagon-features=+hvxv79,+hvx-length128b \
--iree-hexagon-launch-config-selector=hexagon \
--iree-stream-resource-min-offset-alignment=128 \
--iree-opt-data-tiling=false \
--iree-hexagon-enable-vtcm-tiling \
--compile-from=executable-configurations \
--iree-hal-dump-executable-intermediates-to="$TEMP_DIR" \
--iree-llvmcpu-max-allowed-number-of-native-vectors=4906 \
--iree-hexagon-fail-on-stack-frames-larger-than=0 \
"$IN" -o "$EXE"

# register pressure info: --compile-from=executable-configurations should
# produce exactly one dispatch's *.opt.ll for a single-op input; bail out
# instead of silently only looking at one of several.
mapfile -t OPT_LL_FILES < <(ls "$TEMP_DIR"/*.opt.ll 2>/dev/null)
if [ "${#OPT_LL_FILES[@]}" -ne 1 ]; then
  echo "error: expected exactly one *.opt.ll in $TEMP_DIR, found ${#OPT_LL_FILES[@]}:" >&2
  printf '  %s\n' "${OPT_LL_FILES[@]}" >&2
  exit 1
fi
OPT_LL="${OPT_LL_FILES[0]}"

# run llc to get the virtual register count
"$LLC_WITH_DEBUG" -mtriple=hexagon -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b \
  -stop-before=greedy -debug-only=machine-scheduler -o /dev/null "$OPT_LL" >> "$TEMP_DIR/llc.log" 2>&1

# extract the count from all the debug output
"$LOG_ANALYZER" "$TEMP_DIR/llc.log" --per-region --csv "$OUT" > /dev/null
#cp $OPT_LL opt.ll

# cleanup
#rm -r $TEMP_DIR

cat -- "$OUT"
