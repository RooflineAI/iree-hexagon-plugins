#!/usr/bin/env bash
# Usage: integration_tests/profile_on_device.sh <path/to/model/dir> [serial]
# Import, compile and run an integration-test model (any number of inputs) on
# the device under the Tracy runtime. iree-run-module runs with
# TRACY_NO_EXIT=1, so it does not exit until a Tracy profiler has connected and received the trace.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# config
# $1: path to a model directory (the one holding model.yaml and model.py),
# e.g. integration_tests/models/micro/parallel-conv-max-1x32x64x64-f16
# $2 (optional): "serial" to run one dispatch at a time; default is parallel.
if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$2" != serial ) ]]; then
  echo "usage: $0 <path/to/model/dir> [serial]" >&2
  exit 1
fi
# Resolved before the cd below, so a path relative to the caller's cwd works.
MODEL_DIR="$(realpath "$1")"
if [[ ! -f "$MODEL_DIR/model.yaml" ]]; then
  echo "error: no model.yaml in $MODEL_DIR" >&2
  exit 1
fi
cd "$REPO"
MODEL_NAME="$(basename "$MODEL_DIR")"
WORK_DIR="${WORK_DIR:-/tmp/profile_on_device/$MODEL_NAME}" # local dir, where model is compiled
REMOTE_DIR=/data/local/tmp/${USER}/profile_on_device/$MODEL_NAME # dir on the phone to upload model to

EXTRA_COMPILE_FLAGS=""

RUN_FLAGS="" # parallel
if [[ "${2:-}" == serial ]]; then
  RUN_FLAGS+=" --hexagon_max_concurrent_dispatches=1"
fi


# --- Build ------------------------------------------------------------------
bazel build \
  @iree//tools:iree-compile \
  @llvm-project//lld:ld.lld \
  //plugins/runtime/hexagon:hexagon_runtime_aarch64_android_tracy

IREE_COMPILE="$REPO/$(bazel cquery --output=files @iree//tools:iree-compile 2>/dev/null | head -n1)"
LLD="$REPO/$(bazel cquery --output=files @llvm-project//lld:ld.lld 2>/dev/null | head -n1)"
RUNTIME_ZIP_TRACY="$REPO/$(bazel cquery --output=files \
  //plugins/runtime/hexagon:hexagon_runtime_aarch64_android_tracy 2>/dev/null | head -n1)"

# --- Import: torch model -> model.mlir + input*.npy -------------------------
# The import also writes io.txt, listing the entry function and the input and
# output file names, so the run below works for any number of inputs/outputs.
mkdir -p "$WORK_DIR"
.venv/bin/python -c "
from pathlib import Path
from integration_tests.modeltree.discover import MODELS_DIR
from integration_tests.modeltree.spec import load_model_spec
from integration_tests.stages.importing import import_model
model_dir = Path('$MODEL_DIR')
# The spec's name is its path below the model tree; outside the tree, fall
# back to the directory's own name.
root = MODELS_DIR if model_dir.is_relative_to(MODELS_DIR) else model_dir.parent
spec = load_model_spec(model_dir, root)
imported = import_model(spec, Path('$WORK_DIR'))
lines = [f'function {imported.function}']
lines += [f'input {p.name}' for p in imported.input_files]
lines += [f'output {name}' for name in imported.output_names]
Path('$WORK_DIR/io.txt').write_text('\\n'.join(lines) + '\\n')
"

FUNCTION=main
INPUT_FILES=()
IO_FLAGS=""
while read -r kind name; do
  case "$kind" in
    function) FUNCTION="$name" ;;
    input) INPUT_FILES+=("$WORK_DIR/$name"); IO_FLAGS+=" --input=@$name" ;;
    output) IO_FLAGS+=" --output=@$name" ;;
  esac
done < "$WORK_DIR/io.txt"

# --- Compile ----------------------------------------------------------------
"$IREE_COMPILE" "$WORK_DIR/model.mlir" -o "$WORK_DIR/model.vmfb" \
  --iree-hal-target-device=hexagon \
  --iree-input-type=auto \
  --iree-hexagon-v=79 \
  --iree-hexagon-features=+hvxv79,+hvx-length128b \
  --iree-opt-data-tiling=false \
  --iree-stream-resource-min-offset-alignment=128 \
  --iree-hexagon-enable-vtcm-tiling=true \
  --iree-hexagon-enable-hmx-matmul=true \
  --iree-hexagon-linker-path="$LLD" \
  --iree-llvmcpu-embedded-linker-path="$LLD"

# --- Deploy -----------------------------------------------------------------
adb push "$RUNTIME_ZIP_TRACY" "$REMOTE_DIR/runtime.zip"
adb shell "cd $REMOTE_DIR && unzip -oq runtime.zip && rm runtime.zip && chmod +x bin/iree-run-module"
adb push "$WORK_DIR/model.vmfb" ${INPUT_FILES[@]+"${INPUT_FILES[@]}"} "$REMOTE_DIR/"

# --- Run under Tracy --------------------------------------------------------
adb forward tcp:8086 tcp:8086

echo
echo "Running with TRACY_NO_EXIT=1; it will wait for a profiler before exiting."
echo "Attach the Tracy UI (0.11.2), or in another shell:"
echo "  ./third-party/iree/third_party/tracy/capture/build/tracy-capture -o /tmp/capture.tracy"
echo

adb shell "cd $REMOTE_DIR && \
  export DSP_LIBRARY_PATH=$REMOTE_DIR/lib/hexagon && \
  TRACY_NO_EXIT=1 bin/iree-run-module \
    --module=model.vmfb \
    --function=$FUNCTION \
    --device=hexagon \
    $IO_FLAGS \
    $RUN_FLAGS"
