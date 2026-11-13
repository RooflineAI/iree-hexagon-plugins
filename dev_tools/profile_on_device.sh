#!/usr/bin/env bash
# Usage: dev_tools/profile_on_device.sh [--serial] [--pmu-events=<list>] <path/to/model/dir>
#        dev_tools/profile_on_device.sh --list-pmu-events
# Import, compile and run an integration-test model (any number of inputs) on
# the device under the Tracy runtime. iree-run-module runs with
# TRACY_NO_EXIT=1, so it does not exit until a Tracy profiler has connected and received the trace.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PMU_TABLE="$REPO/plugins/runtime/hexagon/arm_dsp/pmu/hexagon_pmu_events_table.inc"

usage() {
  cat <<EOF
usage: $0 [--serial] [--pmu-events=<list>] <path/to/model/dir>
       $0 --list-pmu-events

  <path/to/model/dir>  model directory (the one holding model.yaml and model.py),
                       e.g. integration_tests/models/micro/parallel-conv-max-1x32x64x64-f16
  --serial             run one dispatch at a time on one DSP thread; default is
                       parallel
  --pmu-events=<list>  comma-separated Hexagon PMU events, up to 8, each a name
                       (HVXST_VTCM or HEX_PMU_EVENT_HVXST_VTCM) or a numeric ID
                       (0x0126); counters not given keep the runtime defaults
  --list-pmu-events    print the PMU event names, IDs and descriptions
EOF
}

# Prints "<id>  <name>  <description>" for every event in the PMU table, with
# the description taken from the comment above each definition.
list_pmu_events() {
  awk '
    /^ *\/\// { line = $0; sub(/^ *\/\/ */, "", line); desc = desc (desc == "" ? "" : " ") line; next }
    /^HEXAGON_PMU_EVENT_DEF\(/ {
      match($0, /HEX_PMU_EVENT_[A-Z0-9_]+/); name = substr($0, RSTART + 14, RLENGTH - 14)
      match($0, /0x[0-9A-Fa-f]+/); id = substr($0, RSTART, RLENGTH)
      sub(/^0x[0-9A-Fa-f]+: */, "", desc)
      printf "%s  %-40s  %s\n", id, name, desc
    }
    { desc = "" }
  ' "$PMU_TABLE"
}

# Prints the numeric ID of a PMU event given by number or by name (with or
# without the HEX_PMU_EVENT_ prefix, in any case).
pmu_event_id() {
  local event="$1"
  if [[ "$event" =~ ^(0[xX][0-9a-fA-F]+|[0-9]+)$ ]]; then
    echo "$event"
    return
  fi
  local name="${event^^}"
  name="HEX_PMU_EVENT_${name#HEX_PMU_EVENT_}"
  local id=""
  if [[ "$name" =~ ^[A-Z0-9_]+$ ]]; then
    id="$(sed -n "s/^HEXAGON_PMU_EVENT_DEF(${name}, \(0x[0-9A-Fa-f]*\))$/\1/p" "$PMU_TABLE")"
  fi
  if [[ -z "$id" ]]; then
    echo "error: unknown PMU event '$event' (see --list-pmu-events)" >&2
    return 1
  fi
  echo "$id"
}

# config
MODEL_ARG=""
SERIAL=false
PMU_EVENTS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial) SERIAL=true ;;
    --pmu-events=*) PMU_EVENTS="${1#--pmu-events=}" ;;
    --pmu-events) [[ $# -ge 2 ]] || { usage >&2; exit 1; }; PMU_EVENTS="$2"; shift ;;
    --list-pmu-events) list_pmu_events; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "error: unknown option '$1'" >&2; usage >&2; exit 1 ;;
    *)
      if [[ -n "$MODEL_ARG" ]]; then
        usage >&2
        exit 1
      fi
      MODEL_ARG="$1"
      ;;
  esac
  shift
done
if [[ -z "$MODEL_ARG" ]]; then
  usage >&2
  exit 1
fi

# Resolved here, before the long build, so a typo fails early.
PMU_EVENT_IDS=()
IFS=',' read -ra PMU_EVENT_LIST <<< "$PMU_EVENTS"
for event in ${PMU_EVENT_LIST[@]+"${PMU_EVENT_LIST[@]}"}; do
  event="${event//[[:space:]]/}"
  [[ -n "$event" ]] || continue
  id="$(pmu_event_id "$event")"
  PMU_EVENT_IDS+=("$id")
done
if [[ ${#PMU_EVENT_IDS[@]} -gt 8 ]]; then
  echo "error: ${#PMU_EVENT_IDS[@]} PMU events given, the PMU has 8 counters" >&2
  exit 1
fi

# Resolved before the cd below, so a path relative to the caller's cwd works.
MODEL_DIR="$(realpath "$MODEL_ARG")"
if [[ ! -f "$MODEL_DIR/model.yaml" ]]; then
  echo "error: no model.yaml in $MODEL_DIR" >&2
  exit 1
fi
cd "$REPO"
MODEL_NAME="$(basename "$MODEL_DIR")"
WORK_DIR="${WORK_DIR:-/tmp/profile_on_device/$MODEL_NAME}" # local dir, where model is compiled
REMOTE_DIR=/data/local/tmp/${USER}/profile_on_device/$MODEL_NAME # dir on the phone to upload model to

# The integration tests' device root; their kept modules are reused below.
DEVICE_ROOT="${DEVICE_ROOT:-/data/local/tmp/hexagon_it}"

COMPILE_FLAGS=()
RUN_FLAGS="--module_mode=mmap" # parallel
RUN_CASE=parallel # the integration-test run case with the same runtime flags
if [[ "$SERIAL" == true ]]; then
  # Without async worker threads inside VTCM-tiled kernels, too. Passed after
  # the model's extra_compile_flags, so it overrides a thread count set there.
  COMPILE_FLAGS+=(--iree-hexagon-vtcm-tile-num-threads=1)
  RUN_FLAGS+=" --hexagon_max_concurrent_dispatches=1"
  RUN_CASE=sequential
fi
if [[ ${#PMU_EVENT_IDS[@]} -gt 0 ]]; then
  RUN_FLAGS+=" --hexagon_pmu_events=$(IFS=,; echo "${PMU_EVENT_IDS[*]}")"
fi


# --- Build ------------------------------------------------------------------
bazel build \
  @iree//tools:iree-compile \
  @llvm-project//lld:ld.lld \
  //plugins/runtime/hexagon:hexagon_runtime_aarch64_android_tracy \
  //integration_tests/device/tools:device_tools_aarch64_android

IREE_COMPILE="$REPO/$(bazel cquery --output=files @iree//tools:iree-compile 2>/dev/null | head -n1)"
LLD="$REPO/$(bazel cquery --output=files @llvm-project//lld:ld.lld 2>/dev/null | head -n1)"
RUNTIME_ZIP_TRACY="$REPO/$(bazel cquery --output=files \
  //plugins/runtime/hexagon:hexagon_runtime_aarch64_android_tracy 2>/dev/null | head -n1)"
DEVICE_TOOLS_ZIP="$REPO/$(bazel cquery --output=files \
  //integration_tests/device/tools:device_tools_aarch64_android 2>/dev/null | head -n1)"

# --- Import: torch model -> model.mlir + input*.npy -------------------------
# The import also writes io.txt, listing the entry function, the input and
# output file names, the model's extra_compile_flags from model.yaml and its
# name in the model tree, so the compile, deploy and run below work for any
# model.
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
lines += [f'flag {flag}' for flag in spec.extra_compile_flags]
lines += [f'name {spec.name}']
Path('$WORK_DIR/io.txt').write_text('\\n'.join(lines) + '\\n')
"

FUNCTION=main
INPUT_FILES=()
IO_FLAGS=""
EXTRA_COMPILE_FLAGS=()
while read -r kind name; do
  case "$kind" in
    function) FUNCTION="$name" ;;
    input) INPUT_FILES+=("$WORK_DIR/$name"); IO_FLAGS+=" --input=@$name" ;;
    output) IO_FLAGS+=" --output=@$name" ;;
    flag) EXTRA_COMPILE_FLAGS+=("$name") ;;
    # as the integration tests name the case directory (test_models.py)
    name) CASE_PREFIX="${name//\//_}" ;;
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
  --iree-hexagon-enable-profiler-markers \
  --iree-hexagon-linker-path="$LLD" \
  --iree-llvmcpu-embedded-linker-path="$LLD" \
  ${EXTRA_COMPILE_FLAGS[@]+"${EXTRA_COMPILE_FLAGS[@]}"} \
  ${COMPILE_FLAGS[@]+"${COMPILE_FLAGS[@]}"}

# --- Deploy -----------------------------------------------------------------
adb push "$RUNTIME_ZIP_TRACY" "$REMOTE_DIR/runtime.zip"
adb push "$DEVICE_TOOLS_ZIP" "$REMOTE_DIR/device_tools.zip"
adb shell "cd $REMOTE_DIR && unzip -oq runtime.zip && unzip -oq device_tools.zip && rm runtime.zip device_tools.zip && chmod +x bin/iree-run-module bin/limit_lifetime bin/rsync"
# Send the module with the integration tests' push_module: cache models on the phone
.venv/bin/python -c "
from pathlib import Path
from integration_tests.stages.running import push_module
from integration_tests.tool_paths import resolve_host_tool
push_module(
    module_vmfb=Path('$WORK_DIR/model.vmfb'),
    remote_module='$REMOTE_DIR/model.vmfb',
    kept_module='$DEVICE_ROOT/modules/${CASE_PREFIX}_hexagon-vtcm-hmx_$RUN_CASE/module.vmfb',
    host_rsync=resolve_host_tool('rsync', None),
    device_rsync='$REMOTE_DIR/bin/limit_lifetime $REMOTE_DIR/bin/rsync',
    timeout=4 * 3600.0,
)
"
if [[ ${#INPUT_FILES[@]} -gt 0 ]]; then
  adb push "${INPUT_FILES[@]}" "$REMOTE_DIR/"
fi

# --- Run under Tracy --------------------------------------------------------
adb forward tcp:8086 tcp:8086

echo
echo "Running with TRACY_NO_EXIT=1; it will wait for a profiler before exiting."
echo "Attach the Tracy UI (0.11.2), or in another shell:"
echo "  ./third-party/iree/third_party/tracy/capture/build/tracy-capture -o /tmp/capture.tracy"
echo

adb shell "cd $REMOTE_DIR && \
  echo 1000 > /proc/self/oom_score_adj && \
  export DSP_LIBRARY_PATH=$REMOTE_DIR/lib/hexagon && \
  TRACY_NO_EXIT=1 bin/iree-run-module \
    --module=model.vmfb \
    --function=$FUNCTION \
    --device=hexagon \
    $IO_FLAGS \
    $RUN_FLAGS"
