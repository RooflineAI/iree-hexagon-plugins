# Running an example

Assumes you have already produced a `.vmfb` — see [compiling.md](compiling.md).

## Choose an example

Same idea as in [compiling.md](compiling.md), plus the entry point and input shapes,
which are the only things that actually differ between the two examples:

```sh
EXAMPLE=matmul
FUNCTION=matmul
INPUTS="--input=32x32xf16=1 --input=32x32xf16=1"
```

```sh
# ...or the attention layer:
EXAMPLE=attention_layer
FUNCTION=main
INPUTS="--input=4x1024x128xf16=1 --input=4x1024x128xf16=1 --input=4x1024x128xf16=1"
```

Then, matching the variables from [compiling.md](compiling.md):

```sh
OUT="$PWD/examples/$EXAMPLE/dump"
```

`$INPUTS` is deliberately left unquoted where it is used below, so that it splits into
one argument per `--input=`.

## Run on the Hexagon DSP

**Requires** an Android phone with a Hexagon DSP, connected over ADB, and the Android
NDK installed (see the top-level [README](../README.md)). Verify the device first:

```sh
adb devices
```

### 1.1 Build and deploy the runtime

The runtime is cross-compiled for Android AArch64 plus the DSP and packaged as a zip.
It does not depend on which example you picked, so this only needs doing once:

```sh
bazel build //plugins/runtime/hexagon:hexagon_runtime_aarch64_android

RUNTIME_ZIP="$PWD/$(bazel cquery --output=files \
  //plugins/runtime/hexagon:hexagon_runtime_aarch64_android 2>/dev/null | head -n1)"
```

Push and unpack it:

```sh
REMOTE_DIR=/data/local/tmp/hexagon-example

adb shell "rm -rf '$REMOTE_DIR/bin' '$REMOTE_DIR/lib'"
adb shell "mkdir -p '$REMOTE_DIR'"
adb push "$RUNTIME_ZIP" "$REMOTE_DIR"
adb shell "unzip -o '$REMOTE_DIR/hexagon_runtime_aarch64_android.zip' -d '$REMOTE_DIR'"
adb shell "chmod +x '$REMOTE_DIR/bin/iree-run-module'"
```

The zip contains:

| Path | Contents |
|---|---|
| `bin/iree-run-module` | IREE runtime for the ARM host side, with the Hexagon HAL driver linked in. |
| `bin/iree-benchmark-module` | Same, for benchmarking. |
| `lib/hexagon/libhexagon_dsp_skel.so` | The DSP-side runtime. |
| `lib/libc++_shared.so` | NDK C++ runtime, required to match the build. |

### 1.2 Push the model

```sh
adb push "$OUT/$EXAMPLE.vmfb" "$REMOTE_DIR/"
```

### 1.3 Run

`DSP_LIBRARY_PATH` must point at the directory holding `libhexagon_dsp_skel.so`:

```sh
adb shell "export DSP_LIBRARY_PATH=$REMOTE_DIR/lib/hexagon && \
  $REMOTE_DIR/bin/iree-run-module \
    --module=$REMOTE_DIR/$EXAMPLE.vmfb \
    --function=$FUNCTION \
    $INPUTS \
    --device=hexagon \
    --output=@$REMOTE_DIR/$EXAMPLE-output0.npy"
```

## DSP-side debug logging

DSP output does not appear on stdout; it goes to the Android log. Enable it by writing a
FARF mask next to the DSP library, named after the binary:

```sh
adb shell "printf '0x1f\n' > $REMOTE_DIR/lib/hexagon/iree-run-module.farf"
```

Then watch it while the model runs, in a second terminal:

```sh
adb logcat -s adsprpc
```

## Profiling with Tracy

Build the tracing variant of the runtime and deploy it the same way:

```sh
bazel build //plugins/runtime/hexagon:hexagon_runtime_aarch64_android_tracy

RUNTIME_ZIP="$PWD/$(bazel cquery --output=files \
  //plugins/runtime/hexagon:hexagon_runtime_aarch64_android_tracy 2>/dev/null | head -n1)"

REMOTE_DIR=/data/local/tmp/hexagon-example

adb shell "rm -rf '$REMOTE_DIR/bin' '$REMOTE_DIR/lib'"
adb shell "mkdir -p '$REMOTE_DIR'"
adb push "$RUNTIME_ZIP" "$REMOTE_DIR"
adb shell "unzip -o '$REMOTE_DIR/hexagon_runtime_aarch64_android_tracy.zip' -d '$REMOTE_DIR'"
adb shell "chmod +x '$REMOTE_DIR/bin/iree-run-module'"
```

Forward the Tracy port and run with `TRACY_NO_EXIT=1` so the process waits for a
collector to attach:

```sh
adb forward tcp:8086 tcp:8086

adb shell "export DSP_LIBRARY_PATH=$REMOTE_DIR/lib/hexagon && \
  TRACY_NO_EXIT=1 $REMOTE_DIR/bin/iree-run-module \
    --module=$REMOTE_DIR/$EXAMPLE.vmfb \
    --function=$FUNCTION \
    $INPUTS \
    --device=hexagon"
```

Then attach the Tracy UI (expected version `0.11.2`), or capture headlessly.
DSP hardware counters can be selected per run:

```sh
--hexagon_pmu_events=0x0003,0x0004
```

See [`plugins/runtime/hexagon/README.md`](../plugins/runtime/hexagon/README.md) for the
profiler details and the full PMU event list.
