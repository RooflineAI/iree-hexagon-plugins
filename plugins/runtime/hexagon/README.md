# Qualcomm Hexagon HAL Driver (`hexagon`)

## Directory Structure

- `.`: HAL driver implementation on the ARM host side, all sources compiled for
       ARM
- `registration`: implementation of HAL driver registation, compiled for ARM
- `serialize` serialization of data for transmission from ARM host to DSP, can
              be unit-tested in isolation, can be compiled for any architecture,
              runs on ARM host side
- `test`: unit tests for the parts that can be unit-tested
- `interface`: RPC interface definition, ARM host calls the functions, DSP
               provides the functions, some ARM and some DSP sources are
               generated from this
- `dsp`: implementation on the DSP side, all sources compiled for DSP
- `arm_dsp`: headers for data structures shared between ARM host and DSP, see
             also README.md in this dir

## Executable Imports

Hexagon DMA/HexKL/HexagonMem symbols and the generic helper symbols used by the
lowering (`malloc`, `free`, `memrefCopy`) are resolved natively by the DSP
dynamic loader from `libhexagon_dsp_skel.so`.

## Building

The Hexagon SDK contains x86_64 binaries. The target device is an Android phone
with an aarch64 CPU and a Hexagon DSP. Thus, building is done by cross-compiling
on x86_64.

Integration tests on remote Hexagon devices (e.g. //integration_tests/hexagon)
are not built or run via Bazel; they run directly with `pytest` against
installed wheels on a machine with ADB access to a physical phone, see
`integration_tests/hexagon/README.md`.

Please note that cross-compiling requires an Android NDK to be present on the
system. It can be installed using
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

The cross-compile command is:
```sh
bazel build //plugins/runtime/hexagon:hexagon_runtime_aarch64_android
```

The output file lands at:
```sh
bazel-bin/plugins/runtime/hexagon/hexagon_runtime_aarch64_android.zip
```
(the tracy-enabled build, see [Profiler](#profiler) below, produces
`hexagon_runtime_aarch64_android_tracy.zip` alongside it.)

This zip file contains two binaries and some support files. The Hexagon HAL
driver is compiled directly into both binaries (a static HAL driver
registered via `build_tools/bazel/load_external_hal_drivers.bzl`) - there is
no separate runtime plugin `.so` to load:

- `bin/iree-benchmark-module`:
  This is the IREE runtime for benchmarking. It contains the ARM host part of
  the Hexagon runtime.
- `bin/iree-run-module`:
  This is the main IREE runtime. It contains the ARM host part of the Hexagon
  runtime.
- `lib/hexagon/libhexagon_dsp_skel.so`:
  This is the DSP part of the Hexagon runtime.
- `lib/hexagon/*.farf`:
  These files contains debug flags for the binary with the matching name.
  The flags turn on debug output from the DSP side to `adb logcat -s adsprpc`.
- `lib/libc++_shared.so`:
  Shared C++ library from the Android NDK used for building. The Android NDK
  docs state that the exact C++ library used for building has to be used for
  executing the binary.

## Deployment

Copy `hexagon_runtime_aarch64_android.zip` (described in the previous
section) to the Android phone. Unzip it:
```sh
unzip hexagon_runtime_aarch64_android.zip
```
Set the environment variable `DSP_LIBRARY_PATH` to the absolute path of the
directory that contains `libhexagon_dsk_skel.so`:
```sh
export DSP_LIBRARY_PATH=$(pwd)/lib/hexagon
```

## DSP Debug Output

In order to activate DSP debug output, create a file named
`iree-run-module.farf` with contents `0x1f` in the directory pointed to by
`DSP_LIBRARY_PATH`.

Run `adb logcat -s adsprpc` to receive the debug output while executing
`iree-run-module`.

## Listing all Hexagon Devices

Run this command to get a list of all devices found:
```sh
iree-run-module --dump_devices
```
Look at devices with `hexagon` in the URI.

## Device URI and Options

To make `iree-run-module` use the Hexagon runtime, specify `--device=hexagon`.
It is possible to specify the CDSP explicitly using `--device=hexagon://CDSP`.
To pass options, append `?<key1>=<value1>&<key2>=<value2>&<key3>=<value3>`.
If using the short URI, it needs to be `--device=hexagon:?<key1>=...`.

Note: Watch out for the special meaning of `?` and `&` in shells. Those
characters might have to be escaped or quoted.

The following options are available:
  - `verbose=true`: Turn on verbose (debug) output. (This may make the runtime slow.)
  - `dsp_status_notify=true`: Turn on DSP status notifications.

## OnePlus 13 Phone

The DSP in the OnePlus 13 phone is a CDSP, so use `--device=hexagon://CDSP`.

## Profiler

Hexagon currently has limited profiler support.
The profiler consists of zone data registration on the DSP to export to the host (sampling supported on the host, but not the DSP!) through Tracy.
The execution time is currently implemented through linear mapping and timeline synchronization on command buffer execution dispatch and end.
For this purpose, the RPC execution time is considered negligible.

The expected Tracy version is `0.11.2`. It is needed to forward the port used by Tracy in the phone:
```sh
adb forward tcp:8086 tcp:8086
```
Example script to run tracing:
```sh
DIRECTORY=/data/local/tmp/{your_path}
bazel build //plugins/runtime/hexagon:hexagon_runtime_aarch64_android_tracy
adb shell rm -rf $DIRECTORY/bin $DIRECTORY/lib $DIRECTORY/hexagon_runtime_aarch64_android_tracy.zip
adb push bazel-bin/plugins/runtime/hexagon/hexagon_runtime_aarch64_android_tracy.zip $DIRECTORY
adb shell unzip $DIRECTORY/hexagon_runtime_aarch64_android_tracy.zip -d $DIRECTORY
adb shell chmod +x $DIRECTORY/bin/iree-run-module
adb shell "export DSP_LIBRARY_PATH=$DIRECTORY/farf && TRACY_NO_EXIT=1 $DIRECTORY/bin/iree-run-module --module=$DIRECTORY/model.vmfb --input=@$DIRECTORY/input0.npy --input=@$DIRECTORY/input1.npy --input=@$DIRECTORY/input2.npy --device=hexagon
```
On a different shell, retrieve the output from profiler:
```sh
./third-party/iree/third_party/tracy/capture/build/tracy-capture -o /tmp/capture.tracy
```
Or alternatively, open the Tracy UI and connect to the listed available client to get the output directly.


### PMU Event Selection

Profiler supports configuration for Hexagon's PMU.
This unit contains 8 performance counters that can extract information such as cache misses or committed instructions.
The PMU configuration assumes that only one command buffer is executed at once and will output garbage otherwise.
Events can be configured using an `iree-run-module` flag:

```sh
iree-run-module \
  --device=hexagon \
  --hexagon_pmu_events=0x0003,0x0004
```

`--hexagon_pmu_events` accepts a comma-separated list (or multiple uses of the
flag) of numeric event IDs (decimal or hex, e.g. `0x0003`). Up to 8 events are
used; if fewer are provided, the remaining counters keep the runtime defaults.
The numeric event IDs are listed an explained in [hexagon_pmu_events_table.inc](arm_dsp/pmu/hexagon_pmu_events_table.inc).

These pmu events are displayed in a different plot and are only registered for
`KERNEL` zones and generated-code `MARKER` zones in the DSP.
Other zones currently do not have any PMU information plotted.
These values values are extracted at the beginning and end of the zone and are not interpolated,
therefore they remain constant over the duration of their corresponding zone.

### Some notable PMU events

See `hexagon_pmu_events_ids.h` for IDs:
- Committed work: `HEX_PMU_EVENT_COMMITTED_PKT_ANY`, `HEX_PMU_EVENT_COMMITTED_INSTS`
- Thread utilization: `HEX_PMU_EVENT_COMMITTED_PKT_T0`, `HEX_PMU_EVENT_COMMITTED_PKT_3_THREAD_RUNNING`
- Vector utilization: `HEX_PMU_EVENT_HVX_ACTIVE`, `HEX_PMU_EVENT_HVX_PKT`
- Memory pressure (scalar): `HEX_PMU_EVENT_L2_DU_READ_MISS`, `HEX_PMU_EVENT_L2_DU_STORE_MISS`
- VTCM usage/contention: `HEX_PMU_EVENT_VTCM_FIFO_FULL_CYCLES`, `HEX_PMU_EVENT_TCM_DU_ACCESS`
- Contention/stalls: `HEX_PMU_EVENT_VTCM_FIFO_FULL_CYCLES`, `HEX_PMU_EVENT_ANY_DU_REPLAY`
- TLB pressure: `HEX_PMU_EVENT_JTLB_MISS`, `HEX_PMU_EVENT_ITLB_MISS`
- Thermal / throttling: `HEX_PMU_EVENT_THREAD_LMH_THROTTLE`

## Technical notes

### Memory allocation

Memory allocation can be done through `iree_hal_hexagon_mem_alloc_create`. Its implementation is in `mem_alloc.c`.
There are three alternative allocations that can be done (available in `iree_hal_hexagon_mem_kind_e`). They represent DSP, host and shared allocations.
Note that the behavior of the allocated memory differs depending on whether this framework is used or not.
For example, when passing an allocation through an RPC as an argument, flushing the cache for synchronization with the host at the end of the RPC is automated.
It must be done manually when not passed as an argument though.
