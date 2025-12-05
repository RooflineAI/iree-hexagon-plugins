# Qualcomm Hexagon HAL Driver (`hexagon`)

This is a skeleton of the HAL driver for Qualcomm Hexagon. For now, it
implements most interfaces needed by a HAL driver.

## Directory Structure

- `.`: HAL driver implementation on the ARM side, all sources compiled for ARM
- `registration`: implementation of HAL driver registation, compiled for ARM
- `serialize` serialization of data for transmission from ARM to DSP, can be
              unit-tested in isolation, can be compiled for any architecture,
              runs on ARM side
- `test`: unit tests for the parts that can be unit-tested
- `interface`: RPC interface definition, ARM call the functions, DSP provides
               the functions, some ARM and some DSP sources are generated from
               this
- `dsp`: implementation on the DSP side, all sources compiled for DSP
- `arm_dsp`: headers for data structures shared between ARM and DSP, see also
             README.md in this dir

## Building

The Hexagon SDK contains x86_64 binaries. The target device is an Android phone
with an aarch64 CPU and a Hexagon DSP. Thus, building is done by cross-compiling
on x86_64:

```sh
bazel build --platforms=//platform:aarch64_android @iree//tools:iree-run-module
```

Please note that this requires an Android NDK to be present on the system. It
can be installed on Ubuntu Linux using
```sh
sudo apt install google-android-ndk-r26c-installer
```

There are two files resulting from the above build command:
- `bazel-out/aarch64-*/bin/external/iree/tools/iree-run-module`:
  This is the main IREE runtime, which contains the ARM part of the Hexagon
  runtime.
- `bazel-out/aarch64-*/bin/external/patio_runtime/hexagon/dsp/lib/hexagon/libhexagon_dsp_skel.so`:
  This is the DSP part of the Hexagon runtime.

## Deployment

Copy `iree-run-module` and `libhexagon_dsp_skel.so` (mentioned in th previous
section) to the Android phone. Set the environment variable `DSP_LIBRARY_PATH``
to the absolute path to the directory that contains `libhexagon_dsk_skel.so`.

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
