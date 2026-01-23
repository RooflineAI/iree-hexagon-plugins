# Qualcomm Hexagon HAL Driver (`hexagon`)

This is a skeleton of the HAL driver for Qualcomm Hexagon. For now, it
implements most interfaces needed by a HAL driver.

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

## Building

The Hexagon SDK contains x86_64 binaries. The target device is an Android phone
with an aarch64 CPU and a Hexagon DSP. Thus, building is done by cross-compiling
on x86_64:

```sh
bazel build --platforms=//platform:aarch64_android @patio_runtime//hexagon:hexagon_runtime
```

Please note that this requires an Android NDK to be present on the system. It
can be installed using
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

The output file can be queried:
```sh
bazel cquery --output=files --platforms=//platform:aarch64_android @patio_runtime//hexagon:hexagon_runtime
```
It usually returns:
```sh
bazel-out/aarch64-dbg/bin/external/_main~_repo_rules~patio_runtime/hexagon/hexagon_runtime.zip
```

This zip file contains two binaries and some support files:
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

Copy `hexagon_runtime.zip` (described in the previous section) to the Android
phone. Unzip it:
```sh
unzip hexagon_runtime.zip
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
