We need to use trampoline scripts to call hexagon tools due to bazel's
sandboxing restrictions. This folder contains those trampoline scripts.

The hexagon SDK is downloaded to external/hexagon_sdk/. The Hexagon toolchain
tools are located inside this SDK.

The Hexagon toolchain definition is in toolchain/hexagon_toolchain_config.bzl.
The tool_path instances can refer to absolute paths and relative paths inside
the toolchain/ directory (relative paths starting with .. are not accepted by
bazel). This is enforced by bazel's tool_path implementation in order to respect
the sandboxing and avoiding to break out of the sandbox accidentally.

It is not possible to use absolute paths, because it is not possible to know the
absolute location to which the Hexagon SDK has been downloaded. Bazel prevents
obtaining the absolute path of the download location of the Hexagon SDK in order
to enforce sandboxing.

Using a relative path from the toolchain directory to the downloaded Hexagon SDK
would have to use ../external/hexagon_sdk/..., which is a relative path that
goes to a sibling directory of the current directory. Because tool_path only
accepts relative paths to locations inside the current directory (the toolchain
directory), this relative path cannot be used.

Thus, trampoline scripts located somewhere inside the toolchain/ dir are the
only option. The term "trampoline" means call a wrapper that we can reach and
the wrapper calls the tool out of our reach that we actually wanted to call.
In this case, the toolchain will refer to the trampoline scripts in this folder,
e.g., to "hexagon_trampolines/clang" (which is a relative path accepted by
tool_path). The trampoline script will then call the clang binary in
external/hexagon_sdk/...subdirs.../clang.

FIXME: If there is another (better) option besides trampoline scripts, let's
adapt / improve this.

For reference, the trampoline approach is inspired from here:
https://skia.googlesource.com/skia/+/c5c98fc67cc4/toolchain/linux_amd64_toolchain_config.bzl

The generate.py script is a generator for the trampolines, which have almost
the same contents.
