# Tracy patches

Tracy is IREE's nested submodule `third-party/iree/third_party/tracy`. Both
the Bazel (`@tracy_client`) and the CMake builds compile its client from that
checkout, so these patches change the Tracy client linked into the runtime
tools, e.g. `iree-run-module`.

| Patch | Category | Reason |
| --- | --- | --- |
| [tracy_android_mappings_never_destroyed.patch](tracy_android_mappings_never_destroyed.patch) | Upstream bug fix | Never destroy the Android-only `/proc/self/maps` cache used to make code readable for symbol queries. With `TRACY_NO_EXIT`, Tracy's threads keep answering the capture's symbol queries during `exit()`, after the function-local static was destroyed; a lookup miss then reassigns it and frees its buffer a second time, aborting profiled runs. Still present in upstream `master`. |
