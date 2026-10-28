# Developer tools

Helper scripts for working on a local machine with a device attached. They are
not run or tested in CI.

## Profiling a model by hand

To look at a trace in the Tracy UI, `profile_on_device.sh` builds the tools,
imports and compiles one model from the integration-test model tree, and runs it
on the device under the Tracy runtime with `TRACY_NO_EXIT=1`:

```shell
dev_tools/profile_on_device.sh integration_tests/models/micro/parallel-conv-max-1x32x64x64-f16 [serial]
```

`serial` adds `--hexagon_max_concurrent_dispatches=1`; without it dispatches
run in parallel. The run waits until a profiler connects on port 8086: attach
the Tracy UI (0.11.2), or run `tracy-capture -o <file>.tracy` in another shell.

The script reuses the integration tests' import and module push
(`integration_tests/stages`), so modules already kept on the device by a test
run are not pushed again. See [`integration_tests/README.md`](../integration_tests/README.md)
for the model tree and the automated profile checks.
