# Developer tools

Helper scripts for working on a local machine with a device attached. They are
not run or tested in CI.

## Profiling a model by hand

To look at a trace in the Tracy UI, `profile_on_device.sh` builds the tools,
imports and compiles one model from the integration-test model tree, and runs it
on the device under the Tracy runtime with `TRACY_NO_EXIT=1`:

```shell
dev_tools/profile_on_device.sh [--serial] [--pmu-events=<list>] integration_tests/models/micro/parallel-conv-max-1x32x64x64-f16
```

`--serial` adds `--hexagon_max_concurrent_dispatches=1` to the run and
`--iree-hexagon-vtcm-tile-num-threads=1` to the compile, so one dispatch runs at
a time on a single DSP thread; without it dispatches run in parallel, and
VTCM-tiled kernels split their loops across worker threads.
`--pmu-events=HVXST_VTCM,HVXST_L2,0x0118,...` selects up to 8 PMU counters
(`--hexagon_pmu_events`, see
[`plugins/runtime/hexagon/README.md`](../plugins/runtime/hexagon/README.md)),
each by name, with or without the `HEX_PMU_EVENT_` prefix, or by numeric ID;
`--list-pmu-events` lists the names. The counter values are only meaningful
with `--serial`. The run waits until a profiler
connects on port 8086: attach the Tracy UI (0.11.2), or run
`tracy-capture -o <file>.tracy` in another shell.

The script reuses the integration tests' import and module push
(`integration_tests/stages`), so modules already kept on the device by a test
run are not pushed again. See [`integration_tests/README.md`](../integration_tests/README.md)
for the model tree and the automated profile checks.
