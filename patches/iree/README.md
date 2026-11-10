# IREE patches

These patches integrate the compiler/runtime plugins and allow IREE to build
as a dependency of this repository. Most address embedding and API exposure.

| Patch | Category | Reason |
| --- | --- | --- |
| [iree_extensions_embedded_paths.patch](iree_extensions_embedded_paths.patch) | Shared dependency integration | Configure LLVM and torch-mlir in IREE's extension from injected sources, enable the Hexagon backend and point StableHLO at the shared top-level checkout. Works with the ownership patch below. |
| [iree_dispatch_creation_max_binding_bytes.patch](iree_dispatch_creation_max_binding_bytes.patch) | Compiler feature (to upstream) | Add `--iree-dispatch-creation-max-dispatch-binding-bytes`: elementwise fusion, multi-use producer fusion, dispatch region formation and producer cloning do not grow a dispatch beyond that many bytes of bound tensors. The DSP maps all bindings of a dispatch at once into about 3.75 GiB, so fusion must not merge e.g. all shards of a split table into one dispatch. Drop once upstreamed. |
| [iree_global_opt_split_large_constants.patch](iree_global_opt_split_large_constants.patch) | Compiler feature (to upstream) | Add `iree-global-opt-split-large-constants`: split constants larger than the max allocation size (`--iree-stream-resource-max-allocation-size`) into contiguous shards and rewrite their gather and structured-op uses, so no single allocation exceeds the rpcmem `INT_MAX` limit (e.g. Gemma 4's 4.4 GiB per-layer embedding table). Drop once upstreamed. |
| [iree_hexagon_arch_recognition.patch](iree_hexagon_arch_recognition.patch) | Hexagon platform support | Add Hexagon architecture recognition to CMake and runtime target-platform definitions so DSP builds are accepted. |
| [iree_isa_genrule_location.patch](iree_isa_genrule_location.patch) | Embedded Bazel build fix | Use action-relative `location` paths for VM ISA generator inputs instead of runfiles-relative `rootpath` paths. |
| [iree_llvm_project_extension_owned.patch](iree_llvm_project_extension_owned.patch) | Shared dependency integration | Import configured LLVM/torch-mlir repositories from the extension instead of creating module-private copies; IREE and the plugin then link against the same instance. |
| [iree_restore_bufferize_allow_return_allocs_from_loops.patch](iree_restore_bufferize_allow_return_allocs_from_loops.patch) | Expose bufferization option | Restore the opt-in allowing newly allocated buffers to escape loops. Forward it through empty-tensor analysis and bufferization, as required by the Hexagon pipeline. |
| [iree_static_hexagon_runtime_driver.patch](iree_static_hexagon_runtime_driver.patch) | Runtime plugin integration | Load external HAL driver definitions from the embedding workspace and generate their static registration calls so IREE tools can select the Hexagon driver. |
| [iree_tracy_experimental_context_api_define.patch](iree_tracy_experimental_context_api_define.patch) | Tracing build integration | Enable IREE's experimental tracing-context API for the Bazel Tracy configuration used by the plugin. |
| [iree_tracy_gpu_zone_begin_external_colored.patch](iree_tracy_gpu_zone_begin_external_colored.patch) | Expose profiling API | Add colored external GPU-zone events for Hexagon profiling while preserving the existing uncolored entry point. |
