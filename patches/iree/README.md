# IREE patches

These patches integrate the compiler/runtime plugins and allow IREE to build
as a dependency of this repository. Most address embedding and API exposure.

| Patch | Category | Reason |
| --- | --- | --- |
| [iree_extensions_embedded_paths.patch](iree_extensions_embedded_paths.patch) | Shared dependency integration | Configure LLVM and torch-mlir in IREE's extension from injected sources, enable the Hexagon backend and point StableHLO at the shared top-level checkout. Works with the ownership patch below. |
| [iree_hexagon_arch_recognition.patch](iree_hexagon_arch_recognition.patch) | Hexagon platform support | Add Hexagon architecture recognition to CMake and runtime target-platform definitions so DSP builds are accepted. |
| [iree_isa_genrule_location.patch](iree_isa_genrule_location.patch) | Embedded Bazel build fix | Use action-relative `location` paths for VM ISA generator inputs instead of runfiles-relative `rootpath` paths. |
| [iree_llvm_project_extension_owned.patch](iree_llvm_project_extension_owned.patch) | Shared dependency integration | Import configured LLVM/torch-mlir repositories from the extension instead of creating module-private copies; IREE and the plugin then link against the same instance. |
| [iree_tracy_experimental_context_api_define.patch](iree_tracy_experimental_context_api_define.patch) | Tracing build integration | Enable IREE's experimental tracing-context API for the Bazel Tracy configuration used by the plugin. |
| [iree_tracy_gpu_zone_begin_external_colored.patch](iree_tracy_gpu_zone_begin_external_colored.patch) | Expose profiling API | Add colored external GPU-zone events for Hexagon profiling while preserving the existing uncolored entry point. |
