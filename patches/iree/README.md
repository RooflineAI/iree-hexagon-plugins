# IREE patches

These patches integrate the compiler/runtime plugins and allow IREE to build
as a dependency of this repository. Most address embedding and API exposure,
rather than incompatibility with hexagon-mlir's older LLVM APIs.

| Patch | Category | Reason |
| --- | --- | --- |
| [iree_android_platform_os_detection.patch](iree_android_platform_os_detection.patch) | Android build integration | Recognize our Android platform constraint alongside IREE's define-based setting, selecting the proper pthread/link flags and POSIX signal dependency. |
| [iree_bitcode_library_embedded_include_root.patch](iree_bitcode_library_embedded_include_root.patch) | Embedded Bazel build fix | Resolve LLVM builtin headers and IREE include roots using canonical repository names instead of assuming IREE is the main workspace. |
| [iree_extensions_embedded_paths.patch](iree_extensions_embedded_paths.patch) | Shared dependency integration | Configure LLVM and torch-mlir in IREE's extension from injected sources, enable the Hexagon backend and point StableHLO at the shared top-level checkout. Works with the ownership patch below. |
| [iree_flatcc_embedded_include_root.patch](iree_flatcc_embedded_include_root.patch) | Embedded Bazel build fix | Give flatcc the correct IREE runtime include directory when IREE lives under an external repository name. |
| [iree_hexagon_arch_recognition.patch](iree_hexagon_arch_recognition.patch) | Hexagon platform support | Add Hexagon architecture recognition to CMake and runtime target-platform definitions so DSP builds are accepted. |
| [iree_isa_genrule_location.patch](iree_isa_genrule_location.patch) | Embedded Bazel build fix | Use action-relative `location` paths for VM ISA generator inputs instead of runfiles-relative `rootpath` paths. |
| [iree_llvm_project_extension_owned.patch](iree_llvm_project_extension_owned.patch) | Shared dependency integration | Import configured LLVM/torch-mlir repositories from the extension instead of creating module-private copies; IREE and the plugin then link against the same instance. |
| [iree_restore_bufferize_allow_return_allocs_from_loops.patch](iree_restore_bufferize_allow_return_allocs_from_loops.patch) | Expose bufferization option | Restore the opt-in allowing newly allocated buffers to escape loops. Forward it through empty-tensor analysis and bufferization, as required by the Hexagon pipeline. |
| [iree_static_hexagon_runtime_driver.patch](iree_static_hexagon_runtime_driver.patch) | Runtime plugin integration | Load external HAL driver definitions from the embedding workspace and generate their static registration calls so IREE tools can select the Hexagon driver. |
| [iree_tracy_experimental_context_api_define.patch](iree_tracy_experimental_context_api_define.patch) | Tracing build integration | Enable IREE's experimental tracing-context API for the Bazel Tracy configuration used by the plugin. |
| [iree_tracy_gpu_zone_begin_external_colored.patch](iree_tracy_gpu_zone_begin_external_colored.patch) | Expose profiling API | Add colored external GPU-zone events for Hexagon profiling while preserving the existing uncolored entry point. |

The upgrade rebases the extension, flatcc and bufferization patches onto the
new IREE pin. The Generic-platform async CMake workaround was removed because
the new IREE revision handles that platform upstream.
