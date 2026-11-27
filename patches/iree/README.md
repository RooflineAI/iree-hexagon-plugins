# IREE patches

These patches integrate the compiler/runtime plugins and allow IREE to build
as a dependency of this repository. Most address embedding and API exposure.

| Patch | Category | Reason |
| --- | --- | --- |
| [iree_extensions_embedded_paths.patch](iree_extensions_embedded_paths.patch) | Shared dependency integration | Configure LLVM and torch-mlir in IREE's extension from injected sources, enable the Hexagon backend and point StableHLO at the shared top-level checkout. Works with the ownership patch below. |
| [iree_hal_constant_descriptor_flag.patch](iree_hal_constant_descriptor_flag.patch) | Hexagon codegen support | Add a `Constant` descriptor flag for bindings that every dispatch site binds to an immutable global and only reads, so DMA reads of them may bypass the DSP data cache. |
| [iree_hexagon_arch_recognition.patch](iree_hexagon_arch_recognition.patch) | Hexagon platform support | Add Hexagon architecture recognition to CMake and runtime target-platform definitions so DSP builds are accepted. |
| [iree_isa_genrule_location.patch](iree_isa_genrule_location.patch) | Embedded Bazel build fix | Use action-relative `location` paths for VM ISA generator inputs instead of runfiles-relative `rootpath` paths. |
| [iree_llvm_project_extension_owned.patch](iree_llvm_project_extension_owned.patch) | Shared dependency integration | Import configured LLVM/torch-mlir repositories from the extension instead of creating module-private copies; IREE and the plugin then link against the same instance. |
