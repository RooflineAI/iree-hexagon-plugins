// This test is making sure that query function needed by the HAL to call on functions based on their 
// ordinal assigned by the linking pipeline is correctly added and exposed in the .so file.

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: iree-opt --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN \
// RUN:   --iree-hal-target-device=hexagon \
// RUN:   --iree-hal-dump-executable-intermediates-to=%t \
// RUN:   --iree-hal-dump-executable-binaries-to=%t \
// RUN:   --pass-pipeline='builtin.module(iree-hal-transformation-pipeline)' \
// RUN:   %s
// RUN:   readelf -sW %t/module_test_embedded_elf_hexagon.so | FileCheck %s

// CHECK: FUNC    GLOBAL DEFAULT    {{.}} iree_hal_executable_library_query

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#device_target_hexagon = #hal.device.target<"hexagon", [#executable_target_embedded_elf_hexagon]> : !hal.device
hal.executable public @test {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    builtin.module attributes {llvm.data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", llvm.target_triple = "hexagon-unknown-unknown-elf"} {
      llvm.func @export(%arg0: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg2: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}) -> i32 {
        %0 = llvm.mlir.constant(0 : i32) : i32
        llvm.return %0 : i32
      }
    }
  }
}
