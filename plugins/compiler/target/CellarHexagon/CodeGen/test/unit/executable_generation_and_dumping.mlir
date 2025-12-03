// This test makes sure that:
//  - debug files are properly dumped when passing the --iree-hal-dump-executable-intermediates-to and --iree-hal-dump-executable-binaries-to flags
//  - that the full lowering pipeline runs and outputs a vmfb. This includes serialization and a call to llvm right now
// This test does not check for correctness in any way, only that everything is properly connected

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: iree-opt --iree-plugin=hal_target_hexagon \
// RUN:   --iree-hal-target-backends=hexagon \
// RUN:   --iree-hal-dump-executable-intermediates-to=%t \
// RUN:   --iree-hal-dump-executable-binaries-to=%t \
// RUN:   --pass-pipeline='builtin.module(iree-hal-transformation-pipeline)' \
// RUN:   %s -o %t/matmul.vmfb
// RUN: test -f %t/matmul.vmfb
// RUN: find %t -name "*.ll" | grep .
// RUN: find %t -name "*.s" | grep .
// RUN: find %t -name "*.o" | grep .
// RUN: find %t -name "*.so" | grep .

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 32768 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
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
