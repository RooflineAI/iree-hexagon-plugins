// XFAIL: *
// TODO: Replace this expected failure with a passing diagnostic test once the
// compiler computes transitive stack use from HAL exports, or rejects calls
// between defined functions. Each frame below is under the configured 4096-byte
// per-frame limit, but both are live while @helper runs and their combined size
// exceeds that limit.

// RUN: not iree-opt --iree-load-plugin=hexagon=$ROOF_HEXAGON_COMPILER_PLUGIN \
// RUN:   --iree-hal-target-backends=hexagon \
// RUN:   --iree-hexagon-fail-on-stack-frames-larger-than=4096 \
// RUN:   --pass-pipeline='builtin.module(iree-hal-transformation-pipeline)' \
// RUN:   %s -o /dev/null

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 8192 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

hal.executable public @test {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    builtin.module attributes {llvm.data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", llvm.target_triple = "hexagon-unknown-unknown-elf"} {
      llvm.func @helper() attributes {no_inline} {
        %count = llvm.mlir.constant(3000 : i32) : i32
        %frame = llvm.alloca %count x i8 {alignment = 128 : i64} : (i32) -> !llvm.ptr
        llvm.inline_asm has_side_effects "", "r,~{memory}" %frame : (!llvm.ptr) -> ()
        llvm.return
      }

      llvm.func @export(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 {
        %count = llvm.mlir.constant(3000 : i32) : i32
        %frame = llvm.alloca %count x i8 {alignment = 128 : i64} : (i32) -> !llvm.ptr
        llvm.call @helper() : () -> ()
        llvm.inline_asm has_side_effects "", "r,~{memory}" %frame : (!llvm.ptr) -> ()
        %zero = llvm.mlir.constant(0 : i32) : i32
        llvm.return %zero : i32
      }
    }
  }
}
