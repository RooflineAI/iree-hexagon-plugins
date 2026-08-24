// This test exercises the native DSP runtime-linking path for Hexagon runtime
// symbols. Selected extern calls stay as native unresolved references in the
// generated kernel .so instead of being rewritten to HAL import thunks.

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: iree-opt \
// RUN:   --iree-hal-serialize-all-executables='dump-intermediates-path=%t dump-binaries-path=%t' \
// RUN:   %s
// RUN: FileCheck %s --check-prefix=LLVM < %t/module_test_embedded_elf_hexagon.ll
// RUN: readelf -sW %t/module_test_embedded_elf_hexagon.so | FileCheck %s --check-prefix=ELF

// LLVM: @hexagon_runtime_alloc_1d
// LLVM: @hexagon_runtime_dma_wait
// LLVM: @hexkl_matmul_f16f16_f32
// LLVM: @hexagon_runtime_malloc
// LLVM: @hexagon_runtime_free
// LLVM: @hexagon_runtime_memref_copy
// LLVM: @hexagon_runtime_profiler_zone_begin
// LLVM: @hexagon_runtime_profiler_zone_end
// LLVM-NOT: __import_ordinal_hexagon_runtime_alloc_1d
// LLVM-NOT: __import_ordinal_hexagon_runtime_dma_wait
// LLVM-NOT: __import_ordinal_hexkl_matmul_f16f16_f32
// LLVM-NOT: __import_ordinal_hexagon_runtime_malloc
// LLVM-NOT: __import_ordinal_hexagon_runtime_free
// LLVM-NOT: __import_ordinal_hexagon_runtime_memref_copy
// LLVM-NOT: __import_ordinal_hexagon_runtime_profiler_zone_begin
// LLVM-NOT: __import_ordinal_hexagon_runtime_profiler_zone_end

// ELF: UND hexagon_runtime_alloc_1d
// ELF: UND hexagon_runtime_malloc
// ELF: UND hexagon_runtime_memref_copy
// ELF: UND hexagon_runtime_free
// ELF: UND hexagon_runtime_dma_wait
// ELF: UND hexagon_runtime_profiler_zone_begin
// ELF: UND hexagon_runtime_profiler_zone_end
// ELF: UND hexkl_matmul_f16f16_f32

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", iree.encoding.resolver = #iree_hexagon.hexagon_encoding_resolver<>, link_embedded = false, max_stack_allocation_size = 16384 : i64, native_vector_size = 32 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
#device_target_hexagon = #hal.device.target<"hexagon", [#executable_target_embedded_elf_hexagon]> : !hal.device

hal.executable public @test {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) attributes {hexagon.native_runtime_linking} {
    builtin.module attributes {llvm.data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", llvm.target_triple = "hexagon-unknown-unknown-elf"} {
      llvm.func @hexagon_runtime_alloc_1d(i32, i64, i1) -> !llvm.ptr attributes {hexagon.native_runtime_link}
      llvm.func @hexagon_runtime_dma_wait(i32) attributes {hexagon.native_runtime_link}
      llvm.func @hexkl_matmul_f16f16_f32(i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {hexagon.native_runtime_link}
      llvm.func @hexagon_runtime_malloc(i64) -> !llvm.ptr attributes {hexagon.native_runtime_link}
      llvm.func @hexagon_runtime_free(!llvm.ptr) attributes {hexagon.native_runtime_link}
      llvm.func @hexagon_runtime_memref_copy(i64, !llvm.ptr, !llvm.ptr) attributes {hexagon.native_runtime_link}
      llvm.func @hexagon_runtime_profiler_zone_begin(i32, !llvm.ptr) -> !llvm.ptr attributes {hexagon.native_runtime_link}
      llvm.func @hexagon_runtime_profiler_zone_end(!llvm.ptr) attributes {hexagon.native_runtime_link}

      llvm.func @export(%arg0: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg2: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}) -> i32 {
        %c0_i32 = llvm.mlir.constant(0 : i32) : i32
        %c4_i64 = llvm.mlir.constant(4 : i64) : i64
        %false = llvm.mlir.zero : i1
        %null = llvm.mlir.zero : !llvm.ptr
        %buf = llvm.call @hexagon_runtime_alloc_1d(%c0_i32, %c4_i64, %false) : (i32, i64, i1) -> !llvm.ptr
        %tmp = llvm.call @hexagon_runtime_malloc(%c4_i64) : (i64) -> !llvm.ptr
        llvm.call @hexagon_runtime_memref_copy(%c4_i64, %null, %null) : (i64, !llvm.ptr, !llvm.ptr) -> ()
        llvm.call @hexagon_runtime_free(%tmp) : (!llvm.ptr) -> ()
        llvm.call @hexagon_runtime_dma_wait(%c0_i32) : (i32) -> ()
        %record = llvm.call @hexagon_runtime_profiler_zone_begin(%c0_i32, %null) : (i32, !llvm.ptr) -> (!llvm.ptr)
        llvm.call @hexagon_runtime_profiler_zone_end(%record) : (!llvm.ptr) -> ()
        %result = llvm.call @hexkl_matmul_f16f16_f32(%c4_i64, %c4_i64, %c4_i64, %buf, %null, %null) : (i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
        llvm.return %result : i32
      }
    }
  }
}
