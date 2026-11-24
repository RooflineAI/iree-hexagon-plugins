// Checks the DynamicInsertElementToSelect workaround pass, which runs at the
// end of the LLVM middle-end pipeline: an `insertelement` with a variable
// index is rewritten into a compare against the lane numbers and a `select`,
// but only for the vector types that the Hexagon backend inserts into through
// a stack slot (see DynamicInsertElementToSelect.cpp). The optimized LLVM IR
// is dumped as `.opt.ll`.

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: iree-opt \
// RUN:   --iree-hal-serialize-all-executables='dump-intermediates-path=%t' \
// RUN:   %s -o %t/serialized.mlir
// RUN: FileCheck %s < %t/module_test_embedded_elf_hexagon.opt.ll

// f32, widened to an HVX vector: rewritten, lanes compared as i32.
// CHECK-LABEL: define {{.*}} @insert_v20f32(
// CHECK-SAME:    float %[[X:[0-9]+]], i32 %[[I:[0-9]+]])
// CHECK-NOT:     insertelement <20 x float> %{{[0-9]+}}, float
// CHECK:         %[[IDX:.+]] = insertelement <20 x i32> poison, i32 %[[I]], i64 0
// CHECK:         %[[IDXS:.+]] = shufflevector <20 x i32> %[[IDX]], <20 x i32> poison, <20 x i32> zeroinitializer
// CHECK:         %[[MASK:.+]] = icmp eq <20 x i32> %[[IDXS]], <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
// CHECK:         %[[XI:.+]] = insertelement <20 x float> poison, float %[[X]], i64 0
// CHECK:         %[[XS:.+]] = shufflevector <20 x float> %[[XI]], <20 x float> poison, <20 x i32> zeroinitializer
// CHECK:         select <20 x i1> %[[MASK]], <20 x float> %[[XS]], <20 x float> %{{[0-9]+}}
// CHECK-NOT:     insertelement <20 x float> %{{[0-9]+}}, float

// f32, exactly one HVX vector: rewritten.
// CHECK-LABEL: define {{.*}} @insert_v32f32(
// CHECK:         icmp eq <32 x i32>
// CHECK:         select <32 x i1>

// i8, a pair of HVX vectors: rewritten, lanes compared as i8.
// CHECK-LABEL: define {{.*}} @insert_v256i8(
// CHECK:         icmp eq <256 x i8>
// CHECK:         select <256 x i1>

// i32, one HVX vector: inserted in registers, left alone.
// CHECK-LABEL: define {{.*}} @insert_v32i32(
// CHECK:         insertelement <32 x i32> %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}}
// CHECK-NOT:     select

// f32 below half an HVX vector: no HVX store, left alone.
// CHECK-LABEL: define {{.*}} @insert_v8f32(
// CHECK:         insertelement <8 x float> %{{[0-9]+}}, float %{{[0-9]+}}, i32 %{{[0-9]+}}
// CHECK-NOT:     select

// A constant index, and an extract with a variable index: left alone.
// CHECK-LABEL: define {{.*}} @constant_insert_and_extract_v32f32(
// CHECK:         insertelement <32 x float> %{{[0-9]+}}, float %{{[0-9]+}}, i64 5
// CHECK:         extractelement <32 x float> %{{[0-9]+}}, i32 %{{[0-9]+}}
// CHECK-NOT:     select
// CHECK-LABEL: define {{.*}} @hexagon_executable_library_query(

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", max_stack_allocation_size = 16384 : i64, native_vector_size = 128 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
hal.executable public @test {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    builtin.module attributes {llvm.data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", llvm.target_triple = "hexagon-unknown-unknown-elf"} {
      llvm.func @insert_v20f32(%p: !llvm.ptr, %x: f32, %i: i32) {
        %v = llvm.load %p : !llvm.ptr -> vector<20xf32>
        %r = llvm.insertelement %x, %v[%i : i32] : vector<20xf32>
        llvm.store %r, %p : vector<20xf32>, !llvm.ptr
        llvm.return
      }
      llvm.func @insert_v32f32(%p: !llvm.ptr, %x: f32, %i: i32) {
        %v = llvm.load %p : !llvm.ptr -> vector<32xf32>
        %r = llvm.insertelement %x, %v[%i : i32] : vector<32xf32>
        llvm.store %r, %p : vector<32xf32>, !llvm.ptr
        llvm.return
      }
      llvm.func @insert_v256i8(%p: !llvm.ptr, %x: i8, %i: i32) {
        %v = llvm.load %p : !llvm.ptr -> vector<256xi8>
        %r = llvm.insertelement %x, %v[%i : i32] : vector<256xi8>
        llvm.store %r, %p : vector<256xi8>, !llvm.ptr
        llvm.return
      }
      llvm.func @insert_v32i32(%p: !llvm.ptr, %x: i32, %i: i32) {
        %v = llvm.load %p : !llvm.ptr -> vector<32xi32>
        %r = llvm.insertelement %x, %v[%i : i32] : vector<32xi32>
        llvm.store %r, %p : vector<32xi32>, !llvm.ptr
        llvm.return
      }
      llvm.func @insert_v8f32(%p: !llvm.ptr, %x: f32, %i: i32) {
        %v = llvm.load %p : !llvm.ptr -> vector<8xf32>
        %r = llvm.insertelement %x, %v[%i : i32] : vector<8xf32>
        llvm.store %r, %p : vector<8xf32>, !llvm.ptr
        llvm.return
      }
      llvm.func @constant_insert_and_extract_v32f32(%p: !llvm.ptr, %q: !llvm.ptr, %x: f32, %i: i32) {
        %c5 = llvm.mlir.constant(5 : i64) : i64
        %v = llvm.load %p : !llvm.ptr -> vector<32xf32>
        %r = llvm.insertelement %x, %v[%c5 : i64] : vector<32xf32>
        llvm.store %r, %p : vector<32xf32>, !llvm.ptr
        %e = llvm.extractelement %v[%i : i32] : vector<32xf32>
        llvm.store %e, %q : f32, !llvm.ptr
        llvm.return
      }
    }
  }
}
