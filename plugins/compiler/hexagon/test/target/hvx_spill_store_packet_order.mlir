// A strided copy (a strided slice assignment such as Qwen3-VL's interleaved
// M-RoPE, `x[..., 1:60:3] = y[..., 1:60:3]`) gathers 20 floats into a
// vector<20xf32> with a dynamic insertelement per element. The Hexagon
// backend lowers such an insert through a stack slot: spill the vector with
// an HVX `vmem` store, store the element with a scalar `memw`, reload.
//
// An HVX store can only go in slot 0, and the slot-0 store of a packet takes
// effect last. The packetizer puts the spill and the element store in one
// packet, so the spill overwrites the element and the copy writes zeros.
// The DynamicInsertElementToSelect pass avoids this by rewriting the insert
// into a compare and a select, which the backend lowers in registers.
//
// This checks the assembly: the gather is a vmux, and no packet holds both an
// HVX store to the stack and a scalar store at an immediate offset (the
// element store).

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: iree-opt \
// RUN:   --iree-hal-serialize-all-executables='dump-intermediates-path=%t dump-binaries-path=%t' \
// RUN:   %s -o %t/serialized.mlir
// Print each packet of the assembly on one line.
// RUN: awk '/^[[:space:]]*\{/ { p = ""; next } \
// RUN:      /^[[:space:]]*\}/ { print "PACKET:" p; next } \
// RUN:      { p = p " " $0 }' %t/*.s > %t/packets.txt
// RUN: FileCheck %s --check-prefix=SELECT < %t/packets.txt
// RUN: FileCheck %s --check-prefix=NOPAIR < %t/packets.txt

// The insert is now a select, lowered to a vmux.
// SELECT: PACKET:{{.*}}vmux(q{{[0-3]}},v{{[0-9]+}},v{{[0-9]+}})

// NOPAIR-NOT: vmem(r29+#{{[0-9]+}}) = v{{[0-9]+}}{{.*}}memw(r{{[0-9]+}}+#{{[0-9]+}}) = r{{[0-9]+}}
// NOPAIR-NOT: memw(r{{[0-9]+}}+#{{[0-9]+}}) = r{{[0-9]+}}{{.*}}vmem(r29+#{{[0-9]+}}) = v{{[0-9]+}}

#executable_target_embedded_elf_hexagon = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", hexagon.version = "79", max_stack_allocation_size = 16384 : i64, native_vector_size = 128 : i64, target_triple = "hexagon-unknown-unknown-elf"}>
hal.executable public @strided_copy {
  hal.executable.variant public @embedded_elf_hexagon target(#executable_target_embedded_elf_hexagon) {
    builtin.module attributes {llvm.data_layout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048", llvm.target_triple = "hexagon-unknown-unknown-elf"} {
      // dst[3 * i + 1] = src[3 * i + 1] for i in [0, 20), through a vector,
      // as the strided-copy dispatch is lowered to LLVM.
      llvm.func @strided_copy(%src: !llvm.ptr {llvm.noalias}, %dst: !llvm.ptr {llvm.noalias}) {
        %c0 = llvm.mlir.constant(0 : i32) : i32
        %c1 = llvm.mlir.constant(1 : i32) : i32
        %c3 = llvm.mlir.constant(3 : i32) : i32
        %c20 = llvm.mlir.constant(20 : i32) : i32
        %poison = llvm.mlir.poison : vector<20xf32>
        llvm.br ^gather(%c0, %poison : i32, vector<20xf32>)
      ^gather(%i: i32, %acc: vector<20xf32>):
        %gcond = llvm.icmp "slt" %i, %c20 : i32
        llvm.cond_br %gcond, ^gather_body, ^scatter(%c0 : i32)
      ^gather_body:
        %i3 = llvm.mul %i, %c3 overflow<nsw> : i32
        %goff = llvm.add %i3, %c1 overflow<nsw> : i32
        %gptr = llvm.getelementptr inbounds %src[%goff] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %x = llvm.load %gptr : !llvm.ptr -> f32
        %acc2 = llvm.insertelement %x, %acc[%i : i32] : vector<20xf32>
        %inext = llvm.add %i, %c1 overflow<nsw> : i32
        llvm.br ^gather(%inext, %acc2 : i32, vector<20xf32>)
      ^scatter(%j: i32):
        %scond = llvm.icmp "slt" %j, %c20 : i32
        llvm.cond_br %scond, ^scatter_body, ^exit
      ^scatter_body:
        %y = llvm.extractelement %acc[%j : i32] : vector<20xf32>
        %j3 = llvm.mul %j, %c3 overflow<nsw> : i32
        %soff = llvm.add %j3, %c1 overflow<nsw> : i32
        %sptr = llvm.getelementptr inbounds %dst[%soff] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        llvm.store %y, %sptr : f32, !llvm.ptr
        %jnext = llvm.add %j, %c1 overflow<nsw> : i32
        llvm.br ^scatter(%jnext : i32)
      ^exit:
        llvm.return
      }
    }
  }
}
