// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hexagon-annotate-dispatch-resources)))" --split-input-file --verify-diagnostics %s | FileCheck %s

// Each export is annotated with the resources its entry point uses.
// `vtcm_bytes` sums every heap allocation at the size the runtime allocates
// for it; stack allocations do not count.

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon">
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

hal.executable private @executable {
  hal.executable.variant public @variant target(#target) {
    hal.executable.export public @scalar ordinal(0) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    hal.executable.export public @vector ordinal(1) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    hal.executable.export public @allocs ordinal(2) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    hal.executable.export public @hmx ordinal(3) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    hal.executable.export public @calls_helper ordinal(4) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @scalar() {
        return
      }
      func.func @vector() {
        %cst = arith.constant dense<0.0> : vector<32xf32>
        return
      }
      func.func @allocs() {
        // 1000 bytes, rounded to 1024.
        %a = memref.alloc() : memref<250xf32>
        // 1000 bytes, rounded to 1024; 128-byte granules are 64-byte aligned.
        %b = memref.alloc() alignment = 64 : memref<250xf32>
        // 16 bytes, raised to the 128-byte minimum.
        %c = memref.alloc() : memref<4xf32>
        // 4000 bytes, rounded to 2 KiB for a 2 KiB-aligned VTCM buffer.
        %d = hexagonmem.alloc() <alignment = 2048> : memref<1000xf32, 1>
        // 1000 bytes, rounded to 1024.
        %e = hexagonmem.alloc() : memref<1000xi8, 1>
        // 128 bytes plus up to 128 bytes of padding to the 256-byte alignment.
        %g = memref.alloc() alignment = 256 : memref<32xf32>
        // Stack, not VTCM.
        %f = memref.alloca() : memref<1000xi8>
        memref.dealloc %a : memref<250xf32>
        memref.dealloc %b : memref<250xf32>
        memref.dealloc %c : memref<4xf32>
        hexagonmem.dealloc %d : memref<1000xf32, 1>
        hexagonmem.dealloc %e : memref<1000xi8, 1>
        memref.dealloc %g : memref<32xf32>
        return
      }
      func.func @hmx() {
        %acc = iree_hexagon.hmx.acc.zero : !iree_hexagon.hmx.acc<32x32xf32>
        return
      }
      func.func private @helper() {
        %a = memref.alloc() : memref<250xf32>
        memref.dealloc %a : memref<250xf32>
        return
      }
      func.func @calls_helper() {
        call @helper() : () -> ()
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable.export public @scalar
// CHECK:       } attributes {iree_hexagon.dispatch_resources = #iree_hexagon.dispatch_resources<hvx = false, hmx = false, vtcm_bytes = 0>}
// CHECK-LABEL: hal.executable.export public @vector
// CHECK:       } attributes {iree_hexagon.dispatch_resources = #iree_hexagon.dispatch_resources<hvx = true, hmx = false, vtcm_bytes = 0>}
// CHECK-LABEL: hal.executable.export public @allocs
// CHECK:       } attributes {iree_hexagon.dispatch_resources = #iree_hexagon.dispatch_resources<hvx = false, hmx = false, vtcm_bytes = 7552>}
// CHECK-LABEL: hal.executable.export public @hmx
// CHECK:       } attributes {iree_hexagon.dispatch_resources = #iree_hexagon.dispatch_resources<hvx = true, hmx = true, vtcm_bytes = 0>}
// CHECK-LABEL: hal.executable.export public @calls_helper
// CHECK:       } attributes {iree_hexagon.dispatch_resources = #iree_hexagon.dispatch_resources<hvx = false, hmx = false, vtcm_bytes = 1024>}

// -----

// A dynamically sized allocation cannot be bounded: the export declares all of
// VTCM (INT64_MAX).

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon">
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

hal.executable private @executable {
  hal.executable.variant public @variant target(#target) {
    hal.executable.export public @dynamic ordinal(0) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @dynamic() {
        %n = arith.constant 4 : index
        // expected-warning @+1 {{cannot bound the VTCM this allocation takes; the dispatch declares all of VTCM}}
        %a = memref.alloc(%n) : memref<?xf32>
        memref.dealloc %a : memref<?xf32>
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable.export public @dynamic
// CHECK:       } attributes {iree_hexagon.dispatch_resources = #iree_hexagon.dispatch_resources<hvx = false, hmx = false, vtcm_bytes = 9223372036854775807>}

// -----

// Allocations are counted once, which bounds the peak only outside loops. One
// left inside a loop op should have been hoisted.

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon">
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

hal.executable private @executable {
  hal.executable.variant public @variant target(#target) {
    hal.executable.export public @alloc_in_scf_loop ordinal(0) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @alloc_in_scf_loop() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        scf.for %i = %c0 to %c4 step %c1 {
          // expected-error @+1 {{VTCM allocation inside a loop; it should have been hoisted out of the loop}}
          %a = memref.alloc() : memref<250xf32>
          memref.dealloc %a : memref<250xf32>
        }
        return
      }
    }
  }
}

// -----

// The same for a loop after SCF-to-CF conversion, a control-flow cycle.

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon">
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

hal.executable private @executable {
  hal.executable.variant public @variant target(#target) {
    hal.executable.export public @alloc_in_cfg_loop ordinal(0) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @alloc_in_cfg_loop() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        cf.br ^header(%c0 : index)
      ^header(%i: index):
        %cond = arith.cmpi slt, %i, %c4 : index
        cf.cond_br %cond, ^body, ^exit
      ^body:
        // expected-error @+1 {{VTCM allocation inside a loop; it should have been hoisted out of the loop}}
        %a = memref.alloc() : memref<250xf32>
        memref.dealloc %a : memref<250xf32>
        %next = arith.addi %i, %c1 : index
        cf.br ^header(%next : index)
      ^exit:
        return
      }
    }
  }
}
