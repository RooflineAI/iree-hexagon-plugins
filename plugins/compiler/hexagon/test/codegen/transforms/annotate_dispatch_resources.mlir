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
    hal.executable.export public @alloc_in_loop ordinal(3) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    hal.executable.export public @hmx ordinal(4) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    hal.executable.export public @calls_helper ordinal(5) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
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
        // 1000 bytes.
        %a = memref.alloc() : memref<250xf32>
        // 1000 bytes plus the 64-byte alignment padding.
        %b = memref.alloc() alignment = 64 : memref<250xf32>
        // 16 bytes, raised to the 128-byte pool minimum.
        %c = memref.alloc() : memref<4xf32>
        // 4000 bytes, rounded to 2 KiB for a 2 KiB-aligned VTCM buffer.
        %d = hexagonmem.alloc() <alignment = 2048> : memref<1000xf32, 1>
        // 1000 bytes, rounded to 128 bytes.
        %e = hexagonmem.alloc() : memref<1000xi8, 1>
        // Stack, not VTCM.
        %f = memref.alloca() : memref<1000xi8>
        memref.dealloc %a : memref<250xf32>
        memref.dealloc %b : memref<250xf32>
        memref.dealloc %c : memref<4xf32>
        hexagonmem.dealloc %d : memref<1000xf32, 1>
        hexagonmem.dealloc %e : memref<1000xi8, 1>
        return
      }
      func.func @alloc_in_loop() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        scf.for %i = %c0 to %c4 step %c1 {
          %a = memref.alloc() : memref<250xf32>
          memref.dealloc %a : memref<250xf32>
        }
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
// CHECK:       } attributes {iree_hexagon.dispatch_resources = #iree_hexagon.dispatch_resources<hvx = false, hmx = false, vtcm_bytes = 7312>}
// CHECK-LABEL: hal.executable.export public @alloc_in_loop
// CHECK:       } attributes {iree_hexagon.dispatch_resources = #iree_hexagon.dispatch_resources<hvx = false, hmx = false, vtcm_bytes = 1000>}
// CHECK-LABEL: hal.executable.export public @hmx
// CHECK:       } attributes {iree_hexagon.dispatch_resources = #iree_hexagon.dispatch_resources<hvx = true, hmx = true, vtcm_bytes = 0>}
// CHECK-LABEL: hal.executable.export public @calls_helper
// CHECK:       } attributes {iree_hexagon.dispatch_resources = #iree_hexagon.dispatch_resources<hvx = false, hmx = false, vtcm_bytes = 1000>}

// -----

// A dynamically sized allocation cannot be bounded: the export stays
// unannotated, so the DSP assumes the dispatch needs every resource.

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
        // expected-warning @+1 {{cannot bound the VTCM this allocation takes; the dispatch is left without resource annotation}}
        %a = memref.alloc(%n) : memref<?xf32>
        memref.dealloc %a : memref<?xf32>
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable.export public @dynamic
// CHECK-NOT:   iree_hexagon.dispatch_resources
// CHECK:       builtin.module
