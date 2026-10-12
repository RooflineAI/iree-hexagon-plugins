// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hexagon-propagate-dispatch-resources)))" --split-input-file %s | FileCheck %s

// The dispatch resources in each entry point's translation info are copied
// unchanged to its export. An entry point without them leaves its export
// unannotated.

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon">
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
#resources = #iree_hexagon.dispatch_resources<hvx = true, hmx = true, vtcm_bytes = 28672>
#planned = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<HmxMatmulExpert>, {iree_hexagon.dispatch_resources = #resources}>
#unplanned = #iree_codegen.translation_info<pipeline = #iree_hexagon.pipeline<Default>>

hal.executable private @executable {
  hal.executable.variant public @variant target(#target) {
    hal.executable.export public @planned ordinal(0) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    hal.executable.export public @unplanned ordinal(1) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    hal.executable.export public @no_translation_info ordinal(2) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @planned() attributes {translation_info = #planned} {
        return
      }
      func.func @unplanned() attributes {translation_info = #unplanned} {
        return
      }
      func.func @no_translation_info() {
        return
      }
    }
  }
}

// CHECK:      hal.executable.export public @planned
// CHECK:      } attributes {iree_hexagon.dispatch_resources = #iree_hexagon.dispatch_resources<hvx = true, hmx = true, vtcm_bytes = 28672>}
// CHECK:      hal.executable.export public @unplanned
// CHECK-NOT:  iree_hexagon.dispatch_resources
// CHECK:      hal.executable.export public @no_translation_info
// CHECK-NOT:  iree_hexagon.dispatch_resources
// CHECK:      builtin.module
