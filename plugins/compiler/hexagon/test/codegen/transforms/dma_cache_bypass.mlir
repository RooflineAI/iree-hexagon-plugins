// RUN: iree-opt --iree-stream-resource-min-offset-alignment=128 --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hexagon-dma-cache-bypass)))" --split-input-file --verify-diagnostics %s | FileCheck %s

// Bindings accessed only by DMA bypass the data cache: the DMA transfers are
// tagged on their DDR side, and on their VTCM side as VTCM is not cached.

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon">
#layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable private @executable {
  hal.executable.variant public @variant target(#target) {
    hal.executable.export public @dma_only ordinal(0) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @dma_only() {
        %c0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %in = hal.interface.binding.subspan layout(#layout) binding(0) offset(%c0) : memref<512xf32>
        %out = hal.interface.binding.subspan layout(#layout) binding(1) offset(%c0) : memref<512xf32>
        %in_view = memref.subview %in[256] [256] [1] : memref<512xf32> to memref<256xf32, strided<[1], offset: 256>>
        %vtcm = memref.alloc() : memref<256xf32, 1>
        %tag = memref.alloca() : memref<1xi32>
        memref.dma_start %in_view[%c0], %vtcm[%c0], %c256, %tag[%c0] : memref<256xf32, strided<[1], offset: 256>>, memref<256xf32, 1>, memref<1xi32>
        memref.dma_wait %tag[%c0], %c256 : memref<1xi32>
        memref.dma_start %vtcm[%c0], %out[%c0], %c256, %tag[%c0] : memref<256xf32, 1>, memref<512xf32>, memref<1xi32>
        memref.dma_wait %tag[%c0], %c256 : memref<1xi32>
        memref.dealloc %vtcm : memref<256xf32, 1>
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable.export public @dma_only
// CHECK:       } attributes {iree_hexagon.dma_bypass_bindings = 3 : i64}
// CHECK-LABEL: func.func @dma_only
// CHECK:       memref.dma_start {{.+}} {bypass_dst, bypass_src}
// CHECK:       memref.dma_start {{.+}} {bypass_dst, bypass_src}

// -----

// A binding also accessed through the cache does not bypass it. A written
// binding does not bypass the cache either while another binding is written
// through it (they may share a cache line), but a read-only one does.

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon">
#layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable private @executable {
  hal.executable.variant public @variant target(#target) {
    hal.executable.export public @mixed ordinal(0) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @mixed() {
        %c0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %cst = arith.constant 0.0 : f32
        %loaded = hal.interface.binding.subspan layout(#layout) binding(0) offset(%c0) : memref<256xf32>
        %dma_in = hal.interface.binding.subspan layout(#layout) binding(1) offset(%c0) : memref<256xf32>
        %dma_out = hal.interface.binding.subspan layout(#layout) binding(2) offset(%c0) : memref<256xf32>
        %stored = hal.interface.binding.subspan layout(#layout) binding(3) offset(%c0) : memref<256xf32>
        %vtcm = memref.alloc() : memref<256xf32, 1>
        %tag = memref.alloca() : memref<1xi32>
        memref.dma_start %loaded[%c0], %vtcm[%c0], %c256, %tag[%c0] : memref<256xf32>, memref<256xf32, 1>, memref<1xi32>
        memref.dma_wait %tag[%c0], %c256 : memref<1xi32>
        %x = memref.load %loaded[%c0] : memref<256xf32>
        memref.dma_start %dma_in[%c0], %vtcm[%c0], %c256, %tag[%c0] : memref<256xf32>, memref<256xf32, 1>, memref<1xi32>
        memref.dma_wait %tag[%c0], %c256 : memref<1xi32>
        memref.dma_start %vtcm[%c0], %dma_out[%c0], %c256, %tag[%c0] : memref<256xf32, 1>, memref<256xf32>, memref<1xi32>
        memref.dma_wait %tag[%c0], %c256 : memref<1xi32>
        memref.store %x, %stored[%c0] : memref<256xf32>
        memref.dealloc %vtcm : memref<256xf32, 1>
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable.export public @mixed
// CHECK:       } attributes {iree_hexagon.dma_bypass_bindings = 2 : i64}
// CHECK-LABEL: func.func @mixed
// CHECK:       memref.dma_start {{.+}} {bypass_dst}
// CHECK:       memref.dma_start {{.+}} {bypass_dst, bypass_src}
// CHECK:       memref.dma_start {{.+}} {bypass_src}
