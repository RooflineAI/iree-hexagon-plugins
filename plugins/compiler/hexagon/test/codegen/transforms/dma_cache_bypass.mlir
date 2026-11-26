// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hexagon-dma-cache-bypass)))" --split-input-file %s | FileCheck %s

// DMA transfers bypass the data cache when they read a Constant binding, also
// through a view, and on their VTCM side, as VTCM is not cached. Reading a
// binding that is only ReadOnly, or writing to DDR, goes through the cache.

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon">
#layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Constant">,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable private @executable {
  hal.executable.variant public @variant target(#target) {
    hal.executable.export public @bypass ordinal(0) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @bypass() {
        %c0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %constant = hal.interface.binding.subspan layout(#layout) binding(0) offset(%c0) : memref<512xf32>
        %in = hal.interface.binding.subspan layout(#layout) binding(1) offset(%c0) : memref<256xf32>
        %out = hal.interface.binding.subspan layout(#layout) binding(2) offset(%c0) : memref<256xf32>
        %constant_view = memref.subview %constant[256] [256] [1] : memref<512xf32> to memref<256xf32, strided<[1], offset: 256>>
        %vtcm = memref.alloc() : memref<256xf32, 1>
        %tag = memref.alloca() : memref<1xi32>
        memref.dma_start %constant_view[%c0], %vtcm[%c0], %c256, %tag[%c0] : memref<256xf32, strided<[1], offset: 256>>, memref<256xf32, 1>, memref<1xi32>
        memref.dma_wait %tag[%c0], %c256 : memref<1xi32>
        memref.dma_start %in[%c0], %vtcm[%c0], %c256, %tag[%c0] : memref<256xf32>, memref<256xf32, 1>, memref<1xi32>
        memref.dma_wait %tag[%c0], %c256 : memref<1xi32>
        memref.dma_start %vtcm[%c0], %out[%c0], %c256, %tag[%c0] : memref<256xf32, 1>, memref<256xf32>, memref<1xi32>
        memref.dma_wait %tag[%c0], %c256 : memref<1xi32>
        memref.dma_start %constant[%c0], %out[%c0], %c256, %tag[%c0] : memref<512xf32>, memref<256xf32>, memref<1xi32>
        memref.dma_wait %tag[%c0], %c256 : memref<1xi32>
        memref.dma_start %in[%c0], %out[%c0], %c256, %tag[%c0] : memref<256xf32>, memref<256xf32>, memref<1xi32>
        memref.dma_wait %tag[%c0], %c256 : memref<1xi32>
        memref.dealloc %vtcm : memref<256xf32, 1>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @bypass
// CHECK:       memref.dma_start {{.+}} {bypass_dst, bypass_src}
// CHECK:       memref.dma_start {{.+}} {bypass_dst}
// CHECK:       memref.dma_start {{.+}} {bypass_src}
// CHECK:       memref.dma_start {{.+}} {bypass_src}
// CHECK:       memref.dma_start {{[^{]+}}{{$}}
