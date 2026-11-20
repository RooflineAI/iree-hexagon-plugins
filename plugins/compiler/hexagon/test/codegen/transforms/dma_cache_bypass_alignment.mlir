// RUN: iree-opt --iree-stream-resource-min-offset-alignment=64 --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hexagon-dma-cache-bypass)))" --verify-diagnostics %s

// A DMA writing a binding by bypassing the data cache needs buffers aligned to
// the cache line, or a cache line shared with another buffer could be written
// back over its result.

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon">
#layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

hal.executable private @executable {
  hal.executable.variant public @variant target(#target) {
    // expected-error @+1 {{binding 0 is written by DMA bypassing the data cache, which needs buffers aligned to at least 128 bytes, but they are aligned to 64; pass --iree-stream-resource-min-offset-alignment=128}}
    hal.executable.export public @dma_write ordinal(0) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @dma_write() {
        %c0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %out = hal.interface.binding.subspan layout(#layout) binding(0) offset(%c0) : memref<256xf32>
        %vtcm = memref.alloc() : memref<256xf32, 1>
        %tag = memref.alloca() : memref<1xi32>
        memref.dma_start %vtcm[%c0], %out[%c0], %c256, %tag[%c0] : memref<256xf32, 1>, memref<256xf32>, memref<1xi32>
        memref.dma_wait %tag[%c0], %c256 : memref<1xi32>
        memref.dealloc %vtcm : memref<256xf32, 1>
        return
      }
    }
  }
}
