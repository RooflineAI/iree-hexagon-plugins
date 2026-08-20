// This file contains a 4x1024x128 attention layer. As such, we this lowering will have multiple matrix multiplications and element-wise operations (softmax).
// Since we are running with iree-compile, verifiers are being called and compilation is generating valid code throughout the pipeline.
// Also note that this test is triggering the narrow matrices path in the custom encoder and materializer for hexagon.
// The objective of the test is to check correct lowering of these operations and generation of vector instructions.
// Currently, some vector instructions are generated (vadd, vsplat, even a few vmem), but they are not related to the tiling.
// TODO: Update this test once vector instructions are working to check for their generation

// RUN: iree-compile %s \
// RUN:   --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN \
// RUN:   --iree-hal-target-device=hexagon \
// RUN:   --iree-hexagon-v=79 \
// RUN:   --iree-hexagon-features=+hvxv79,+hvx-length128b \
// RUN:   --iree-hal-dump-executable-binaries-to=%t \
// RUN:   -o %t.vmfb
// RUN: test -s %t.vmfb

// // run: cat %t/module_module_linked_embedded_elf_hexagon.s | FileCheck %s
// // check: vmem/vscatter/vgather, etc...

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map3 = affine_map<() -> ()>
#map4 = affine_map<(d0, d1, d2) -> ()>
module @module {
  util.func public @main$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
    %cst = arith.constant dense<0> : tensor<i64>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0xFF800000 : f32
    %true = arith.constant true
    %cst_2 = arith.constant 0xFFF0000000000000 : f64
    %cst_3 = arith.constant 0.29730177875068026 : f64
    %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<4x1024x128xf16>
    %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<4x1024x128xf16>
    %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<4x1024x128xf16>
    %3 = tensor.empty() : tensor<4x1024x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<4x1024x128xf16>) outs(%3 : tensor<4x1024x128xf32>) {
    ^bb0(%in: f16, %out: f32):
      %39 = arith.extf %in : f16 to f32
      linalg.yield %39 : f32
    } -> tensor<4x1024x128xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<4x1024x128xf16>) outs(%3 : tensor<4x1024x128xf32>) {
    ^bb0(%in: f16, %out: f32):
      %39 = arith.extf %in : f16 to f32
      linalg.yield %39 : f32
    } -> tensor<4x1024x128xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<4x1024x128xf16>) outs(%3 : tensor<4x1024x128xf32>) {
    ^bb0(%in: f16, %out: f32):
      %39 = arith.extf %in : f16 to f32
      linalg.yield %39 : f32
    } -> tensor<4x1024x128xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<4x1024x128xf32>) outs(%3 : tensor<4x1024x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %39 = arith.truncf %cst_3 : f64 to f32
      %40 = arith.mulf %in, %39 : f32
      linalg.yield %40 : f32
    } -> tensor<4x1024x128xf32>
    %8 = tensor.empty() : tensor<4x128x1024xf32>
    %transposed = linalg.transpose ins(%5 : tensor<4x1024x128xf32>) outs(%8 : tensor<4x128x1024xf32>) permutation = [0, 2, 1] 
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<4x128x1024xf32>) outs(%8 : tensor<4x128x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %39 = arith.truncf %cst_3 : f64 to f32
      %40 = arith.mulf %in, %39 : f32
      linalg.yield %40 : f32
    } -> tensor<4x128x1024xf32>
    %10 = tensor.empty() : tensor<4x1024x1024xf32>
    %11 = linalg.fill ins(%cst_0 : f32) outs(%10 : tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32>
    %12 = linalg.batch_matmul ins(%7, %9 : tensor<4x1024x128xf32>, tensor<4x128x1024xf32>) outs(%11 : tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32>
    %13 = tensor.empty() : tensor<4x1024xi64>
    %14 = linalg.fill ins(%c0_i64 : i64) outs(%13 : tensor<4x1024xi64>) -> tensor<4x1024xi64>
    %15 = tensor.empty() : tensor<4x1024xf32>
    %16 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<4x1024xf32>) -> tensor<4x1024xf32>
    %17:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%12 : tensor<4x1024x1024xf32>) outs(%16, %14 : tensor<4x1024xf32>, tensor<4x1024xi64>) {
    ^bb0(%in: f32, %out: f32, %out_4: i64):
      %39 = linalg.index 2 : index
      %40 = arith.index_cast %39 : index to i64
      %41 = arith.maximumf %in, %out : f32
      %42 = arith.cmpf ogt, %in, %out : f32
      %43 = arith.select %42, %40, %out_4 : i64
      linalg.yield %41, %43 : f32, i64
    } -> (tensor<4x1024xf32>, tensor<4x1024xi64>)
    %expanded = tensor.expand_shape %17#0 [[0], [1, 2]] output_shape [4, 1024, 1] : tensor<4x1024xf32> into tensor<4x1024x1xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %expanded : tensor<4x1024x1024xf32>, tensor<4x1024x1xf32>) outs(%10 : tensor<4x1024x1024xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %39 = arith.subf %in, %in_4 : f32
      linalg.yield %39 : f32
    } -> tensor<4x1024x1024xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18 : tensor<4x1024x1024xf32>) outs(%10 : tensor<4x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %39 = math.exp %in : f32
      linalg.yield %39 : f32
    } -> tensor<4x1024x1024xf32>
    %20 = tensor.empty() : tensor<4x1024x1xf32>
    %21 = linalg.fill ins(%cst_0 : f32) outs(%20 : tensor<4x1024x1xf32>) -> tensor<4x1024x1xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : tensor<4x1024x1024xf32>) outs(%21 : tensor<4x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %39 = arith.addf %in, %out : f32
      linalg.yield %39 : f32
    } -> tensor<4x1024x1xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %22 : tensor<4x1024x1024xf32>, tensor<4x1024x1xf32>) outs(%10 : tensor<4x1024x1024xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %39 = arith.divf %in, %in_4 : f32
      linalg.yield %39 : f32
    } -> tensor<4x1024x1024xf32>
    %24 = tensor.empty() : tensor<4x1024x1024xi1>
    %25 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12 : tensor<4x1024x1024xf32>) outs(%24 : tensor<4x1024x1024xi1>) {
    ^bb0(%in: f32, %out: i1):
      %39 = arith.extf %in : f32 to f64
      %40 = arith.cmpf oeq, %39, %cst_2 : f64
      linalg.yield %40 : i1
    } -> tensor<4x1024x1024xi1>
    %26 = tensor.empty() : tensor<4x1024x1xi1>
    %27 = linalg.fill ins(%true : i1) outs(%26 : tensor<4x1024x1xi1>) -> tensor<4x1024x1xi1>
    %28 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%25 : tensor<4x1024x1024xi1>) outs(%27 : tensor<4x1024x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %39 = arith.andi %in, %out : i1
      linalg.yield %39 : i1
    } -> tensor<4x1024x1xi1>
    %29 = tensor.empty() : tensor<f32>
    %30 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = []} ins(%cst : tensor<i64>) outs(%29 : tensor<f32>) {
    ^bb0(%in: i64, %out: f32):
      %39 = arith.sitofp %in : i64 to f32
      linalg.yield %39 : f32
    } -> tensor<f32>
    %31 = linalg.generic {indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30 : tensor<f32>) outs(%10 : tensor<4x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4x1024x1024xf32>
    %32 = linalg.generic {indexing_maps = [#map2, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %31, %23 : tensor<4x1024x1xi1>, tensor<4x1024x1024xf32>, tensor<4x1024x1024xf32>) outs(%10 : tensor<4x1024x1024xf32>) {
    ^bb0(%in: i1, %in_4: f32, %in_5: f32, %out: f32):
      %39 = arith.select %in, %in_4, %in_5 : f32
      linalg.yield %39 : f32
    } -> tensor<4x1024x1024xf32>
    %33 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<4x1024x128xf32>) -> tensor<4x1024x128xf32>
    %34 = linalg.batch_matmul ins(%32, %6 : tensor<4x1024x1024xf32>, tensor<4x1024x128xf32>) outs(%33 : tensor<4x1024x128xf32>) -> tensor<4x1024x128xf32>
    %35 = tensor.empty() : tensor<4x1024x128xf16>
    %36 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34 : tensor<4x1024x128xf32>) outs(%35 : tensor<4x1024x128xf16>) {
    ^bb0(%in: f32, %out: f16):
      %39 = arith.truncf %in : f32 to f16
      linalg.yield %39 : f16
    } -> tensor<4x1024x128xf16>
    %37 = hal.tensor.barrier join(%36 : tensor<4x1024x128xf16>) => %arg4 : !hal.fence
    %38 = hal.tensor.export %37 : tensor<4x1024x128xf16> -> !hal.buffer_view
    util.return %38 : !hal.buffer_view
  }
  util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = util.null : !hal.fence
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    %1 = util.call @main$async(%arg0, %arg1, %arg2, %0, %fence) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> !hal.buffer_view
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) flags("None") : i32
    util.return %1 : !hal.buffer_view
  }
}
