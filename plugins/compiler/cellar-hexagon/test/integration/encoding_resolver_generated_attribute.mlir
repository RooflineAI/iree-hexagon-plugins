// Verify SpecializeEncodings materializes the hexagon resolver with correct encoding_info
// and that the isSerialized function from HexagonSerializableAttr returns true when appropriate 
// (EncodeHostTensors does not fail and eliminates stream.tensor.sizeof as a result)
// This test checks integration of multiple passes the lowering pipeline depends on and correct working of all of them

// RUN: iree-compile %s \
// RUN:   --iree-opt-data-tiling=true \
// RUN:   --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN \
// RUN:   --iree-hal-target-device=hexagon \
// RUN:   --iree-hexagon-features=+hvxv79,+hvx-length128b \
// RUN:   --compile-to=stream \
// RUN:   --mlir-disable-threading \
// RUN:   --mlir-print-ir-after=iree-stream-specialize-encodings \
// RUN:   --mlir-print-ir-after=iree-stream-encode-host-tensors \
// RUN:   -o /dev/null 2>&1 | FileCheck %s

// CHECK: IR Dump After SpecializeEncodingsPass
// CHECK: #iree_encoding.layout<[#iree_hexagon.hexagon_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [4, 4], outerDimsPerm = [0, 1]}}>]>
// CHECK: #iree_encoding.layout<[#iree_hexagon.hexagon_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [1, 0], innerTileSizes = [4, 4], outerDimsPerm = [1, 0]}}>]>

// CHECK: IR Dump After EncodeHostTensorsPass
// CHECK-NOT: stream.tensor.sizeof

module {
  func.func @matmul(%lhs: tensor<4x4xf32>, %rhs: tensor<4x4xf32>)
      -> tensor<4x4xf32> {
    %init = tensor.empty() : tensor<4x4xf32>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<4x4xf32>, tensor<4x4xf32>)
        outs(%init : tensor<4x4xf32>)
        -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }
}
