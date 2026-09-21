// LLVMCPUTileLastOpAndFuseProducerConsumerPass can select independent anchors
// on either side of the distribution root. Verify end-to-end that a live
// producer before the root is tiled at its inner-parallel level.
//
// RUN: iree-opt \
// RUN:   --iree-hexagon-enable-vtcm-tiling=false \
// RUN:   --mlir-disable-threading \
// RUN:   --mlir-print-ir-after=iree-llvmcpu-tile-and-fuse-producer-consumer \
// RUN:   --mlir-print-ir-module-scope \
// RUN:   --pass-pipeline='builtin.module(iree-hexagon-select-lowering-strategy,func.func(iree-hexagon-lower-executable-target))' \
// RUN:   %s 2>&1 | FileCheck %s

#target = #hal.executable.target<"hexagon", "embedded-elf-hexagon", {cpu = "hexagonv79", cpu_features = "+hvxv79,+hvx-length128b", max_stack_allocation_size = 16384 : i64, target_triple = "hexagon-unknown-unknown-elf"}>

func.func @producer_has_root_private_dimension(
    %input: tensor<4x8x64xf32>) -> (tensor<4x8x64xf32>, tensor<4x8xf32>)
    attributes {hal.executable.target = #target} {
  %producer_empty = tensor.empty() : tensor<4x8x64xf32>
  %producer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input : tensor<4x8x64xf32>)
      outs(%producer_empty : tensor<4x8x64xf32>) {
  ^bb0(%value: f32, %out: f32):
    %scaled = arith.mulf %value, %value : f32
    linalg.yield %scaled : f32
  } -> tensor<4x8x64xf32>

  %root_empty = tensor.empty() : tensor<4x8xf32>
  %root = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1, 0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%producer : tensor<4x8x64xf32>)
      outs(%root_empty : tensor<4x8xf32>) {
  ^bb0(%value: f32, %out: f32):
    linalg.yield %value : f32
  } -> tensor<4x8xf32>
  return %producer, %root : tensor<4x8x64xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: IR Dump After LLVMCPUTileAndFuseProducerConsumerPass: iree-llvmcpu-tile-and-fuse-producer-consumer{anchor-on-root-op=false
// CHECK-SAME: tiling-level=vector_inner_parallel}
// CHECK-LABEL: func.func @producer_has_root_private_dimension(
// CHECK: scf.forall (%[[INNER_IV:.+]]) = (0) to (64) step (32)
// CHECK: tensor.extract_slice {{.*}}[0, 0, %[[INNER_IV]]] [1, 8, 32]
// CHECK: linalg.generic
// CHECK-SAME: tensor<1x8x32xf32>
