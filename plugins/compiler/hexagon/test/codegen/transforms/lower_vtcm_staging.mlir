// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-hexagon-lower-vtcm-staging))' \
// RUN:   --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @lower_vtcm_stage(
// CHECK: %[[COPY:.+]] = bufferization.alloc_tensor() copy(%arg0) {memory_space = 1 : i64} : tensor<4x4xf32>
// CHECK: return %[[COPY]] : tensor<4x4xf32>
// CHECK-NOT: iree_hexagon.stage_to_vtcm

func.func @lower_vtcm_stage(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = iree_hexagon.stage_to_vtcm %arg0 : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @lower_vtcm_empty(
// CHECK: %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<4x4xf32>
// CHECK: return %[[ALLOC]] : tensor<4x4xf32>
// CHECK-NOT: iree_hexagon.vtcm_empty

func.func @lower_vtcm_empty() -> tensor<4x4xf32> {
  %0 = iree_hexagon.vtcm_empty() : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

// Dynamic copy

// CHECK-LABEL: func.func @lower_vtcm_stage_dynamic(
// CHECK:     %[[COPY:.+]] = bufferization.alloc_tensor() copy(%arg0)
// CHECK-SAME:   {memory_space = 1 : i64} : tensor<?x4xf32>
// CHECK:     return %[[COPY]] : tensor<?x4xf32>
// CHECK-NOT: iree_hexagon.stage_to_vtcm

func.func @lower_vtcm_stage_dynamic(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = iree_hexagon.stage_to_vtcm %arg0 : tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

// -----

// Dynamic allocation

// CHECK-LABEL: func.func @lower_vtcm_empty_dynamic(
// CHECK:     %[[ALLOC:.+]] = bufferization.alloc_tensor(%arg0)
// CHECK-SAME:   {memory_space = 1 : i64} : tensor<?x4xf32>
// CHECK:     return %[[ALLOC]] : tensor<?x4xf32>
// CHECK-NOT: iree_hexagon.vtcm_empty

func.func @lower_vtcm_empty_dynamic(%arg0: index) -> tensor<?x4xf32> {
  %0 = iree_hexagon.vtcm_empty(%arg0) : tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}
