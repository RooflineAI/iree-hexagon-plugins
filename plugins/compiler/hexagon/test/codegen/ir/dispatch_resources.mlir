// RUN: iree-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK: #iree_hexagon.dispatch_resources<hvx = true, hmx = false, threads = 1, vtcm_bytes = 4096>
func.func @roundtrip() attributes {resources = #iree_hexagon.dispatch_resources<hvx = true, hmx = false, threads = 1, vtcm_bytes = 4096>} {
  return
}

// -----

// expected-error @+1 {{vtcm_bytes must be nonnegative, got -1}}
func.func @negative_vtcm() attributes {resources = #iree_hexagon.dispatch_resources<hvx = true, hmx = false, threads = 1, vtcm_bytes = -1>} {
  return
}

// -----

// expected-error @+1 {{threads must be at least 1, got 0}}
func.func @zero_threads() attributes {resources = #iree_hexagon.dispatch_resources<hvx = true, hmx = false, threads = 0, vtcm_bytes = 0>} {
  return
}
