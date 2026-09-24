// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "KernelDispatch.h"

#include "Planning/DispatchPlanner.h"
#include "hexagon/CodeGen/Pipelines/TranslationPipeline.h"

namespace mlir::iree_compiler::hexagon::codegen {

LogicalResult initHexagonLaunchConfig(FunctionOpInterface funcOp) {
  planning::PlanningOptions options;
  options.enableVTCM = isHexagonVTCMTilingEnabled();
  options.enableHmxMatmul = isHexagonHmxMatmulEnabled();
  return planning::configureDispatch(funcOp, options);
}

} // namespace mlir::iree_compiler::hexagon::codegen
