// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_KERNELDISPATCH_H_
#define ROOF_HEXAGON_CODEGEN_KERNELDISPATCH_H_

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::cellar_hexagon::codegen {

mlir::LogicalResult initHexagonLaunchConfig(mlir::FunctionOpInterface funcOp);

} // namespace mlir::iree_compiler::cellar_hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_KERNELDISPATCH_H_
