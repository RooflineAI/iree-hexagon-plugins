// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_KERNELDISPATCH_H_
#define ROOF_HEXAGON_CODEGEN_KERNELDISPATCH_H_

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"

// This folder contains a replacement for the LLVMCPU launch-config
// selection policy in upstream IREE for Hexagon. It emits the same public IR
// contract (`translation_info` and `#iree_cpu.lowering_config`) so the Hexagon
// pipeline can keep reusing LLVMCPU lowering passes, but appends additional
// Hexagon-specific attributes.
//
// The strategy selection for Hexagon is currently only a placeholder and under
// development.
//
// For runnable examples and the currently expected behavior, see
// `plugins/compiler/hexagon/test/codegen/strategy/`
// `hexagon_select_lowering_strategy.mlir`.

namespace mlir::iree_compiler::hexagon::codegen {

constexpr llvm::StringLiteral kHexagonVTCMTilingConfigAttrName =
    "hexagon_vtcm_tiling_config";

/// Public facade for Hexagon launch-config selection.
mlir::LogicalResult initHexagonLaunchConfig(mlir::FunctionOpInterface funcOp);

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_KERNELDISPATCH_H_
