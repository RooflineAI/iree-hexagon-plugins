// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_KERNELDISPATCH_H_
#define ROOF_HEXAGON_CODEGEN_KERNELDISPATCH_H_

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"

// Hexagon launch-config selection emits translation_info,
// #iree_hexagon.lowering_config and optional VTCM staging configuration for
// the Hexagon lowering pipelines.
//
// Strategy selection is implemented under `Planning/`.
// This header exposes only the pass-facing facade and the one
// Hexagon-specific attribute name consumed by later transforms.
//
// For runnable examples and the currently expected behavior, see
// `plugins/compiler/hexagon/test/codegen/strategy/`.

namespace mlir::iree_compiler::hexagon::codegen {

constexpr llvm::StringLiteral kHexagonVTCMTilingConfigAttrName =
    "hexagon_vtcm_tiling_config";

/// Public facade for Hexagon launch-config selection.
mlir::LogicalResult initHexagonLaunchConfig(mlir::FunctionOpInterface funcOp);

} // namespace mlir::iree_compiler::hexagon::codegen

#endif // ROOF_HEXAGON_CODEGEN_KERNELDISPATCH_H_
