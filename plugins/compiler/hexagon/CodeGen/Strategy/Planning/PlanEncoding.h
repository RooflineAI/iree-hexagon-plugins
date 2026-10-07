// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_PLANENCODING_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_PLANENCODING_H_

#include "DispatchPlanTypes.h"
#include "PipelineContract.h"

#include "hexagon/CodeGen/IR/HexagonAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {

/// Prepared configuration for one operation. Root-only stages, such as the
/// VTCM stage, can only be present on the entry whose op equals
/// DispatchShape::root.
struct EncodedOpPlan {
  Operation *op = nullptr;
  IREE::Hexagon::LoweringConfigAttr loweringConfig;
};

/// Fully prepared configuration. Constructing this object does not mutate IR.
struct EncodedDispatchPlan {
  FunctionOpInterface entryPoint;
  IREE::Codegen::TranslationInfoAttr translationInfo;
  llvm::SmallVector<EncodedOpPlan> operations;
};

/// Mechanically assigns compute tiles to Hexagon levels using local iterator
/// type and root coverage, encodes root-only fields, and prepares all
/// attributes without applying them. The complete plan must already be
/// verified.
FailureOr<EncodedDispatchPlan>
encodeDispatchPlan(const PlanningContext &context,
                   const DispatchShape &dispatchShape, const DispatchPlan &plan,
                   const PipelineContract &pipelineContract);

/// The only planning operation allowed to mutate MLIR.
LogicalResult applyEncodedDispatchPlan(const EncodedDispatchPlan &encodedPlan);

} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_PLANENCODING_H_
