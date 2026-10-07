// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PlanEncoding.h"

#include "DecisionTrace.h"

#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

using IREE::Hexagon::LoweringConfigAttr;
using IREE::Hexagon::TilingLevel;

SmallVector<int64_t> getSizes(ArrayRef<TileDecision> decisions) {
  return llvm::map_to_vector(
      decisions, [](const TileDecision &tile) { return tile.size; });
}

bool anyNonZero(ArrayRef<int64_t> sizes) {
  return llvm::any_of(sizes, [](int64_t size) { return size != 0; });
}

void appendLevel(SmallVectorImpl<NamedAttribute> &items, MLIRContext *context,
                 TilingLevel level, ArrayRef<int64_t> sizes,
                 bool preserveAllZero = false) {
  if (sizes.empty() || (!preserveAllZero && !anyNonZero(sizes)))
    return;
  items.emplace_back(IREE::Hexagon::getTilingLevelName(level),
                     LoweringConfigAttr::getTilingLevelAttr(context, sizes));
}

LoweringConfigAttr
encodeOpConfig(const DispatchShape &shape, const OpShape &opShape,
               ArrayRef<TileDecision> distribution,
               ArrayRef<TileDecision> cache, ArrayRef<TileDecision> compute,
               ArrayRef<TileDecision> vtcm, ArrayRef<TileDecision> hmx,
               bool preserveZeroDistribution) {
  SmallVector<int64_t> distributionSizes = getSizes(distribution);
  SmallVector<int64_t> cacheSizes = getSizes(cache);
  SmallVector<int64_t> common(compute.size(), 0);
  SmallVector<int64_t> reduction(compute.size(), 0);
  SmallVector<int64_t> inner(compute.size(), 0);
  // Assign each dimension to exactly one vector level. getVectorSizes() merges
  // these levels and returns nullopt if two levels contain a nonzero value for
  // the same dimension, causing downstream vector-size consumers to skip it.
  for (const LocalDim &dimension : opShape.dimensions) {
    int64_t size = compute[dimension.position].size;
    switch (classifyComputeTileLevel(shape, dimension)) {
    case ComputeTileLevel::Reduction:
      reduction[dimension.position] = size;
      break;
    case ComputeTileLevel::CommonParallel:
      common[dimension.position] = size;
      break;
    case ComputeTileLevel::InnerParallel:
      inner[dimension.position] = size;
      break;
    }
  }

  SmallVector<NamedAttribute> items;
  MLIRContext *context = opShape.op->getContext();
  appendLevel(items, context, TilingLevel::DistributionTiles, distributionSizes,
              preserveZeroDistribution);
  appendLevel(items, context, TilingLevel::CacheParallelTiles, cacheSizes);
  appendLevel(items, context, TilingLevel::VectorCommonParallelTiles, common);
  appendLevel(items, context, TilingLevel::VectorReductionTiles, reduction);
  appendLevel(items, context, TilingLevel::VectorInnerParallelTiles, inner);
  appendLevel(items, context, TilingLevel::VTCMTiles, getSizes(vtcm));
  appendLevel(items, context, TilingLevel::HmxTiles, getSizes(hmx));
  if (items.empty())
    return {};
  return LoweringConfigAttr::get(context, items);
}

} // namespace

FailureOr<EncodedDispatchPlan>
encodeDispatchPlan(const PlanningContext &context,
                   const DispatchShape &dispatchShape, const DispatchPlan &plan,
                   const PipelineContract &pipelineContract) {
  FunctionOpInterface entryPoint = context.entryPoint;
  EncodedDispatchPlan encoded;
  encoded.entryPoint = entryPoint;
  MLIRContext *mlirContext = entryPoint.getContext();
  DictionaryAttr pipelineConfig;
  if (plan.strategy.requestLoopPeeling) {
    StringAttr name = getEnableLoopPeelingAttrName(mlirContext);
    pipelineConfig = DictionaryAttr::get(
        mlirContext,
        ArrayRef<NamedAttribute>{{name, UnitAttr::get(mlirContext)}});
  }
  encoded.translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      mlirContext,
      IREE::Hexagon::PipelineAttr::get(mlirContext, plan.strategy.pipeline),
      SymbolRefAttr(), /*workgroupSize=*/{}, /*subgroupSize=*/0,
      pipelineConfig);

  llvm::DenseMap<Operation *, const OpComputeTilePlan *> nonRootPlansByOp;
  for (const OpComputeTilePlan &opPlan : plan.nonRootComputeTilePlans)
    nonRootPlansByOp[opPlan.op] = &opPlan;

  for (const OpShape &opShape : dispatchShape.operations) {
    EncodedOpPlan opPlan;
    opPlan.op = opShape.op;
    if (opShape.op == dispatchShape.root) {
      const RootTilingPlan &root = plan.strategy.rootTiling;
      // Root-anchored Hexagon passes identify the root by the unique operation
      // carrying a distribution level. Preserve the all-zero Hexagon level as
      // that marker when required by the pipeline contract.
      ArrayRef<TileDecision> vtcm;
      if (root.vtcm)
        vtcm = root.vtcm->tileSizes;
      opPlan.loweringConfig =
          encodeOpConfig(dispatchShape, opShape, root.distributionTile,
                         root.cacheTile, root.computeTile, vtcm, root.hmxTile,
                         pipelineContract.requiresUniqueRootAnchor);
    } else if (auto it = nonRootPlansByOp.find(opShape.op);
               it != nonRootPlansByOp.end()) {
      opPlan.loweringConfig = encodeOpConfig(
          dispatchShape, opShape, /*distribution=*/{}, /*cache=*/{},
          it->second->computeTile, /*vtcm=*/{}, /*hmx=*/{},
          /*preserveZeroDistribution=*/false);
    }
    if (opPlan.loweringConfig)
      encoded.operations.push_back(opPlan);
  }

  if (pipelineContract.requiresUniqueRootAnchor) {
    int64_t anchors = 0;
    Operation *anchor = nullptr;
    for (const EncodedOpPlan &opPlan : encoded.operations) {
      if (opPlan.loweringConfig &&
          opPlan.loweringConfig.hasWorkgroupTilingLevel()) {
        ++anchors;
        anchor = opPlan.op;
      }
    }
    if (anchors != 1 || anchor != dispatchShape.root) {
      entryPoint.emitError(
          "encoded Hexagon plan does not have one unique root anchor");
      return failure();
    }
  }
  context.trace.record(DecisionStage::Encoding, DecisionKind::Derived,
                       "encoded verified dispatch plan");
  return encoded;
}

LogicalResult applyEncodedDispatchPlan(const EncodedDispatchPlan &encodedPlan) {
  FunctionOpInterface entryPoint = encodedPlan.entryPoint;
  if (getTranslationInfo(encodedPlan.entryPoint)) {
    entryPoint.emitError(
        "cannot apply Hexagon dispatch plan: translation info already exists");
    return failure();
  }
  for (const EncodedOpPlan &opPlan : encodedPlan.operations) {
    if (getLoweringConfig(opPlan.op)) {
      opPlan.op->emitError(
          "cannot apply Hexagon dispatch plan: operation already has a "
          "lowering configuration");
      return failure();
    }
  }

  if (failed(setTranslationInfo(encodedPlan.entryPoint,
                                encodedPlan.translationInfo)))
    return failure();
  for (const EncodedOpPlan &opPlan : encodedPlan.operations)
    setLoweringConfig(opPlan.op, opPlan.loweringConfig);
  return success();
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning
