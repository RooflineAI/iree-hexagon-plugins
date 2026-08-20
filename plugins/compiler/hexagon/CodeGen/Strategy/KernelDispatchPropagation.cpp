// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "KernelDispatchPropagation.h"
#include "KernelDispatchHeuristics.h"

#include "cellar-hexagon/CodeGen/Pipelines/TranslationPipeline.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/IndexingMapOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <optional>

namespace mlir::iree_compiler::cellar_hexagon::codegen {
namespace {

using mlir::AffineDimExpr;
using mlir::AffineMap;
using mlir::DestinationStyleOpInterface;
using mlir::FunctionOpInterface;
using mlir::IndexingMapOpInterface;
using mlir::Operation;
using mlir::ShapedType;
using mlir::TilingInterface;
using mlir::Value;
using mlir::iree_compiler::getLoweringConfig;

/// Tracks loop dimensions in a dispatch-wide dimension space.
///
/// Each TilingInterface op starts with fresh global dimension ids for its local
/// loops. Those ids are then unified through SSA producer/consumer
/// relationships, using indexing maps when an op exposes them.
/// This implementation is copied from LLVMCPU's KernelDispatch.cpp
class IterationDimTracker {
public:
  explicit IterationDimTracker(llvm::ArrayRef<Operation *> operations) {
    for (Operation *op : operations) {
      if (mlir::isa<TilingInterface>(op) && shouldSetLoweringConfig(op)) {
        this->operations.push_back(op);
      }
    }

    // Ensure operations are processed in topological order.
    mlir::computeTopologicalSorting(this->operations);
    buildDimMapping();
  }

  /// Returns all global dimension indices associated with the given operation.
  llvm::ArrayRef<int64_t> getAllGlobalDimIdx(Operation *op) const {
    auto it = operationToGlobalDimMaps.find(op);
    if (it == operationToGlobalDimMaps.end()) {
      return {};
    }
    return it->second;
  }

  /// Returns the global dimension index corresponding to the given local loop
  /// dimension `pos` for the specified operation.
  int64_t getGlobalDimIdx(Operation *op, int64_t pos) const {
    llvm::ArrayRef<int64_t> globalDims = getAllGlobalDimIdx(op);
    return globalDims[pos];
  }

  /// Returns the total number of unique global dimension indices.
  int64_t getTotalLoopNum() const { return totalLoopNum; }

private:
  /// Builds and unifies dimension index mappings for all operations,
  /// using producer–consumer SSA value relationships.
  void buildDimMapping() {
    // Tracks equivalent global dimension indices.
    llvm::EquivalenceClasses<int64_t> indicesEquivalence;
    // For each SSA value, maps its local dimension index to a global index.
    // Value -> (value dimension -> provisional global dimension).
    llvm::SmallDenseMap<Value, llvm::SmallVector<int64_t>> valueToGlobalDimMaps;

    for (Operation *op : operations) {
      auto tilingOp = mlir::cast<TilingInterface>(op);
      int64_t numLoops = tilingOp.getLoopIteratorTypes().size();
      // Unconditionally assign new global indices, to be unified later.
      for (int64_t i = 0; i < numLoops; ++i) {
        int64_t globalIndex = totalLoopNum++;
        indicesEquivalence.insert(globalIndex);
        operationToGlobalDimMaps[op].push_back(globalIndex);
      }

      // The assigned global dimension indices are now unified based on
      // producer–consumer SSA value relationships:
      // - For operations implementing `IndexingMapOpInterface`, unify
      // dimensions by iterating over their indexing maps.
      // - For all other (unknown) operations, assume an identity mapping for
      // any value whose rank matches the operation’s loop count.
      mlir::TypeSwitch<Operation *>(op)
          .Case([&](IndexingMapOpInterface indexingMapOp) {
            propagateOnIndexingMapOp(indexingMapOp, indicesEquivalence,
                                     valueToGlobalDimMaps);
          })
          .Default([&](Operation *unknownOp) {
            propagateOnUnknownOp(unknownOp, indicesEquivalence,
                                 valueToGlobalDimMaps, numLoops);
          });
    }

    compactGlobalDims(indicesEquivalence);
  }

  /// Assign a unique identifier to all equivalent dimensions
  /// LLVMCPU does this in two steps, but this is currently uneeded for Hexagon
  void compactGlobalDims(llvm::EquivalenceClasses<int64_t> &equivalence) {
    llvm::SmallDenseMap<int64_t, int64_t> dimToCompactDim;
    int64_t nextCompactDim = 0;

    for (auto it = equivalence.begin(); it != equivalence.end(); ++it) {
      if (!(*it)->isLeader()) {
        continue;
      }
      for (auto mit = equivalence.member_begin(**it);
           mit != equivalence.member_end(); ++mit) {
        dimToCompactDim[*mit] = nextCompactDim;
      }
      ++nextCompactDim;
    }

    for (auto &opEntry : operationToGlobalDimMaps) {
      for (int64_t &dim : opEntry.second) {
        dim = dimToCompactDim.lookup(dim);
      }
    }
    totalLoopNum = nextCompactDim;
  }

  /// Ties loop dimensions together based on the operation’s indexing maps,
  /// considering only simple result dimension expressions (`AffineDimExpr`).
  ///
  /// Complex expressions (e.g., `affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2,
  /// d1 * 3 + d3)>`) are ignored because they fall outside the "loop dimension"
  /// concept. Such expressions describe how indices are computed within the
  /// innermost loop body, but they do not directly identify which loop
  /// dimensions correspond or should be tied.
  void propagateOnIndexingMapOp(
      IndexingMapOpInterface indexingMapOp,
      llvm::EquivalenceClasses<int64_t> &indicesEquivalence,
      llvm::SmallDenseMap<Value, llvm::SmallVector<int64_t>>
          &valueToGlobalDimMaps) {
    Operation *op = indexingMapOp.getOperation();

    // Operand maps tell us which consumer loop dimension reads each producer
    // value dimension. When a producer result already has a global dimension
    // mapping, unify the consumer loop with that producer dimension.
    for (mlir::OpOperand &operand : op->getOpOperands()) {
      Value value = operand.get();
      // Skip operands that have no known mapping from their producers.
      if (!valueToGlobalDimMaps.contains(value)) {
        continue;
      }
      AffineMap map = indexingMapOp.getMatchingIndexingMap(&operand);
      for (auto [dim, expr] : llvm::enumerate(map.getResults())) {
        // Stop if the current dimension exceeds the number of mapped ones.
        if (dim >= valueToGlobalDimMaps[value].size()) {
          break;
        }
        // Skip on complex expressions.
        auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr);
        if (!dimExpr) {
          continue;
        }
        int64_t loopDim = dimExpr.getPosition();
        // Unify the dimension index between the producer and the current op.
        indicesEquivalence.unionSets(valueToGlobalDimMaps[value][dim],
                                     operationToGlobalDimMaps[op][loopDim]);
      }
    }
    // Propagate to results.
    auto dsOp = cast<DestinationStyleOpInterface>(op);
    for (OpResult result : op->getResults()) {
      OpOperand *operand = dsOp.getTiedOpOperand(result);
      AffineMap map = indexingMapOp.getMatchingIndexingMap(operand);
      for (auto [dim, expr] : llvm::enumerate(map.getResults())) {
        // Skip on complex expressions.
        auto dimExpr = dyn_cast<AffineDimExpr>(expr);
        if (!dimExpr) {
          continue;
        }
        int64_t pos = dimExpr.getPosition();
        valueToGlobalDimMaps[result].push_back(
            operationToGlobalDimMaps[op][pos]);
      }
    }
  }

  /// Ties the dimensions of operations with their operands, if the operand rank
  /// matches the operation’s loop count.
  void propagateOnUnknownOp(
      Operation *op, llvm::EquivalenceClasses<int64_t> &indicesEquivalence,
      llvm::SmallDenseMap<Value, SmallVector<int64_t>> &valueToGlobalDimMaps,
      int64_t numLoops) {
    for (OpOperand &operand : op->getOpOperands()) {
      Value value = operand.get();
      if (!valueToGlobalDimMaps.contains(value) ||
          numLoops != cast<ShapedType>(value.getType()).getRank()) {
        continue;
      }
      for (int64_t i = 0; i < numLoops; ++i) {
        indicesEquivalence.unionSets(valueToGlobalDimMaps[value][i],
                                     operationToGlobalDimMaps[op][i]);
      }
    }
    // Propagate to results.
    for (Value result : op->getResults()) {
      if (numLoops == cast<ShapedType>(result.getType()).getRank()) {
        valueToGlobalDimMaps[result] = operationToGlobalDimMaps[op];
      }
    }
  }

  llvm::SmallVector<Operation *> operations;
  int64_t totalLoopNum = 0;
  llvm::SmallDenseMap<Operation *, llvm::SmallVector<int64_t>>
      operationToGlobalDimMaps;
};

static llvm::SmallVector<int64_t>
getStaticLoopRanges(TilingInterface tilingOp) {
  mlir::OpBuilder builder(tilingOp.getContext());
  builder.setInsertionPoint(tilingOp);
  llvm::SmallVector<int64_t> staticLoopRanges;
  for (mlir::Range range : tilingOp.getIterationDomain(builder)) {
    staticLoopRanges.push_back(
        mlir::getConstantIntValue(range.size).value_or(ShapedType::kDynamic));
  }
  return staticLoopRanges;
}

static int64_t getTileOrZero(Operation *op, llvm::ArrayRef<int64_t> tileSizes,
                             int64_t pos) {
  if (pos < 0 || pos >= static_cast<int64_t>(tileSizes.size())) {
    op->emitWarning()
        << "Unexpected position " << pos << " for tile sizes of length "
        << tileSizes.size()
        << "; returning 0. Something went wrong in the heuristic inference.";
    return 0;
  }
  return tileSizes[pos];
}

static llvm::SmallVector<int64_t>
buildRootVectorIntent(const RootLoweringPlan &rootPlan,
                      Operation *rootOperation,
                      const IterationDimTracker &dimTracker) {
  llvm::SmallVector<int64_t> rootVectorIntent(dimTracker.getTotalLoopNum(), 0);
  if (!rootPlan.opPlan) {
    rootOperation->emitWarning()
        << "Missing opPlan for root operation; something went wrong in the "
           "heuristic inference.";
    return rootVectorIntent;
  }

  llvm::ArrayRef<int64_t> rootGlobalDims =
      dimTracker.getAllGlobalDimIdx(rootOperation);
  for (auto [localDim, globalDim] : llvm::enumerate(rootGlobalDims)) {
    int64_t tile = std::max(
        getTileOrZero(rootOperation,
                      rootPlan.opPlan->tileLevels.vectorCommonParallel,
                      localDim),
        getTileOrZero(rootOperation,
                      rootPlan.opPlan->tileLevels.vectorReduction, localDim));
    if (tile > 0) {
      rootVectorIntent[globalDim] = std::max(rootVectorIntent[globalDim], tile);
    }
  }
  return rootVectorIntent;
}

static int64_t clampToStaticBound(int64_t tile, int64_t staticBound) {
  if (tile <= 0 || ShapedType::isDynamic(staticBound)) {
    return tile;
  }
  return std::min(tile, staticBound);
}

static void overlayRootVectorIntent(Operation *op, OpLoweringPlan &plan,
                                    const IterationDimTracker &dimTracker,
                                    llvm::ArrayRef<int64_t> rootVectorIntent) {
  auto tilingOp = mlir::dyn_cast<TilingInterface>(op);
  if (!tilingOp) {
    return;
  }

  llvm::SmallVector<int64_t> staticLoopRanges = getStaticLoopRanges(tilingOp);
  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes =
      tilingOp.getLoopIteratorTypes();
  for (auto [localDim, iteratorType] : llvm::enumerate(iteratorTypes)) {
    int64_t globalDim = dimTracker.getGlobalDimIdx(op, localDim);
    if (globalDim >= static_cast<int64_t>(rootVectorIntent.size())) {
      op->emitWarning() << "skipping propagated vector intent for local dim "
                        << localDim << ": mapped global dim " << globalDim
                        << " is outside root vector intent size "
                        << rootVectorIntent.size();
      continue;
    }

    int64_t tile = rootVectorIntent[globalDim];
    // Zero means no vector intent was propagated from the root operation
    if (tile <= 0) {
      continue;
    }
    tile = clampToStaticBound(tile, localDim < staticLoopRanges.size()
                                        ? staticLoopRanges[localDim]
                                        : ShapedType::kDynamic);

    auto warnMismatchedPlan = [&]() {
      op->emitWarning() << "skipping propagated vector intent for local dim "
                        << localDim
                        << ": lowering plan has inconsistent tile vector sizes";
    };

    if (iteratorType == mlir::utils::IteratorType::parallel) {
      if (localDim >= plan.tileLevels.vectorCommonParallel.size()) {
        warnMismatchedPlan();
        continue;
      }
      plan.tileLevels.vectorCommonParallel[localDim] = tile;
      continue;
    }

    if (iteratorType == mlir::utils::IteratorType::reduction) {
      if (localDim >= plan.tileLevels.vectorReduction.size()) {
        warnMismatchedPlan();
        continue;
      }
      plan.tileLevels.vectorReduction[localDim] = tile;
    }
  }
}

} // namespace

NonRootPlanPropagator::NonRootPlanPropagator(
    FunctionOpInterface entryPointFn, Operation *rootOperation,
    llvm::ArrayRef<Operation *> computeOps, const RootLoweringPlan &rootPlan)
    : entryPointFn(entryPointFn), rootOperation(rootOperation),
      computeOps(computeOps.begin(), computeOps.end()), rootPlan(rootPlan) {}

llvm::SmallVector<std::pair<Operation *, OpLoweringPlan>, 0>
NonRootPlanPropagator::inferPlans() const {
  llvm::SmallVector<std::pair<Operation *, OpLoweringPlan>, 0> plans;
  IterationDimTracker dimTracker(computeOps);
  llvm::SmallVector<int64_t> rootVectorIntent =
      buildRootVectorIntent(rootPlan, rootOperation, dimTracker);

  // Non-root ops still get their own local heuristic first. Propagation is a
  // sparse override: only loop dimensions proven equivalent to a root dimension
  // with vector intent are changed. Distribution tiling stays local for now.
  PolicyConfig policyConfig;
  for (Operation *op : computeOps) {
    if (op == rootOperation || !shouldSetLoweringConfig(op) ||
        getLoweringConfig(op)) {
      continue;
    }

    auto opPlan = inferOpLoweringPlan(entryPointFn, op, policyConfig);
    if (!opPlan) {
      continue;
    }
    overlayRootVectorIntent(op, *opPlan, dimTracker, rootVectorIntent);
    if (isHexagonVTCMTilingEnabled() && rootPlan.vtcmTiling) {
      std::fill(opPlan->tileLevels.cacheParallel.begin(),
                opPlan->tileLevels.cacheParallel.end(), 0);
    }
    plans.emplace_back(op, *opPlan);
  }
  return plans;
}

} // namespace mlir::iree_compiler::cellar_hexagon::codegen
