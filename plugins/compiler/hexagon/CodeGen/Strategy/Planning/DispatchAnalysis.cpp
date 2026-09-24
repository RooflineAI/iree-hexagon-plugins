// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "DispatchAnalysis.h"

#include "DecisionTrace.h"

#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/IndexingMapOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

constexpr int64_t kUnknownGlobalDim = -1;

bool isEligible(Operation *op) {
  auto tilingOp = dyn_cast<TilingInterface>(op);
  return tilingOp && !tilingOp.getLoopIteratorTypes().empty();
}

/// Builds a dispatch-wide dimension space from operation-local loop spaces.
/// Each eligible operation starts with fresh dimension identifiers. Simple SSA
/// value/indexing-map relationships union equivalent identifiers before the
/// resulting equivalence classes are compacted.
///
/// The provisional-ID, union-find, and compaction algorithm is derived from
/// LLVMCPU's `IterationDimTracker` in `Codegen/LLVMCPU/KernelDispatch.cpp`.
/// This implementation is not kept in lockstep with LLVMCPU: it records shaped
/// values at first use so readers of external values can be related, excludes
/// `tensor.empty` from that rule, and uses a rank-matched identity fallback for
/// operations without indexing maps. In particular, it does not reproduce
/// LLVMCPU's specialized pack/unpack propagation.
class DimensionGraph {
public:
  explicit DimensionGraph(ArrayRef<Operation *> orderedComputeOps) {
    llvm::copy_if(orderedComputeOps, std::back_inserter(operations),
                  isEligible);
    computeTopologicalSorting(operations);
    build();
  }

  ArrayRef<int64_t> getDimensions(Operation *op) const {
    auto it = operationDimensions.find(op);
    return it == operationDimensions.end() ? ArrayRef<int64_t>() : it->second;
  }

  int64_t getDimensionCount() const { return compactDimensionCount; }

private:
  void build() {
    llvm::EquivalenceClasses<int64_t> equivalence;
    llvm::SmallDenseMap<Value, SmallVector<int64_t>> valueDimensions;

    for (Operation *op : operations) {
      auto tilingOp = cast<TilingInterface>(op);
      int64_t loopCount = tilingOp.getLoopIteratorTypes().size();
      SmallVector<int64_t> &opDimensions = operationDimensions[op];
      for (int64_t i = 0; i < loopCount; ++i) {
        int64_t id = provisionalDimensionCount++;
        equivalence.insert(id);
        opDimensions.push_back(id);
      }

      if (auto indexingOp = dyn_cast<IndexingMapOpInterface>(op)) {
        propagateIndexingOp(indexingOp, equivalence, valueDimensions);
      } else {
        propagateIdentityOp(op, loopCount, equivalence, valueDimensions);
      }
    }

    compact(equivalence);
  }

  void propagateIndexingOp(
      IndexingMapOpInterface indexingOp,
      llvm::EquivalenceClasses<int64_t> &equivalence,
      llvm::SmallDenseMap<Value, SmallVector<int64_t>> &valueDimensions) {
    Operation *op = indexingOp.getOperation();
    ArrayRef<int64_t> localDimensions = operationDimensions.find(op)->second;

    for (OpOperand &operand : op->getOpOperands()) {
      auto shapedType = dyn_cast<ShapedType>(operand.get().getType());
      if (!shapedType)
        continue;
      // A tensor.empty carries no data, so two operations using the same one
      // as their destination are not iterating a shared value - they just got
      // the same allocation, and CSE makes that happen for any two results of
      // equal type. Unifying through it merges unrelated dimensions: in a
      // transposed-RHS contraction whose LHS producer and accumulator fill
      // share an empty, the root's own n and k collapse into one unified
      // dimension, and every bound derived from that dimension is then wrong.
      if (isa_and_present<tensor::EmptyOp>(operand.get().getDefiningOp()))
        continue;
      AffineMap map = indexingOp.getMatchingIndexingMap(&operand);
      SmallVector<int64_t> operandDimensions(shapedType.getRank(),
                                             kUnknownGlobalDim);
      for (auto [valueDim, expression] : llvm::enumerate(map.getResults())) {
        if (valueDim >= operandDimensions.size())
          break;
        // Only a direct affine dimension establishes a one-to-one loop
        // correspondence. Composed affine expressions compute indices but do
        // not identify two dispatch dimensions as equivalent.
        auto dimExpression = dyn_cast<AffineDimExpr>(expression);
        if (!dimExpression ||
            dimExpression.getPosition() >= localDimensions.size())
          continue;
        operandDimensions[valueDim] =
            localDimensions[dimExpression.getPosition()];
      }

      auto [known, inserted] =
          valueDimensions.try_emplace(operand.get(), operandDimensions);
      // Record the first reader of an external value (including block
      // arguments) so later readers can be unified with it.
      if (inserted)
        continue;
      for (auto [valueDim, currentDimension] :
           llvm::enumerate(operandDimensions)) {
        if (valueDim >= known->second.size() ||
            currentDimension == kUnknownGlobalDim)
          continue;
        if (known->second[valueDim] == kUnknownGlobalDim) {
          known->second[valueDim] = currentDimension;
          continue;
        }
        equivalence.unionSets(known->second[valueDim], currentDimension);
      }
    }

    auto destinationOp = dyn_cast<DestinationStyleOpInterface>(op);
    if (!destinationOp)
      return;
    for (OpResult result : op->getResults()) {
      auto shapedType = dyn_cast<ShapedType>(result.getType());
      if (!shapedType)
        continue;
      // Preserve result-dimension positions when an indexing expression is not
      // a direct loop dimension. Compacting the known entries would make later
      // readers associate a value dimension with the wrong loop.
      SmallVector<int64_t> resultDimensions(shapedType.getRank(),
                                            kUnknownGlobalDim);
      OpOperand *tiedOperand = destinationOp.getTiedOpOperand(result);
      AffineMap map = indexingOp.getMatchingIndexingMap(tiedOperand);
      for (auto [valueDim, expression] : llvm::enumerate(map.getResults())) {
        if (valueDim >= resultDimensions.size())
          break;
        auto dimExpression = dyn_cast<AffineDimExpr>(expression);
        if (!dimExpression ||
            dimExpression.getPosition() >= localDimensions.size())
          continue;
        resultDimensions[valueDim] =
            localDimensions[dimExpression.getPosition()];
      }
      valueDimensions[result] = std::move(resultDimensions);
    }
  }

  void propagateIdentityOp(
      Operation *op, int64_t loopCount,
      llvm::EquivalenceClasses<int64_t> &equivalence,
      llvm::SmallDenseMap<Value, SmallVector<int64_t>> &valueDimensions) {
    ArrayRef<int64_t> localDimensions = operationDimensions.find(op)->second;
    for (OpOperand &operand : op->getOpOperands()) {
      auto shapedType = dyn_cast<ShapedType>(operand.get().getType());
      if (!shapedType || shapedType.getRank() != loopCount)
        continue;
      // Without an indexing interface, a rank match is the only available
      // evidence, so conservatively assume an identity loop-to-value mapping.
      auto [known, inserted] = valueDimensions.try_emplace(
          operand.get(), llvm::to_vector(localDimensions));
      if (inserted || known->second.size() != loopCount)
        continue;
      for (int64_t i = 0; i < loopCount; ++i) {
        if (known->second[i] == kUnknownGlobalDim) {
          known->second[i] = localDimensions[i];
          continue;
        }
        equivalence.unionSets(known->second[i], localDimensions[i]);
      }
    }
    for (Value result : op->getResults()) {
      auto shapedType = dyn_cast<ShapedType>(result.getType());
      if (shapedType && shapedType.getRank() == loopCount)
        valueDimensions[result] = llvm::to_vector(localDimensions);
    }
  }

  void compact(llvm::EquivalenceClasses<int64_t> &equivalence) {
    llvm::DenseMap<int64_t, int64_t> compactIds;
    int64_t nextId = 0;
    for (auto it = equivalence.begin(); it != equivalence.end(); ++it) {
      if (!(*it)->isLeader())
        continue;
      for (auto member = equivalence.member_begin(**it);
           member != equivalence.member_end(); ++member)
        compactIds[*member] = nextId;
      ++nextId;
    }
    for (auto &[op, dimensions] : operationDimensions) {
      (void)op;
      for (int64_t &dimension : dimensions)
        dimension = compactIds.lookup(dimension);
    }
    compactDimensionCount = nextId;
  }

  SmallVector<Operation *> operations;
  llvm::SmallDenseMap<Operation *, SmallVector<int64_t>> operationDimensions;
  int64_t provisionalDimensionCount = 0;
  int64_t compactDimensionCount = 0;
};

SmallVector<int64_t> getStaticLoopRanges(Operation *op, int64_t loopCount) {
  // TilingInterface::getIterationDomain may materialize dim operations. Keep
  // analysis read-only by using the static indexing-map query when available
  // and conservatively treating other loop bounds as dynamic.
  if (auto indexingOp = dyn_cast<IndexingMapOpInterface>(op))
    return indexingOp.getStaticLoopRanges();
  return SmallVector<int64_t>(loopCount, ShapedType::kDynamic);
}

} // namespace

FailureOr<DispatchShape>
analyzeDispatch(const PlanningContext &context,
                ArrayRef<Operation *> orderedComputeOps) {
  FailureOr<Operation *> root = getRootOperation(orderedComputeOps);
  if (failed(root))
    return failure();

  DimensionGraph graph(orderedComputeOps);

  DispatchShape shape;
  shape.entryPoint = context.entryPoint;
  shape.root = *root;
  shape.globalDimensions.resize(graph.getDimensionCount());
  for (auto [id, global] : llvm::enumerate(shape.globalDimensions))
    global.id = id;

  for (Operation *op : orderedComputeOps) {
    auto tilingOp = dyn_cast<TilingInterface>(op);
    if (!tilingOp || tilingOp.getLoopIteratorTypes().empty())
      continue;
    ArrayRef<int64_t> globalIds = graph.getDimensions(op);
    SmallVector<utils::IteratorType> iteratorTypes =
        tilingOp.getLoopIteratorTypes();
    SmallVector<int64_t> staticRanges =
        getStaticLoopRanges(op, iteratorTypes.size());
    if (globalIds.size() != iteratorTypes.size() ||
        staticRanges.size() != iteratorTypes.size()) {
      op->emitError("inconsistent loop rank while analyzing Hexagon dispatch");
      return failure();
    }
    if (auto indexingOp = dyn_cast<IndexingMapOpInterface>(op)) {
      for (OpOperand &operand : op->getOpOperands()) {
        if (!isa<ShapedType>(operand.get().getType()))
          continue;
        if (indexingOp.getMatchingIndexingMap(&operand).getNumDims() !=
            iteratorTypes.size()) {
          op->emitError(
              "operand indexing map rank does not match analyzed loop rank");
          return failure();
        }
      }
    }

    OpShape opShape;
    opShape.op = op;
    opShape.ordinal = shape.operations.size();
    for (auto [position, globalId] : llvm::enumerate(globalIds)) {
      LocalDim local;
      local.position = position;
      local.global = globalId;
      local.staticExtent = staticRanges[position];
      local.iteratorType = iteratorTypes[position];
      opShape.dimensions.push_back(local);
      if (op == shape.root)
        shape.globalDimensions[local.global].presentInRoot = true;
    }
    shape.operations.push_back(std::move(opShape));
  }

  if (const OpShape *rootShape = findOpShape(shape, shape.root)) {
    context.trace.recordForOp(
        DecisionStage::Analysis, DecisionKind::Selected, rootShape->ordinal,
        shape.root->getName().getStringRef(), "selected as dispatch root");
  } else {
    shape.root = nullptr;
    context.trace.record(DecisionStage::Fallback, DecisionKind::Fallback,
                         "dispatch has no usable compute root");
  }
  return shape;
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning
