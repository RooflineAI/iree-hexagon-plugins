// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Selecting a Tilesize based of enumeration+ cost function

#include "VectorTileSearch.h"

#include "DecisionTrace.h"
#include "RegisterEstimation/DispatchRegisterGraph.h"
#include "RegisterEstimation/EstimatorConfig.h"
#include "Strategies/StrategySupport.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <memory>

#define DEBUG_TYPE "iree-hexagon-dispatch-planning"

namespace mlir::iree_compiler::hexagon::codegen::planning {

namespace {

// Search spaces beyond this are skipped outright
constexpr int64_t kMaxCandidates = 200000;

/// Every divisor of `bound`, ascending. `bound` is always static and
/// positive (AnchorDim::extent's contract).
SmallVector<int64_t> divisorsOf(int64_t bound) {
  SmallVector<int64_t> divisors;
  for (int64_t i = 1; i * i <= bound; ++i) {
    if (bound % i != 0)
      continue;
    divisors.push_back(i);
    if (i != bound / i)
      divisors.push_back(bound / i);
  }
  llvm::sort(divisors);
  return divisors;
}
// could probably cache/precompute for relevant sizes at compile time - if we
// are working on multiple dispatches called once per dispatch

/// Candidate sizes for one dim.
// The vectorized dim matches the vector width, codegen needs to handle
// peel/remainder loops if necessary
SmallVector<int64_t> candidatesForDim(int64_t bound, bool isVectorizedDim,
                                      int64_t lanes) {
  if (!isVectorizedDim || lanes <= 1)
    return divisorsOf(bound);
  SmallVector<int64_t> candidates;
  for (int64_t k = 1;; ++k) {
    int64_t candidate = std::min(bound, k * lanes);
    candidates.push_back(candidate);
    if (candidate == bound)
      break;
  }
  return candidates;
}

inline int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

/// One load/store node's contribution to traffic: which dims its
/// tile shape depends on, and its element width.
struct TrafficNode {
  llvm::SmallBitVector dims;
  unsigned bits = 0;
};

/// VTCM traffic (in bits moved) for one candidate tile
int64_t trafficOf(ArrayRef<TrafficNode> nodes, ArrayRef<int64_t> bound,
                  const llvm::SmallBitVector &isReductionDim,
                  ArrayRef<int64_t> tile) {
  int64_t total = 0;
  for (const TrafficNode &node : nodes) {
    int64_t elements = 1;
    int64_t multiplicity = 1;
    for (unsigned d = 0, e = tile.size(); d < e; ++d) {
      bool inMap = node.dims.test(d);
      if (inMap)
        elements *= tile[d];
      if (inMap || !isReductionDim.test(d))
        multiplicity *= ceilDiv(bound[d], tile[d]);
    }
    total += elements * multiplicity * static_cast<int64_t>(node.bits);
  }
  return total;
}

// iteration size of the tile, trip count of the innermost loop
// used for the cost metric
int64_t iterationsOf(ArrayRef<int64_t> bound, ArrayRef<int64_t> tile) {
  int64_t iterations = 1;
  for (unsigned d = 0, e = tile.size(); d < e; ++d)
    iterations *= ceilDiv(bound[d], tile[d]);
  return iterations;
}

/// Whether `tile` matches a tile shape known to crash Hexagon's LLVM
/// backend: `llvm::EVT::getSimpleVT(): isSimple()` in type legalization,
/// reproduced on `matmul_prime_197x768x768`
/// if the vectorized dim's tile spans a native vector count that is not one of
/// {1, 2, 4} and a non-vectorized dim ==1, backend crashes with the error above
/// might be a bug in the backend since having 3 vectors active should be
/// allowed as well; probably has implications regarding instruction packing
/// Nevertheless, for now, outright reject such shapes
bool matchesKnownHexagonCrashShape(ArrayRef<int64_t> tile,
                                   ArrayRef<AnchorDim> anchorDims,
                                   std::optional<unsigned> vectorizedDim,
                                   int64_t lanes) {
  if (!vectorizedDim || lanes <= 0)
    return false;
  int64_t laneVectors = ceilDiv(tile[*vectorizedDim], lanes);
  // matched expected register count
  if (laneVectors == 1 || laneVectors == 2 || laneVectors == 4)
    return false;
  // only is a problem if another dim ==1
  for (unsigned d = 0, e = anchorDims.size(); d < e; ++d) {
    if (d == *vectorizedDim ||
        anchorDims[d].iteratorType == utils::IteratorType::reduction)
      continue;
    if (tile[d] > 0 && tile[d] <= 2)
      return true;
  }
  return false;
}

/// Evaluates `candidate`'s register pressure and checks it against `budget`,
/// logging the outcome either way. Failure means `candidate` is infeasible:
/// either `evaluate` itself failed, or its peak exceeds `budget`.
FailureOr<RegisterPressure>
evaluateWithinBudget(const DispatchRegisterGraph &graph,
                     const ArrayRef<int64_t> candidate, const int64_t budget) {
  FailureOr<RegisterPressure> pressure = graph.evaluate(candidate);
  if (failed(pressure) || pressure->vector > budget) {
    LLVM_DEBUG({
      llvm::dbgs() << "[vector-tile-search] candidate tile=[";
      llvm::interleaveComma(candidate, llvm::dbgs());
      llvm::dbgs() << "] ";
      if (succeeded(pressure))
        llvm::dbgs() << "peak=" << pressure->vector << " (budget=" << budget
                     << ") rejected\n";
      else
        llvm::dbgs() << "evaluate failed\n";
    });
    return failure(); // either analysis failed, or over budget
  }
  return pressure;
}

/// The best candidate tile found so far
struct BestCandidate {
  bool found = false;
  SmallVector<int64_t> tile;
  int64_t traffic = 0;
  int64_t iterations = 0;
  double utilization = 0.0;

  /// Is the other candidate Better?
  /// Check for better traffic first, then fewer loop iterations,
  /// then higher vector register utilization
  /// tile was already checked against the budget
  bool considerIfBetter(ArrayRef<int64_t> candidate,
                        const RegisterPressure &pressure,
                        ArrayRef<TrafficNode> trafficNodes,
                        ArrayRef<int64_t> bound,
                        const llvm::SmallBitVector &isReductionDim) {
    int64_t candidateTraffic =
        trafficOf(trafficNodes, bound, isReductionDim, candidate);
    int64_t candidateIterations = iterationsOf(bound, candidate);
    LLVM_DEBUG({
      llvm::dbgs() << "[vector-tile-search] candidate tile=[";
      llvm::interleaveComma(candidate, llvm::dbgs());
      llvm::dbgs() << "] peak=" << pressure.vector
                   << " traffic=" << candidateTraffic
                   << " iterations=" << candidateIterations << "\n";
    });

    bool better =
        !found || candidateTraffic < traffic ||
        (candidateTraffic == traffic && candidateIterations < iterations) ||
        (candidateTraffic == traffic && candidateIterations == iterations &&
         pressure.utilization() > utilization);
    if (!better)
      return false;

    found = true;
    tile.assign(candidate.begin(), candidate.end());
    traffic = candidateTraffic;
    iterations = candidateIterations;
    utilization = pressure.utilization();
    return true;
  }
};

/// The tile vector over all anchor dims: each entry at the tile its op loop
/// (AnchorDim::op, AnchorDim::loop) gets from that op's plan. The root's
/// entries have no plan here and stay 0
SmallVector<int64_t>
getPinnedTile(const DispatchRegisterGraph &graph,
              ArrayRef<OpComputeTilePlan> nonRootComputeTilePlans) {
  ArrayRef<AnchorDim> anchorDims = graph.getAnchorDims();
  SmallVector<int64_t> tile(anchorDims.size(), 0);
  for (auto [entry, anchorDim] : llvm::enumerate(anchorDims)) {
    const OpComputeTilePlan *plan =
        llvm::find_if(nonRootComputeTilePlans, [&](const OpComputeTilePlan &p) {
          return p.op == anchorDim.op;
        });
    if (plan == nonRootComputeTilePlans.end() ||
        anchorDim.loop >= plan->computeTile.size())
      continue;
    tile[entry] = plan->computeTile[anchorDim.loop].size;
  }
  return tile;
}

} // namespace

LogicalResult
searchVectorTiling(const PlanningContext &context,
                   const DispatchShape &dispatchShape,
                   DispatchStrategy &strategy,
                   ArrayRef<OpComputeTilePlan> nonRootComputeTilePlans,
                   const VectorTileSearchConfig &config) {
  // disabled by command line flag:
  if (!context.options.enableVectorTileSearch)
    return success();
  // Only pipelines that take the root's vector tile from
  // RootTilingPlan::computeTile are in scope; HMX's are left out
  if (strategy.pipeline != IREE::CPU::LoweringPipeline::DoubleTilingExpert)
    return success();

  auto rootOp = dyn_cast_or_null<linalg::LinalgOp>(dispatchShape.root);
  if (!rootOp)
    return success();

  const OpShape &rootShape = getRootShape(dispatchShape);
  unsigned num_root_dims = rootShape.dimensions.size();
  if (strategy.rootTiling.computeTile.size() != num_root_dims)
    return success(); // unexpected shape; leave the heuristic tile alone.

  // helper for recording a Fallback decision
  auto recordFallback = [&](const Twine &reason) {
    context.trace.recordForOp(
        DecisionStage::ComputeTile, DecisionKind::Fallback, rootShape.ordinal,
        dispatchShape.root->getName().getStringRef(),
        Twine("vector tile search: ") + reason + "; kept heuristic tile");
  };

  // Build the Estimator Graph
  DispatchGraphOptions graphOptions;
  graphOptions.config = std::make_shared<EstimatorConfig>();
  std::string failureReason;
  graphOptions.onFailure = [&](StringRef reason) {
    failureReason = reason.str();
  };
  FailureOr<DispatchRegisterGraph> graph =
      DispatchRegisterGraph::build(context.entryPoint, rootOp, graphOptions);
  if (failed(graph)) {
    recordFallback("graph build failed (" + failureReason + ")");
    return success();
  }
  ArrayRef<AnchorDim> anchorDims = graph->getAnchorDims();
  if (anchorDims.size() < num_root_dims) {
    recordFallback("fewer anchor dims than root loops");
    return success();
  }

  // compute search space bounds
  // The VTCM tile bounds the search
  ArrayRef<TileDecision> vtcm =
      strategy.rootTiling.vtcm
          ? ArrayRef<TileDecision>(strategy.rootTiling.vtcm->tileSizes)
          : ArrayRef<TileDecision>();

  SmallVector<int64_t> bound(num_root_dims);
  SmallVector<bool> fixed(num_root_dims, false);
  for (unsigned d = 0; d < num_root_dims; ++d) {
    bound[d] = anchorDims[d].extent;
    if (d < vtcm.size() && vtcm[d].size != 0) {
      bound[d] = vtcm[d].size;
      fixed[d] = vtcm[d].hardwareFixed;
    }
  }

  // The lane dim: the root's own output value's innermost result dim. The
  // graph fixes the loop order and the vectorized dim at build time (report
  // section 2), so this is a property of the anchor, not of the candidate.
  AffineMap outputMap = rootOp.getIndexingMapsArray().back();
  std::optional<unsigned> laneDim = getLaneDim(outputMap);
  Type outputElementType =
      getElementTypeOrSelf(rootOp.getDpsInitOperand(0)->get().getType());
  int64_t lanes = getTypeNativeVectorWidth(context, outputElementType);

  // build the candidate tile shapes
  SmallVector<SmallVector<int64_t>> perDim(num_root_dims);
  int64_t totalCandidates = 1;
  for (unsigned dim = 0; dim < num_root_dims; ++dim) {
    if (fixed[dim] || bound[dim] <= 0) {
      perDim[dim] = {bound[dim]};
    } else {
      bool isLane = laneDim && *laneDim == dim;
      perDim[dim] = candidatesForDim(bound[dim], isLane, lanes);
    }
    totalCandidates *= static_cast<int64_t>(perDim[dim].size());
    if (totalCandidates <= 0 || totalCandidates > kMaxCandidates) {
      recordFallback("search space too large (>" + Twine(kMaxCandidates) +
                     " candidates)");
      return success();
    }
  }

  llvm::SmallBitVector isReductionDim(num_root_dims);
  for (unsigned d = 0; d < num_root_dims; ++d)
    isReductionDim[d] =
        anchorDims[d].iteratorType == utils::IteratorType::reduction;

  // collect the nodes causing memory traffic for later cost estimation
  SmallVector<TrafficNode> trafficNodes;
  for (const Node &node : graph->nodes) {
    if (node.dead)
      continue;
    if (node.kind != NodeKind::Load && node.kind != NodeKind::Store)
      continue;
    unsigned bits = getElementBitWidth(node.elementType);
    if (bits == 0)
      continue;
    llvm::SmallBitVector dims = getDimsOf(node.shapeMap);
    dims.resize(
        num_root_dims); // a dim beyond n is a pinned fused-op loop, always
                        // at its full extent: no multiplicity contribution
                        // either way, so clipping it is safe.
    trafficNodes.push_back({std::move(dims), bits});
  }

  // configuration for fast candidate reject
  int64_t budget =
      context.target.architecturalVectorRegisterCount - config.registerMargin;
  // the tightest cap that applies to this dispatch: the global default,
  // tightened if a costly op is present
  int64_t uMax = config.maxUnrolledVectorOps;
  for (const Node &node : graph->nodes) {
    if (node.dead || !node.op)
      continue;
    auto it = config.maxUnrolledVectorOpsByExpansionOp.find(
        node.op->getName().getStringRef());
    if (it != config.maxUnrolledVectorOpsByExpansionOp.end())
      uMax = std::min(uMax, it->second);
  }

  BestCandidate best;
#ifndef NDEBUG
  int64_t evaluatedCandidates = 0;
#endif

  // the fused loops no root loop reaches are not searched: their entries
  // keep the tile their op's own plan gives them
  SmallVector<int64_t> tile = getPinnedTile(*graph, nonRootComputeTilePlans);
  SmallVector<unsigned> index(num_root_dims, 0);
  // n == 0 means no root dims to enumerate at all, nothing to do
  bool done = num_root_dims == 0;
  unsigned bumped_dim = 0;
  while (!done) {
    // read next candidate
    for (unsigned d = 0; d < num_root_dims; ++d)
      tile[d] = perDim[d][index[d]];
    ArrayRef<int64_t> rootTile =
        ArrayRef<int64_t>(tile).take_front(num_root_dims);

    // compute total tile size (compiler should fuse with above loop itself)
    int64_t volume = 1;
    for (unsigned d = 0; d < num_root_dims; ++d)
      volume *= tile[d];

    // number of vectors to hold a full input tile
    // corresponds to the number of vec instructions needed to express this
    bool withinUnrollCap = lanes > 0 && ceilDiv(volume, lanes) <= uMax;
    // shapes knowing to crash the backend (happend on a matmul with small M
    // and large N for me)
    bool crashShape = matchesKnownHexagonCrashShape(
        rootTile, anchorDims.take_front(num_root_dims), laneDim, lanes);
    if (withinUnrollCap && !crashShape) {
      FailureOr<RegisterPressure> pressure =
          evaluateWithinBudget(*graph, tile, budget);
      LLVM_DEBUG(++evaluatedCandidates);
      if (succeeded(pressure)) {
        // fused loops stay out of the cost metric, since they are not varied
        // for now
        best.considerIfBetter(rootTile, *pressure, trafficNodes, bound,
                              isReductionDim);
      } else {
        // skip other candidates, increasing size will not stay in budget
        unsigned lastIndex = perDim[0].size() - 1;
        if (index[0] < lastIndex)
          index[0] = lastIndex;
        if (isReductionDim.test(0)) {
          // unless reduction in dim0 , in this case, register-cost could
          // de-crease, if full reduction tile is resident at once
          index[0]--;
        }
      }
    } else if (crashShape) {
      LLVM_DEBUG({
        llvm::dbgs() << "[vector-tile-search] candidate tile=[";
        llvm::interleaveComma(tile, llvm::dbgs());
        llvm::dbgs() << "] matches a known Hexagon backend crash shape, "
                        "rejected\n";
      });
    }

    // next candidate
    for (bumped_dim = 0; bumped_dim < index.size(); ++bumped_dim) {
      if (++index[bumped_dim] < perDim[bumped_dim].size())
        break;
      index[bumped_dim] = 0; // overflow: bump next dim
    }
    // all have overflowed
    done = bumped_dim == index.size();
  }

  LLVM_DEBUG(llvm::dbgs() << "[vector-tile-search] evaluated "
                          << evaluatedCandidates << " of " << totalCandidates
                          << " candidates\n");

  if (!best.found) {
    recordFallback("no candidate fit the register budget (<= " + Twine(budget) +
                   ")");
    return success();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[vector-tile-search] selected tile=[";
    llvm::interleaveComma(best.tile, llvm::dbgs());
    llvm::dbgs() << "] traffic=" << best.traffic
                 << " iterations=" << best.iterations
                 << " utilization=" << best.utilization << "\n";
  });

  // move the chosen tile to the plan
  SmallVector<TileDecision> chosen(num_root_dims);
  for (unsigned d = 0; d < num_root_dims; ++d)
    chosen[d] = TileDecision{best.tile[d]};
  context.trace.recordTilePlan(DecisionStage::ComputeTile,
                               DecisionKind::Selected, rootShape.ordinal,
                               dispatchShape.root->getName().getStringRef(),
                               "vector-tile-search", chosen);
  strategy.rootTiling.computeTile = std::move(chosen);
  return success();
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning
