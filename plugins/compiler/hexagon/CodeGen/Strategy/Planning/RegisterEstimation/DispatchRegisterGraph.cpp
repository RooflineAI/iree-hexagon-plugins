// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "DispatchRegisterGraph.h"

#include "EstimatorConfig.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

#define DEBUG_TYPE "hexagon-dispatch-register-graph"

namespace mlir::iree_compiler::hexagon::codegen::planning {

/// helper for dumping state during debugging
StringRef stringifyNodeKind(NodeKind kind) {
  switch (kind) {
  case NodeKind::BodyOp:
    return "body-op";
  case NodeKind::Load:
    return "load";
  case NodeKind::Store:
    return "store";
  case NodeKind::Invariant:
    return "invariant";
  case NodeKind::Shuffle:
    return "shuffle";
  case NodeKind::Accumulator:
    return "accumulator";
  case NodeKind::MultiplyAccumulate:
    return "multiply_accumulate";
  case NodeKind::Reduced:
    return "reduced";
  }
  return "unknown";
}

/// helper to build a failure result with a given reason
LogicalResult DispatchRegisterGraph::fail(const Twine &reason) const {
  LLVM_DEBUG(llvm::dbgs() << "[dispatch-register-graph] " << reason << "\n");
  if (options.onFailure)
    options.onFailure(reason.str());
  return failure();
}

// if a tile splits up a reduction
bool DispatchRegisterGraph::isReductionTiled(
    ArrayRef<int64_t> resolvedTile) const {
  for (auto [size, dim] : llvm::zip_equal(resolvedTile, anchorDims))
    if (dim.iteratorType == utils::IteratorType::reduction && size < dim.extent)
      return true;
  return false;
}

bool DispatchRegisterGraph::chunksAFusedRelayout(
    ArrayRef<int64_t> resolvedTile) const {
  for (const Node &node : nodes) {
    if (node.dead || !node.fullyMaterialized || node.kind == NodeKind::Load)
      continue;
    for (unsigned d = 0, e = anchorDims.size(); d < e; ++d) {
      if (anchorDims[d].iteratorType != utils::IteratorType::reduction ||
          !node.shapeMap.isFunctionOfDim(d))
        continue;
      if (resolvedTile[d] < anchorDims[d].extent)
        return true;
    }
  }
  return false;
}

FailureOr<DispatchRegisterGraph>
DispatchRegisterGraph::build(FunctionOpInterface dispatch,
                             linalg::LinalgOp anchor,
                             DispatchGraphOptions options) {
  DispatchRegisterGraph graph;
  graph.dispatch = dispatch;
  graph.anchor = anchor;
  graph.config =
      options.config ? options.config : std::make_shared<EstimatorConfig>();
  graph.options = std::move(options);
  graph.options.config = graph.config;

  if (!dispatch)
    return graph.fail("no dispatch function");
  if (failed(buildNodeGraph(graph)) || failed(buildSchedule(graph)))
    return failure();

  return graph;
}

NodeWeights DispatchRegisterGraph::weighNodes(ArrayRef<int64_t> tile) const {
  int64_t vectorBits = getConfig().vectorBits;
  NodeWeights weights;
  weights.registers.reserve(nodes.size());
  weights.usefulBits.reserve(nodes.size());
  // A phase that runs over several reduction tiles keeps its invariant
  // broadcasts materialized across its trips; one that runs once streams them.
  bool reductionTiled = isReductionTiled(tile);
  for (const Node &node : nodes) {
    const Footprint &footprint =
        node.broadcast && reductionTiled ? *node.broadcast : node.footprint;
    weights.registers.push_back(footprint.registers(tile, vectorBits));
    weights.usefulBits.push_back(footprint.usefulBits(tile, vectorBits));
  }

  // A horizontal reduction whose tile does not cover the whole reduction
  // extent keeps every step's partial result alive across loop iterations
  if (reduction && reduction->horizontal && reductionTiled) {
    int64_t &registers = weights.registers[reduction->accumulator];
    int64_t &usefulBits = weights.usefulBits[reduction->accumulator];
    for (NodeIdx operand : reduction->wideningOperands) {
      registers = std::max(registers, weights.registers[operand]);
      usefulBits = std::max(usefulBits, weights.usefulBits[operand]);
    }
  }
  return weights;
}

namespace {
/// The peak of the live set over the schedule, and where it is reached.
struct Peak {
  RegisterPressure pressure;
  StepIdx position = 0;
};
} // namespace

/// Liveliness sweep over the nodes in the graph
static Peak sweepPeak(const DispatchRegisterGraph &graph,
                      ArrayRef<int64_t> tile, const NodeWeights &weights) {
  auto sum = [](ArrayRef<int64_t> perNode, ArrayRef<NodeIdx> values) {
    int64_t total = 0;
    for (NodeIdx value : values)
      total += perNode[value];
    return total;
  };

  int64_t vectorBits = graph.getConfig().vectorBits;
  int64_t peakRegisters = 0;
  int64_t peakBits = 0;
  StepIdx peakPosition = 0;
  int64_t live = 0;
  int64_t liveBits = 0;
  for (auto [position, step] : llvm::enumerate(graph.steps)) {
    live += sum(weights.registers, step.starts);
    liveBits += sum(weights.usefulBits, step.starts);
    const Node &node = graph.nodes[step.node];
    // registers of the input being re-used, if their values are consumed by
    // this step
    int64_t reused = std::min(sum(weights.registers, step.reusable),
                              weights.registers[step.node]);
    int64_t reusedBits = std::min(sum(weights.usefulBits, step.reusable),
                                  weights.usefulBits[step.node]);
    int64_t at = live - reused + node.temporaries.registers(tile, vectorBits);
    if (at > peakRegisters) {
      peakRegisters = at;
      peakBits =
          liveBits - reusedBits + node.temporaries.usefulBits(tile, vectorBits);
      peakPosition = position;
    }
    live -= sum(weights.registers, step.ends);
    liveBits -= sum(weights.usefulBits, step.ends);
  }
  assert(peakBits <= peakRegisters * vectorBits &&
         "the peak holds more data than its registers fit");
  return Peak{RegisterPressure{peakRegisters,
                               llvm::divideCeilSigned(peakBits, int64_t{8}),
                               vectorBits / 8},
              peakPosition};
}

FailureOr<RegisterPressure>
DispatchRegisterGraph::evaluate(ArrayRef<int64_t> tileSizes) const {

  if (tileSizes.size() != anchorDims.size())
    return fail("tile vector has " + Twine(tileSizes.size()) +
                " entries but the anchor has " + Twine(anchorDims.size()) +
                " dims");

  // resolve tile: replace 0 with the full extend
  SmallVector<int64_t> resolved(tileSizes);
  for (auto [size, dim] : llvm::zip_equal(resolved, anchorDims)) {
    if (size < 0 || size > dim.extent)
      return fail("tile size " + Twine(size) + " outside [0, " +
                  Twine(dim.extent) + "]");
    if (size == 0)
      size = dim.extent;
  }

  if (getConfig().fusedRelayoutChunkPolicy == FusedRelayoutChunkPolicy::Fail &&
      chunksAFusedRelayout(resolved))
    return fail("chunks a fused producer's materialized reduction dim "
                "(unpriced per-step relayout cost)");

  // compute weight of the nodes with the given tile zize
  NodeWeights weights = weighNodes(resolved);
  // liveliness sweep to find peak node
  Peak peak = sweepPeak(*this, resolved, weights);

  LLVM_DEBUG(
      dump(llvm::errs(), tileSizes, weights, peak.position, peak.pressure));

  return peak.pressure;
}

ArrayRef<AnchorDim> DispatchRegisterGraph::getAnchorDims() const {
  return anchorDims;
}

ArrayRef<linalg::LinalgOp> DispatchRegisterGraph::getOps() const { return ops; }

// helper for dumping internal state for debugging
void DispatchRegisterGraph::dump(llvm::raw_ostream &os,
                                 const ArrayRef<int64_t> tileSizes,
                                 const NodeWeights &weights,
                                 StepIdx peak_position,
                                 const RegisterPressure &pressure) const {

  os << "dispatch-register-graph: " << ops.size() << " op(s), " << numPhases
     << " phase(s), tile [";
  llvm::interleaveComma(tileSizes, os);
  os << "]\n  peak vector=" << pressure.vector
     << " useful=" << pressure.usefulBytes
     << "B utilization=" << llvm::format("%.2f", pressure.utilization())
     << "\n";
  for (auto [position, step] : llvm::enumerate(steps)) {
    const Node &node = nodes[step.node];
    const LiveInterval &interval = live[step.node];
    os << "  #" << position << " " << stringifyNodeKind(node.kind) << " ";
    if (node.op)
      os << node.op->getName().getStringRef() << " ";
    os << stringifyRegClass(node.regClass) << " map " << node.shapeMap
       << " weight=" << weights.registers[step.node] << " useful="
       << llvm::divideCeilSigned(weights.usefulBits[step.node], int64_t{8})
       << "B live=[" << interval.start << "," << interval.end << "]";
    if (interval.start <= peak_position && peak_position <= interval.end)
      os << " *peak*";
    os << "\n";
  }
}

} // namespace mlir::iree_compiler::hexagon::codegen::planning
