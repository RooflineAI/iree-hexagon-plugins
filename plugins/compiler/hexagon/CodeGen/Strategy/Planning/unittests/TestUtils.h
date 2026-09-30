// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_PLANNING_UNITTESTS_TESTUTILS_H_
#define ROOF_HEXAGON_CODEGEN_PLANNING_UNITTESTS_TESTUTILS_H_

#include "hexagon/CodeGen/Strategy/Planning/DispatchPlanTypes.h"
#include "hexagon/CodeGen/Strategy/Planning/RegisterEstimation/DispatchRegisterGraph.h"
#include "hexagon/CodeGen/Strategy/Planning/RegisterEstimation/EstimatorConfig.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>

// Shared plumbing for unittests
namespace mlir::iree_compiler::hexagon::codegen::planning {
namespace {

inline DialectRegistry getTestDialectRegistry() {
  DialectRegistry registry;
  registry.insert<func::FuncDialect, linalg::LinalgDialect,
                  tensor::TensorDialect, arith::ArithDialect>();
  return registry;
}

static_assert(TargetInfo().nativeVectorBytes == 128,
              "TargetInfo's default nativeVectorBytes changed - every "
              "hand-computed register count in these test files assumes a "
              "128-byte vector register and must be recomputed");
static_assert(HexagonVectorBits == TargetInfo().nativeVectorBytes*8,
              "EstimatorConfig's default vector width changed");

/// Parses `source` and returns the single linalg op it contains. Keeps the
/// owning module alive by returning it alongside the op.
inline std::pair<OwningOpRef<ModuleOp>, SmallVector<linalg::LinalgOp>>
parseLinalgOps(MLIRContext &context, StringRef source) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, &context);
  SmallVector<linalg::LinalgOp> ops;
  if (module)
    module->walk([&](linalg::LinalgOp found) { ops.push_back(found); });
  return {std::move(module), std::move(ops)};
}

/// Parses `source` and returns the last linalg op it contains - the one the
/// per-op tests are about, their IR holding exactly one.
inline std::pair<OwningOpRef<ModuleOp>, linalg::LinalgOp>
parseSingleLinalgOp(MLIRContext &context, StringRef source) {
  auto [module, ops] = parseLinalgOps(context, source);
  return {std::move(module), ops.empty() ? linalg::LinalgOp() : ops.back()};
}

// Each test file supplies one dispatch's MLIR and a set of tile
// candidates; the harness below owns the build-evaluate-and-assert
// boilerplate, so a test file states only (IR, configs).
//
/// One tile candidate to check `DispatchRegisterGraph` against.
struct DispatchConfig {
  std::string name;

  // --- input ---------------------------------------------------------------
  /// Anchor-dim order, 0 = untiled (the dim's full static extent).
  SmallVector<int64_t> tileSizes;

  // --- expected output -----------------------------------------------------
  /// Peak vector registers `evaluate` reports.
  int64_t expectedVector = 0;
  /// When set, `build` or `evaluate` must fail and the reported reason must
  /// contain this text - a phrase from the rejection message.
  std::string expectedFailure;

  std::string note;

  /// Bytes of data `evaluate` reports at the peak (RegisterPressure::
  /// usefulBytes). -1 = not checked.
  int64_t expectedUsefulBytes = -1;
  /// Overrides the default estimator config's policy
  FusedRelayoutChunkPolicy fusedRelayoutChunkPolicy =
      FusedRelayoutChunkPolicy::Fail;
};

/// Prints `DispatchConfig::name` as the gtest instantiation suffix.
inline std::string
dispatchConfigTestName(const testing::TestParamInfo<DispatchConfig> &info) {
  return info.param.name;
}

/// Parses `source` and returns the first function it contains - the dispatch
/// the test is about, its IR holding exactly one.
inline std::pair<OwningOpRef<ModuleOp>, FunctionOpInterface>
parseDispatchFunction(MLIRContext &context, StringRef source) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, &context);
  FunctionOpInterface dispatch;
  if (module) {
    module->walk([&](FunctionOpInterface found) {
      if (!dispatch)
        dispatch = found;
    });
  }
  return {std::move(module), dispatch};
}

/// Marks the op the test passes to `build` as the anchor.
inline constexpr llvm::StringLiteral kAnchorAttrName = "anchor";

/// Builds the graph for `dispatch` and checks one candidate against `config`.
inline void expectDispatchEstimate(FunctionOpInterface dispatch,
                                   const DispatchConfig &config) {
  std::string reason;
  DispatchGraphOptions options;
  options.onFailure = [&reason](StringRef message) {
    if (reason.empty())
      reason = message.str();
  };
  if (config.fusedRelayoutChunkPolicy != FusedRelayoutChunkPolicy::Fail) {
    auto estimatorConfig = std::make_shared<EstimatorConfig>();
    estimatorConfig->fusedRelayoutChunkPolicy = config.fusedRelayoutChunkPolicy;
    options.config = std::move(estimatorConfig);
  }

  SmallVector<linalg::LinalgOp> ops;
  dispatch->walk([&ops](linalg::LinalgOp found) { ops.push_back(found); });

  auto expectReportedFailure = [&](StringRef stage) {
    ASSERT_FALSE(config.expectedFailure.empty())
        << "unexpected " << stage.str() << " failure: " << reason << "; "
        << config.note;
    EXPECT_NE(reason.find(config.expectedFailure), std::string::npos)
        << "expected the reason to mention '" << config.expectedFailure
        << "', got: " << reason;
  };

  // The production caller passes its dispatch root. The test IR says which op
  // that is with an `anchor` unit attribute; a single-op dispatch needs none.
  linalg::LinalgOp anchor;
  if (ops.size() == 1)
    anchor = ops.front();
  for (linalg::LinalgOp op : ops)
    if (op->hasAttr(kAnchorAttrName))
      anchor = op;
  ASSERT_TRUE(anchor) << "mark the dispatch root with {"
                      << kAnchorAttrName.str() << "}; " << config.note;

  FailureOr<DispatchRegisterGraph> graph =
      DispatchRegisterGraph::build(dispatch, anchor, options);
  if (failed(graph)) {
    expectReportedFailure("build");
    return;
  }

  FailureOr<RegisterPressure> pressure = graph->evaluate(config.tileSizes);
  if (failed(pressure)) {
    expectReportedFailure("evaluate");
    return;
  }

  ASSERT_TRUE(config.expectedFailure.empty())
      << "expected a failure mentioning '" << config.expectedFailure
      << "', but the tile was accepted; " << config.note;

  if (config.expectedVector != -1)
    EXPECT_EQ(pressure->vector, config.expectedVector) << config.note;
  if (config.expectedUsefulBytes != -1)
    EXPECT_EQ(pressure->usefulBytes, config.expectedUsefulBytes) << config.note;
}

/// Declares the standard parameterized dispatch-estimation test suite.
/// Usage: `suite` is the test name, `ir` the dispatch's MLIR source as a
/// `R"mlir(...)mlir"` string literal, and the remaining args are the
/// DispatchConfigs to check it against.
///
/// The measurement scripts in ../reference key on this macro's name to find
/// each suite's IR and configs.
#define REGISTER_ESTIMATION_TEST_SUITE(suite, ir, ...)                         \
  class suite##DispatchEstimationTest                                          \
      : public ::testing::TestWithParam<DispatchConfig> {};                    \
  TEST_P(suite##DispatchEstimationTest, MatchesFormula) {                      \
    MLIRContext context(getTestDialectRegistry());                             \
    context.loadAllAvailableDialects();                                        \
    auto [module, dispatch] = parseDispatchFunction(context, ir);              \
    ASSERT_TRUE(dispatch);                                                     \
    expectDispatchEstimate(dispatch, GetParam());                              \
  }                                                                            \
  INSTANTIATE_TEST_SUITE_P(Configs, suite##DispatchEstimationTest,             \
                           ::testing::Values(__VA_ARGS__),                     \
                           dispatchConfigTestName)

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen::planning

#endif // ROOF_HEXAGON_CODEGEN_PLANNING_UNITTESTS_TESTUTILS_H_
