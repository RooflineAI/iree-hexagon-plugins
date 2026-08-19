// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "KernelDispatchHeuristics.h"

#include "cellar-hexagon/CodeGen/Pipelines/TranslationPipeline.h"

#include "hexagon/Conversion/LinalgToLLVM/VTCMTilingOptions.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

#define DEBUG_TYPE "iree-hexagon-kernel-dispatch"

namespace mlir::iree_compiler::cellar_hexagon::codegen {
namespace {

using mlir::FunctionOpInterface;
using mlir::Operation;
using mlir::ShapedType;
using mlir::TilingInterface;
using mlir::Type;
using mlir::TypeSwitch;
using mlir::iree_compiler::IREE::CPU::LoweringPipeline;

namespace linalg = mlir::linalg;

// ===== Utility helpers =====

static int64_t ceilDiv(int64_t lhs, int64_t rhs) {
  return (lhs + rhs - 1) / rhs;
}

// Initialize a plan with zeroes for all tile levels.
// As a small reminder, a tiling of 0 means that no tiling will be performed.
static OpLoweringPlan createZeroPlan(int64_t numLoops) {
  OpLoweringPlan plan;
  plan.tileLevels.distribution.assign(numLoops, 0);
  plan.tileLevels.cacheParallel.assign(numLoops, 0);
  plan.tileLevels.vectorCommonParallel.assign(numLoops, 0);
  plan.tileLevels.vectorReduction.assign(numLoops, 0);
  plan.tileLevels.vectorInnerParallel.assign(numLoops, 0);
  return plan;
}

static int64_t getNativeVectorSizeInBytes(FunctionOpInterface entryPointFn,
                                          const PolicyConfig &policyConfig) {
  auto targetAttr =
      mlir::iree_compiler::IREE::HAL::ExecutableTargetAttr::lookup(
          entryPointFn);
  if (targetAttr) {
    auto configAttr = targetAttr.getConfiguration();
    if (auto cpuFeaturesAttr =
            configAttr.getAs<mlir::StringAttr>("cpu_features")) {
      llvm::StringRef cpuFeatures = cpuFeaturesAttr.getValue();
      if (cpuFeatures.contains("+hvx-length128b")) {
        return 128;
      }
      if (cpuFeatures.contains("+hvx-length64b")) {
        return 64;
      }
    }
    if (auto nativeVectorSize = mlir::iree_compiler::getConfigNativeVectorSize(
            targetAttr.getConfiguration())) {
      return std::max<int64_t>(1, *nativeVectorSize);
    }
  }
  return policyConfig.target.nativeVectorBytes;
}

// Returns the vector width for the given type
static int64_t getTypeNativeVectorWidth(FunctionOpInterface entryPointFn,
                                        const PolicyConfig &policyConfig,
                                        Type type) {
  Type elementType = mlir::getElementTypeOrSelf(type);
  assert(elementType.isIntOrFloat() && "expected a numeric element type");
  int64_t byteWidth =
      mlir::iree_compiler::IREE::Util::getRoundedElementByteWidth(elementType);

  int64_t res =
      getNativeVectorSizeInBytes(entryPointFn, policyConfig) / byteWidth;
  assert(res > 0 && "computed illegal vector width for type");
  return res;
}

// Chooses the largest tile not exceeding `preferred` that divides the static
// loop bound exactly while also respecting the `multipleOf` granularity.
// For dynamic bounds we cannot reason about divisibility, so we keep the
// preferred tile unchanged.
//
// Examples:
// - chooseStaticFactor(128, 64) = 64
// - chooseStaticFactor(100, 64) = 50
// - chooseStaticFactor(96, 64, 8) = 48
// - chooseStaticFactor(dynamic, 64, 8) = 64
static int64_t chooseStaticTilingFactor(int64_t staticBound, int64_t preferred,
                                        int64_t multipleOf = 1) {
  if (preferred <= 0) {
    return 0;
  }
  if (ShapedType::isDynamic(staticBound)) {
    return preferred;
  }
  int64_t start = std::min(staticBound, preferred);
  int64_t step = std::max<int64_t>(1, multipleOf);
  for (int64_t candidate = start - start % step; candidate >= 1;
       candidate -= step) {
    if (staticBound % candidate == 0) {
      return candidate;
    }
  }

  return std::min(preferred, staticBound);
}

// ===== Helpers to identify op types and shapes =====

/// Returns true when a linalg contraction is within the matmul-shaped subset
/// currently modeled by the Hexagon backend.
static bool isSupportedContraction(linalg::LinalgOp linalgOp) {
  // Our only supported contraction currently is a matmul with at most one batch
  // dimension.
  auto contractionDims = linalg::inferContractionDims(linalgOp);
  if (failed(contractionDims)) {
    return false;
  }
  if (contractionDims->m.size() != 1 || contractionDims->n.size() != 1 ||
      contractionDims->k.size() != 1 || contractionDims->batch.size() > 1) {
    return false;
  }
  return true;
}

static bool isBufferLikeCopyOp(linalg::LinalgOp op) {
  return op.hasPureBufferSemantics() && linalg::isaCopyOpInterface(op);
}

static bool is2DConvOp(linalg::LinalgOp op) {
  return linalg::isaConvolutionOpOfType<linalg::Conv2DNhwcHwcfOp>(op) ||
         linalg::isaConvolutionOpOfType<linalg::Conv2DNchwFchwOp>(op);
}

static bool is2DDepthConvOp(linalg::LinalgOp op) {
  return linalg::isaConvolutionOpOfType<linalg::DepthwiseConv2DNhwcHwcOp>(op);
}

static bool is2DPoolingOp(linalg::LinalgOp op) {
  return linalg::isaConvolutionOpOfType<linalg::PoolingNhwcSumOp>(op) ||
         linalg::isaConvolutionOpOfType<linalg::PoolingNhwcMaxOp>(op) ||
         linalg::isaConvolutionOpOfType<linalg::PoolingNhwcMaxUnsignedOp>(op) ||
         linalg::isaConvolutionOpOfType<linalg::PoolingNhwcMinOp>(op) ||
         linalg::isaConvolutionOpOfType<linalg::PoolingNhwcMinUnsignedOp>(op) ||
         linalg::isaConvolutionOpOfType<linalg::PoolingNchwSumOp>(op) ||
         linalg::isaConvolutionOpOfType<linalg::PoolingNchwMaxOp>(op);
}

enum class Conv2DDimOrder { Nchw, Nhwc };

static Conv2DDimOrder getConv2DDimOrder(linalg::LinalgOp op) {
  if (linalg::isaConvolutionOpOfType<linalg::Conv2DNchwFchwOp>(op) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNchwSumOp>(op) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNchwMaxOp>(op)) {
    return Conv2DDimOrder::Nchw;
  }
  if (linalg::isaConvolutionOpOfType<linalg::Conv2DNhwcHwcfOp>(op) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNhwcSumOp>(op) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNhwcMaxOp>(op) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNhwcMaxUnsignedOp>(op) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNhwcMinOp>(op) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNhwcMinUnsignedOp>(op) ||
      linalg::isaConvolutionOpOfType<linalg::DepthwiseConv2DNhwcHwcOp>(op)) {
    return Conv2DDimOrder::Nhwc;
  }
  llvm::report_fatal_error("unsupported conv op");
}

// ===================================================

static std::optional<VTCMTilingPlan> inferVTCMTilingPlan(linalg::LinalgOp op) {
  if (!op.hasPureTensorSemantics()) {
    op.emitWarning()
        << "VTCM tiling is only supported for tensor semantics ops";
    return std::nullopt;
  }

  VTCMTilingPlan plan;

  // TODO: Reimplement this however we want (or keep reusing the one from
  // hexagon-mlir)
  // Right now, we are deferring this tiling decision to hexagon-mlir
  std::optional<SmallVector<int64_t>> tileSizes =
      mlir::hexagon::determineTileSizes(op);
  if (!tileSizes) {
    op.emitWarning() << "Failed to determine VTCM tile sizes for op: " << op;
    return std::nullopt;
  }

  llvm::append_range(plan.tileSizes, *tileSizes);

  return plan;
}

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

static Type getDefaultVectorType(TilingInterface tilingOp) {
  Operation *op = tilingOp.getOperation();
  if (op->getNumResults() > 0) {
    return op->getResult(0).getType();
  }
  if (op->getNumOperands() > 0) {
    return op->getOperand(0).getType();
  }
  return {};
}

static OpLoweringPlan inferInnermostParallelVectorPlan(
    FunctionOpInterface entryPointFn, llvm::ArrayRef<int64_t> staticLoopRanges,
    llvm::ArrayRef<mlir::utils::IteratorType> iteratorTypes, Type vectorType,
    const PolicyConfig &policyConfig, bool tileOuterParallelDimsToOne) {
  OpLoweringPlan plan = createZeroPlan(iteratorTypes.size());
  if (!vectorType) {
    return plan;
  }

  int64_t vectorTile =
      getTypeNativeVectorWidth(entryPointFn, policyConfig, vectorType);
  llvm::SmallVector<int64_t> parallelDims;
  for (auto [idx, iteratorType] : llvm::enumerate(iteratorTypes)) {
    if (iteratorType != mlir::utils::IteratorType::parallel) {
      continue;
    }
    parallelDims.push_back(idx);
    if (tileOuterParallelDimsToOne) {
      plan.tileLevels.vectorCommonParallel[idx] =
          chooseStaticTilingFactor(staticLoopRanges[idx], 1);
    }
  }
  if (!parallelDims.empty()) {
    int64_t innermost = parallelDims.back();
    // Use std::min rather than an exact-divisor search: the
    // CPUDoubleTilingExpert and CPUBufferOpsTileAndVectorize pipelines both
    // enable loop peeling, so a non-divisible innermost tile should be handled
    // cleanly.
    plan.tileLevels.vectorCommonParallel[innermost] =
        ShapedType::isDynamic(staticLoopRanges[innermost])
            ? vectorTile
            : std::min(staticLoopRanges[innermost], vectorTile);
  }
  return plan;
}

// Use only TilingInterface information for the default vector-only tiling
// path. If we cannot infer a representative vector element type, fall back to
// an all-zero plan.
static OpLoweringPlan
inferDefaultTilingInterfacePlan(FunctionOpInterface entryPointFn,
                                TilingInterface tilingOp,
                                const PolicyConfig &policyConfig) {
  Type vectorType = getDefaultVectorType(tilingOp);
  if (!vectorType) {
    return createZeroPlan(tilingOp.getLoopIteratorTypes().size());
  }
  return inferInnermostParallelVectorPlan(
      entryPointFn, getStaticLoopRanges(tilingOp),
      tilingOp.getLoopIteratorTypes(), vectorType, policyConfig,
      /*tileOuterParallelDimsToOne=*/true);
}

static OpLoweringPlan inferBufferCopyPlan(FunctionOpInterface entryPointFn,
                                          TilingInterface tilingOp,
                                          const PolicyConfig &policyConfig) {
  Type vectorType = getDefaultVectorType(tilingOp);
  if (!vectorType) {
    return createZeroPlan(tilingOp.getLoopIteratorTypes().size());
  }

  auto iteratorTypes = tilingOp.getLoopIteratorTypes();
  llvm::SmallVector<int64_t> staticLoopRanges = getStaticLoopRanges(tilingOp);
  OpLoweringPlan plan = createZeroPlan(iteratorTypes.size());
  int64_t vectorTile =
      getTypeNativeVectorWidth(entryPointFn, policyConfig, vectorType);

  llvm::SmallVector<int64_t> parallelDims;
  for (auto [idx, iteratorType] : llvm::enumerate(iteratorTypes)) {
    if (iteratorType != mlir::utils::IteratorType::parallel) {
      continue;
    }
    parallelDims.push_back(idx);
  }
  if (parallelDims.empty()) {
    return plan;
  }

  for (size_t i = 0; i + 1 < parallelDims.size(); ++i) {
    plan.tileLevels.vectorCommonParallel[parallelDims[i]] = 1;
  }

  int64_t innermostParallel = parallelDims.back();
  int64_t staticBound = staticLoopRanges[innermostParallel];
  plan.tileLevels.vectorCommonParallel[innermostParallel] =
      ShapedType::isDynamic(staticBound) ? vectorTile
                                         : std::min(staticBound, vectorTile);

  plan.tileLevels.distribution = plan.tileLevels.vectorCommonParallel;
  return plan;
}

static llvm::SmallVector<int64_t>
getDefaultConvVectorTileSizes(FunctionOpInterface entryPointFn,
                              linalg::LinalgOp op, int64_t vectorSize) {
  bool isSupported = is2DConvOp(op) || is2DDepthConvOp(op) || is2DPoolingOp(op);
  (void)isSupported;
  assert(isSupported && "conv op is not supported");

  if (is2DConvOp(op)) {
    return {1, 1, vectorSize, vectorSize, 1, 1, vectorSize};
  }
  if (is2DDepthConvOp(op)) {
    return {1, 1, vectorSize, vectorSize, 1, vectorSize};
  }
  return {1, 1, vectorSize, vectorSize, 1, vectorSize};
}

// This plan is currently WIP and completely hardcoded for now. It has been
// observed to lower correctly on resnet-50. It has been determined by using the
// plan determined by KernelDispatch.cpp from LLVMCPU for a custom architecture.
static OpLoweringPlan inferConvPlan(FunctionOpInterface entryPointFn,
                                    linalg::LinalgOp op,
                                    const PolicyConfig &policyConfig) {
  Type vectorType =
      getDefaultVectorType(cast<TilingInterface>(op.getOperation()));
  if (!vectorType) {
    return createZeroPlan(op.getNumLoops());
  }

  int64_t vectorSize =
      getTypeNativeVectorWidth(entryPointFn, policyConfig, vectorType);
  llvm::SmallVector<int64_t> targetTileSizes =
      getDefaultConvVectorTileSizes(entryPointFn, op, vectorSize);

  if (getConv2DDimOrder(op) == Conv2DDimOrder::Nchw) {
    llvm::SmallVector<int64_t> permutation;
    if (is2DConvOp(op)) {
      permutation = {0, 3, 1, 2, 6, 4, 5};
    } else if (is2DPoolingOp(op)) {
      permutation = {0, 3, 1, 2, 4, 5};
    } else {
      llvm::report_fatal_error("unsupported NCHW convolution op");
    }
    applyPermutationToVector(targetTileSizes, permutation);
  }

  OpLoweringPlan plan = createZeroPlan(op.getNumLoops());
  llvm::SmallVector<int64_t> staticLoopRanges = op.getStaticLoopRanges();
  auto iteratorTypes = op.getIteratorTypesArray();
  for (auto [idx, iteratorType] : llvm::enumerate(iteratorTypes)) {
    if (iteratorType != mlir::utils::IteratorType::parallel) {
      continue;
    }
    int64_t preferredTile =
        idx < targetTileSizes.size() ? targetTileSizes[idx] : 1;
    plan.tileLevels.vectorCommonParallel[idx] =
        chooseStaticTilingFactor(staticLoopRanges[idx], preferredTile);
    plan.tileLevels.distribution[idx] =
        plan.tileLevels.vectorCommonParallel[idx];
  }
  return plan;
}

// Perform the following tiling:
//   - All batch dimensions are fully tiled at the vector level.
//   - The n dimension is tiled based to the vector width and vector tile
//   cap
//   - The m dimension is tiled based vector register and the estimation of how
//   many of them will be used.
//   - The k dimension is tiled based on the reduction tile cap and n dimension.
static OpLoweringPlan inferContractionPlan(FunctionOpInterface entryPointFn,
                                           linalg::LinalgOp op,
                                           const PolicyConfig &policyConfig) {
  // The contraction dimensions are assumed to have already been checked
  if (!isSupportedContraction(op)) {
    op->emitWarning()
        << "unexpected tiling plan selected, falling back to generic "
           "tiling: "
        << *op;
    return inferDefaultTilingInterfacePlan(
        entryPointFn, cast<TilingInterface>(op.getOperation()), policyConfig);
  }

  auto contractionDims = linalg::inferContractionDims(op).value();
  int64_t numLoops = op.getNumLoops();
  OpLoweringPlan plan = createZeroPlan(numLoops);
  llvm::SmallVector<int64_t> staticLoopRanges = op.getStaticLoopRanges();

  int64_t mDim = contractionDims.m.front();
  int64_t nDim = contractionDims.n.front();
  int64_t kDim = contractionDims.k.front();

  Type accumulatorType = op.getDpsInitOperand(0)->get().getType();
  int64_t accumulatorVectorWidth =
      getTypeNativeVectorWidth(entryPointFn, policyConfig, accumulatorType);
  // For now, nTile equals accumulatorVectorWidth and therefore
  // accumulatorRegsPerRow is always 1. This is a temporary simplification meant
  // to be revisited in the future.
  int64_t nTile = accumulatorVectorWidth;
  int64_t accumulatorRegsPerRow =
      std::max<int64_t>(1, ceilDiv(nTile, accumulatorVectorWidth));
  int64_t mTile = std::max<int64_t>(
      1,
      std::min<int64_t>(policyConfig.caps.contractionMTile,
                        policyConfig.target.usableVectorRegisterCount /
                            std::max<int64_t>(3, 3 * accumulatorRegsPerRow)));
  int64_t kTile = std::min<int64_t>(policyConfig.caps.contractionKTile, nTile);

  // Detect transposed-RHS contractions, i.e. ops where k is the innermost
  // (fastest-varying) dimension of the RHS operand rather than n.
  //
  // Standard layout  B[..., k, n]: vectorizing n loads contiguous RHS elements.
  // Transposed layout B[..., n, k]: vectorizing n requires strided/gather loads
  // (elements are k*sizeof(f32) bytes apart), which is expensive on HVX.
  //
  // For transposed RHS we keep nTile=vector_width (so SIMD vectorization over n
  // is preserved) but also raise kTile to vector_width, bypassing the
  // contractionKTile cap. This gives a 32x32 inner tile: the gather cost is
  // the same as with kTile=8, but each gather result is reused across 32 k
  // iterations instead of 8, improving the arithmetic-to-load ratio.
  bool isTransposedRhs = false;
  {
    auto indexingMaps = op.getIndexingMapsArray();
    for (int64_t i = 0; i < (int64_t)op.getNumDpsInputs(); ++i) {
      AffineMap map = indexingMaps[i];
      // Identify the RHS: the input that contains the n-dim in its map.
      bool hasN = llvm::any_of(map.getResults(), [nDim](AffineExpr e) {
        auto d = mlir::dyn_cast<AffineDimExpr>(e);
        return d && (int64_t)d.getPosition() == nDim;
      });
      if (!hasN)
        continue;
      if (map.getNumResults() > 0) {
        if (auto d = mlir::dyn_cast<AffineDimExpr>(
                map.getResult(map.getNumResults() - 1))) {
          if ((int64_t)d.getPosition() == kDim)
            isTransposedRhs = true;
        }
      }
      break;
    }
  }

  // TODO: This is very questionable, but is meant as a temporary fallback for
  // development. The normal lowering will use outer products and accumulations
  // in order to perform a matrix multiplication of this shape. When the weight
  // matrix is laid out in memory as kxn, tiling over the n dimension yields
  // acceptable performance for now. Nevertheless, in the transposed case this
  // same tiling results in awful performance. Instead, tile to register length
  // over the k dimension. This will not generate properly vectorized code at
  // all for the batched matmul, but at least previous operations in the fused
  // dispatch will be properly vectorized. Possible improvements regarding this
  // issue would be to rearrange the data into a different layout beforehand, or
  // alternatively to switch to a different algorithm (inner product + reduction
  // instead, examples of this are available in the ggml public repository).
  if (isTransposedRhs) {
    mTile = 1;
    nTile = 1;
    kTile = accumulatorVectorWidth; // bypass contractionKTile cap for this case
  }

  // TODO: This is hardcoded for now, revisit later
  for (unsigned batchDim : contractionDims.batch) {
    plan.tileLevels.cacheParallel[batchDim] = 1;
    plan.tileLevels.vectorCommonParallel[batchDim] = 1;
  }
  plan.tileLevels.vectorCommonParallel[mDim] =
      chooseStaticTilingFactor(staticLoopRanges[mDim], mTile);
  // Prefer the full vector width over hunting for an exact divisor of the
  // loop bound. CPUDoubleTilingExpert enables loop peeling for tensor
  // contractions, so the remainder tile is handled without scalar fallback.
  plan.tileLevels.vectorCommonParallel[nDim] =
      ShapedType::isDynamic(staticLoopRanges[nDim])
          ? nTile
          : std::min(staticLoopRanges[nDim], nTile);
  plan.tileLevels.vectorReduction[kDim] =
      chooseStaticTilingFactor(staticLoopRanges[kDim], kTile);

  // TODO: This is hardcoded for now, the objective it to limit temporary
  // buffer sizes when things go wrong to avoid crashes
  plan.tileLevels.cacheParallel[mDim] =
      chooseStaticTilingFactor(staticLoopRanges[mDim], 64);
  plan.tileLevels.cacheParallel[nDim] =
      chooseStaticTilingFactor(staticLoopRanges[nDim], 64);
  plan.tileLevels.cacheParallel[kDim] = 0;

  return plan;
}

// Generic ops' vectorization pattern:
// - prefer vectorization on the innermost parallel dimension
// - keep outer parallel dimensions scalar (tile 1)
// - if a reduction loop exists, vectorize only the innermost reduction.
static OpLoweringPlan inferGenericPlan(FunctionOpInterface entryPointFn,
                                       linalg::GenericOp genericOp,
                                       const PolicyConfig &policyConfig) {
  OpLoweringPlan plan = createZeroPlan(genericOp.getNumLoops());

  llvm::SmallVector<int64_t> staticLoopRanges = genericOp.getStaticLoopRanges();
  auto iteratorTypes = genericOp.getIteratorTypesArray();
  Type resultType = genericOp.getNumDpsInits()
                        ? genericOp.getDpsInitOperand(0)->get().getType()
                        : genericOp->getOperand(0).getType();
  int64_t vectorTile =
      getTypeNativeVectorWidth(entryPointFn, policyConfig, resultType);

  // We collect parallel dimensions from inner-to-outer so we can identify the
  // innermost one. The vector is then reversed back to program order so entries
  // remain ordered from outermost to innermost for the scalar-tile
  // initialization loop below.
  llvm::SmallVector<int64_t> parallelDims;
  int64_t innermostReduction = -1;
  for (int64_t idx = iteratorTypes.size() - 1; idx >= 0; --idx) {
    if (iteratorTypes[idx] == mlir::utils::IteratorType::parallel) {
      parallelDims.push_back(idx);
    } else if (iteratorTypes[idx] == mlir::utils::IteratorType::reduction &&
               innermostReduction < 0) {
      innermostReduction = idx;
    }
  }
  std::reverse(parallelDims.begin(), parallelDims.end());

  // When the innermost loop is a reduction over the contiguous axis and there
  // are at least two outer parallel dims (e.g. softmax over the last dim of a
  // 3-D tensor), vectorizing the parallel dim would require strided/gather
  // loads while the reduction dim is sequential. Redirect the full vector
  // width to the reduction and leave the innermost parallel scalar.
  bool vectorizeReduction =
      !parallelDims.empty() && innermostReduction >= 0 &&
      innermostReduction == (int64_t)(iteratorTypes.size() - 1) &&
      (int64_t)parallelDims.size() >= 2;

  if (!parallelDims.empty()) {
    int64_t innermostParallel = parallelDims.back();
    for (size_t i = 0; i + 1 < parallelDims.size(); ++i) {
      int64_t dim = parallelDims[i];
      plan.tileLevels.vectorCommonParallel[dim] =
          chooseStaticTilingFactor(staticLoopRanges[dim], 1);
      // TODO: This is hardcoded for now, the objective it to limit temporary
      // buffer sizes when things go wrong in other tilings, WIP
      plan.tileLevels.cacheParallel[dim] =
          chooseStaticTilingFactor(staticLoopRanges[dim], 64);
    }
    // Same rationale as inferInnermostParallelVectorPlan: prefer the full
    // vector width and let loop peeling handle any non-divisible remainder.
    plan.tileLevels.vectorCommonParallel[innermostParallel] =
        vectorizeReduction
            ? 1
            : (ShapedType::isDynamic(staticLoopRanges[innermostParallel])
                   ? vectorTile
                   : std::min(staticLoopRanges[innermostParallel], vectorTile));
  }

  if (innermostReduction >= 0) {
    int64_t reductionTile = vectorizeReduction
                                ? vectorTile
                                : policyConfig.caps.genericReductionTile;
    plan.tileLevels.vectorReduction[innermostReduction] =
        chooseStaticTilingFactor(staticLoopRanges[innermostReduction],
                                 reductionTile);
  }

  return plan;
}

} // namespace

bool shouldSetLoweringConfig(Operation *op) {
  if (auto tilingOp = mlir::dyn_cast<TilingInterface>(op)) {
    return !tilingOp.getLoopIteratorTypes().empty();
  }
  return false;
}

std::optional<OpLoweringPlan>
inferOpLoweringPlan(FunctionOpInterface entryPointFn, Operation *op,
                    const PolicyConfig &policyConfig) {
  if (auto linalgOp = mlir::dyn_cast<linalg::LinalgOp>(op)) {
    if (isBufferLikeCopyOp(linalgOp)) {
      return inferBufferCopyPlan(entryPointFn, cast<TilingInterface>(op),
                                 policyConfig);
    }
  }

  return TypeSwitch<Operation *, std::optional<OpLoweringPlan>>(op)
      .Case<linalg::LinalgOp>([&](auto linalgOp) {
        // Classify by structure before op type
        if (is2DConvOp(linalgOp) || is2DDepthConvOp(linalgOp) ||
            is2DPoolingOp(linalgOp)) {
          return inferConvPlan(entryPointFn, linalgOp, policyConfig);
        }
        if (isSupportedContraction(linalgOp)) {
          return inferContractionPlan(entryPointFn, linalgOp, policyConfig);
        }
        if (auto genericOp =
                mlir::dyn_cast<linalg::GenericOp>(linalgOp.getOperation())) {
          return inferGenericPlan(entryPointFn, genericOp, policyConfig);
        }
        return inferDefaultTilingInterfacePlan(
            entryPointFn, cast<TilingInterface>(linalgOp.getOperation()),
            policyConfig);
      })
      .Case<TilingInterface>([&](auto tilingOp) {
        return inferDefaultTilingInterfacePlan(entryPointFn, tilingOp,
                                               policyConfig);
      })
      .Default([&](Operation *) { return std::nullopt; });
}

RootLoweringPlan selectRootLoweringPlan(FunctionOpInterface entryPointFn,
                                        Operation *rootOperation) {
  PolicyConfig policyConfig;
  RootLoweringPlan selection;
  selection.pipeline = LoweringPipeline::Default;

  if (auto linalgOp = mlir::dyn_cast<linalg::LinalgOp>(rootOperation)) {
    if (linalg::isaContractionOpInterface(linalgOp) &&
        !isSupportedContraction(linalgOp)) {
      return selection;
    }
  }

  if (auto opPlan =
          inferOpLoweringPlan(entryPointFn, rootOperation, policyConfig)) {
    selection.opPlan = *opPlan;

    // VTCM tiling is done once for every dispatch, independently of any other
    // choices. This is expected to be a dispatch wide decision orthogonal to
    // other tiling choices for now. Note that this overlaps with the workgroup
    // tiling which is also supposed to be dispatch wide, but we are fixing the
    // number of workgroups to 1 for now.
    auto linalgOp = mlir::dyn_cast<linalg::LinalgOp>(rootOperation);
    if (!linalgOp)
      return selection;

    if (isHexagonVTCMTilingEnabled()) {
      selection.vtcmTiling = inferVTCMTilingPlan(linalgOp);
      if (selection.vtcmTiling) {
        std::fill(selection.opPlan->tileLevels.cacheParallel.begin(),
                  selection.opPlan->tileLevels.cacheParallel.end(), 0);
      }
    }

    // TODO: Revisit when performance becomes relevant.
    // BufferLikeCopies in the context of padding tiled buffers have been
    // observed. They consist of a copy from the original buffer into a new one
    // with additional space for padding. In this context, we would ideally use
    // the DMA without any VTCM tiling whatsoever.
    if (isBufferLikeCopyOp(linalgOp)) {
      selection.pipeline = LoweringPipeline::BufferOpsTileAndVectorize;
    } else if (is2DConvOp(linalgOp) || is2DDepthConvOp(linalgOp) ||
               is2DPoolingOp(linalgOp)) {
      selection.pipeline = LoweringPipeline::ConvTileAndDecomposeExpert;
    } else {
      selection.pipeline = LoweringPipeline::DoubleTilingExpert;
      selection.enableLoopPeeling = linalgOp.hasPureTensorSemantics();
    }
  }
  return selection;
}

} // namespace mlir::iree_compiler::cellar_hexagon::codegen
