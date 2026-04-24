// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/CodeGen/Strategy/KernelDispatch.h"

#include <algorithm>
#include <optional>

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenEnums.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-hexagon-kernel-dispatch"

namespace mlir::iree_compiler::cellar_hexagon::codegen {
namespace {

// This file is in a very experimental state and not properly tested for the
// time being (manually tested on a handful of examples, and not present in
// integration tests).
//
// This file is a Hexagon-owned replacement for the LLVMCPU launch-config
// selection policy in upstream IREE. It deliberately keeps emitting the same
// public IR contract (`translation_info` and `#iree_cpu.lowering_config`) so
// the rest of the Hexagon pipeline can keep reusing LLVMCPU lowering passes.
//
// The main difference from upstream is that this version is much
// simpler: it supports only the op families currently needed by Hexagon
// (contractions, fills, and generic ops) and uses local heuristics instead of
// LLVMCPU's dispatch-wide dimension tracker. This means behavior can diverge
// from LLVMCPU when non-root producer/consumer ops need more precise
// propagation than these local heuristics can infer.
//
// For runnable examples and the current expected behavior, see
// `plugins/compiler/cellar-hexagon/test/codegen/strategy/`
// `hexagon_select_lowering_strategy_study.mlir`.

using mlir::failure;
using mlir::FunctionOpInterface;
using mlir::LogicalResult;
using mlir::Operation;
using mlir::ShapedType;
using mlir::TilingInterface;
using mlir::Type;
using mlir::TypeSwitch;
using mlir::iree_compiler::getComputeOps;
using mlir::iree_compiler::getLoweringConfig;
using mlir::iree_compiler::getRootOperation;
using mlir::iree_compiler::getTranslationInfo;
using mlir::iree_compiler::setLoweringConfig;
using mlir::iree_compiler::setOpConfigAndEntryPointFnTranslation;
using mlir::iree_compiler::setTranslationInfo;
using mlir::iree_compiler::IREE::Codegen::DispatchLoweringPassPipeline;
using mlir::iree_compiler::IREE::CPU::LoweringConfigAttr;

static llvm::cl::opt<int64_t> clHexagonDistributionTileSize(
    "iree-hexagon-distribution-size",
    llvm::cl::desc("Default Hexagon workgroup tile size for parallel loops."),
    llvm::cl::init(64));

static llvm::cl::opt<int64_t> clHexagonGenericVectorTileSize(
    "iree-hexagon-generic-vector-size",
    llvm::cl::desc("Maximum Hexagon vector tile width cap in elements after "
                   "mapping the element type to an HVX-native width."),
    llvm::cl::init(128));

struct RootConfigSelection {
  LoweringConfigAttr config;
  DispatchLoweringPassPipeline pipeline;
  bool usedFallback = false;
};

// TODO: These values are arbitrary for now and need to be studied in depth.
// So far, they have been selected by comparing the output with the LLVMCPU
// kernelDispatch.cpp file and trying to ensure reliability (over speed!).
// In its current state, do note that this only uses a fraction of the
// vector registers and their available bitwidth.
struct PolicyCaps {
  int64_t defaultDistributionTile = clHexagonDistributionTileSize;
  // TODO: These should be queried from the target description rather than
  // hardcoded.
  // HVX has 32 vector registers; 24 is a conservative scratch-space reserve
  // that is not derived from actual register pressure measurement.
  int64_t targetThreadCount = 8;
  int64_t usableVectorRegisterCount = 24;
  int64_t nativeVectorBytes = 128;
  // TODO: 8 f32 elements = 32 bytes = one quarter of a 128-byte HVX vector
  // register. A generic f32 op vectorized at this width is unlikely to emit
  // full-width HVX vector instructions.
  int64_t defaultF32VectorTile = 32;
  int64_t defaultF16VectorTile = 64;
  int64_t defaultI8VectorTile = 128;
  int64_t defaultVectorTile = clHexagonGenericVectorTileSize;
  int64_t defaultReductionTile = 16;
  int64_t defaultGenericReductionTile = 8;
  int64_t defaultContractionMTile = 8;
  // TODO: kTile is element-type-agnostic. For i8 inputs, kTile=8 means 8 bytes
  // per K-iteration, which is 1/16th of a 128-byte HVX register and fails to
  // amortize the memory access overhead.
  int64_t defaultContractionKTile = 8;
};

static bool shouldSetLoweringConfig(Operation *op);
static std::optional<LoweringConfigAttr>
buildConfigForOp(FunctionOpInterface entryPointFn, Operation *op,
                 const PolicyCaps &policyCaps, bool enableGenericDistribution);
static bool isRootOutputGeneric(Operation *rootOperation,
                                mlir::linalg::GenericOp genericOp);
static LoweringConfigAttr buildRootOutputParallelConfig(
    FunctionOpInterface entryPointFn, Operation *op,
    llvm::ArrayRef<int64_t> staticLoopRanges,
    llvm::ArrayRef<mlir::utils::IteratorType> iteratorTypes, Type resultType,
    const PolicyCaps &policyCaps);

// Helper to build `#iree_cpu.lowering_config`.
//
// Upstream LLVMCPU has a richer builder that also handles cache tiling and
// scalable-vector policy. Hexagon currently uses only distribution and vector
// tiling levels, but must still preserve the exact attribute shape that the
// downstream LLVMCPU passes expect.
class LoweringConfigGenerator {
public:
  explicit LoweringConfigGenerator(Operation *op)
      : ctx(op->getContext()), rootOp(op) {}

  void setDistributionTileSizes(llvm::ArrayRef<int64_t> tileSizes) {
    distTileSizes.assign(tileSizes.begin(), tileSizes.end());
  }

  void setVectorTileSizes(llvm::ArrayRef<int64_t> tileSizes) {
    vectorTileSizes.assign(tileSizes.begin(), tileSizes.end());
  }

  void setVectorInnerTileSizes(llvm::ArrayRef<int64_t> tileSizes) {
    vectorInnerTileSizes.assign(tileSizes.begin(), tileSizes.end());
  }

  LoweringConfigAttr generate() const {
    llvm::SmallVector<mlir::NamedAttribute> items;
    if (!distTileSizes.empty() &&
        llvm::any_of(distTileSizes, [](int64_t size) { return size != 0; })) {
      appendLoweringConfigLevelAttr(
          items, mlir::iree_compiler::IREE::CPU::TilingLevel::DistributionTiles,
          distTileSizes);
    }
    if (!vectorTileSizes.empty()) {
      llvm::SmallVector<int64_t> parallelTileSizes(vectorTileSizes);
      llvm::SmallVector<int64_t> reductionTileSizes;
      llvm::SmallVector<bool> parallelScalableFlags(parallelTileSizes.size(),
                                                    false);
      llvm::SmallVector<bool> reductionScalableFlags;
      splitParallelAndReductionTiles(rootOp, parallelTileSizes,
                                     reductionTileSizes, &parallelScalableFlags,
                                     &reductionScalableFlags);
      appendLoweringConfigLevelAttr(items,
                                    mlir::iree_compiler::IREE::CPU::
                                        TilingLevel::VectorCommonParallelTiles,
                                    parallelTileSizes, parallelScalableFlags);
      appendLoweringConfigLevelAttr(
          items,
          mlir::iree_compiler::IREE::CPU::TilingLevel::VectorReductionTiles,
          reductionTileSizes, reductionScalableFlags);
    }
    if (!vectorInnerTileSizes.empty()) {
      // Match LLVMCPU lowering-config construction by materializing explicit
      // non-scalable flags for vector tiling levels. The textual IR is
      // identical either way, but downstream LLVMCPU passes assume the flag
      // array is present and indexed in lock-step with the tile sizes.
      llvm::SmallVector<bool> innerScalableFlags(vectorInnerTileSizes.size(),
                                                 false);
      appendLoweringConfigLevelAttr(
          items,
          mlir::iree_compiler::IREE::CPU::TilingLevel::VectorInnerParallelTiles,
          vectorInnerTileSizes, innerScalableFlags);
    }
    return LoweringConfigAttr::get(ctx, items);
  }

private:
  static unsigned getNumLoops(Operation *op) {
    if (auto tilingOp = mlir::dyn_cast<TilingInterface>(op)) {
      return tilingOp.getLoopIteratorTypes().size();
    }
    return 0;
  }

  static void splitParallelAndReductionTiles(
      Operation *op, llvm::SmallVectorImpl<int64_t> &parallelSizes,
      llvm::SmallVectorImpl<int64_t> &reductionSizes,
      llvm::SmallVectorImpl<bool> *parallelScalableFlags = nullptr,
      llvm::SmallVectorImpl<bool> *reductionScalableFlags = nullptr) {
    reductionSizes.assign(parallelSizes.size(), 0);
    if (parallelScalableFlags) {
      parallelScalableFlags->resize(parallelSizes.size(), false);
    }
    if (reductionScalableFlags) {
      reductionScalableFlags->assign(parallelSizes.size(), false);
    }
    auto tilingOp = mlir::dyn_cast<TilingInterface>(op);
    if (!tilingOp) {
      return;
    }
    for (auto [index, iteratorType] :
         llvm::enumerate(tilingOp.getLoopIteratorTypes())) {
      if (iteratorType == mlir::utils::IteratorType::reduction) {
        reductionSizes[index] = parallelSizes[index];
        parallelSizes[index] = 0;
        if (parallelScalableFlags && reductionScalableFlags) {
          (*reductionScalableFlags)[index] = (*parallelScalableFlags)[index];
          (*parallelScalableFlags)[index] = false;
        }
      }
    }
  }

  void appendLoweringConfigLevelAttr(
      llvm::SmallVectorImpl<mlir::NamedAttribute> &items,
      mlir::iree_compiler::IREE::CPU::TilingLevel level,
      llvm::ArrayRef<int64_t> tileSizes,
      llvm::ArrayRef<bool> scalableFlags = {}) const {
    if (llvm::all_of(tileSizes, [](int64_t size) { return size == 0; })) {
      return;
    }
    items.emplace_back(
        mlir::iree_compiler::IREE::CPU::getTilingLevelName(level),
        LoweringConfigAttr::getTilingLevelAttr(ctx, tileSizes, scalableFlags));
  }

  mlir::MLIRContext *ctx;
  Operation *rootOp;
  llvm::SmallVector<int64_t> distTileSizes;
  llvm::SmallVector<int64_t> vectorTileSizes;
  llvm::SmallVector<int64_t> vectorInnerTileSizes;
};

class MultiLoweringConfigGenerator {
public:
  MultiLoweringConfigGenerator(FunctionOpInterface entryPointFn,
                               Operation *rootOperation,
                               llvm::ArrayRef<Operation *> computeOps)
      : entryPointFn(entryPointFn), rootOperation(rootOperation),
        computeOps(computeOps.begin(), computeOps.end()) {
    initPolicyCapsFromRoot();
  }

  void propagate() {
    for (Operation *op : computeOps) {
      if (op == rootOperation || !shouldSetLoweringConfig(op) ||
          getLoweringConfig(op)) {
        continue;
      }
      // Upstream LLVMCPU uses a global dimension tracker to classify ops such
      // as the final cast after a contraction as "root-output-like". The
      // simplified Hexagon propagation does not have that machinery, so we
      // match the important case directly to avoid changing later tiling
      // behavior for contraction result epilogues.
      auto genericOp = mlir::dyn_cast<mlir::linalg::GenericOp>(op);
      if (genericOp && isRootOutputGeneric(rootOperation, genericOp)) {
        setLoweringConfig(op,
                          buildRootOutputParallelConfig(
                              entryPointFn, op, genericOp.getStaticLoopRanges(),
                              genericOp.getIteratorTypesArray(),
                              genericOp.getDpsInitOperand(0)->get().getType(),
                              policyCaps));
        continue;
      }
      auto config = buildConfigForOp(entryPointFn, op, policyCaps,
                                     /*enableGenericDistribution=*/false);
      if (!config) {
        continue;
      }
      setLoweringConfig(op, *config);
    }
  }

private:
  // Seed the conservative non-root policy from the root op so producer and
  // consumer configs stay roughly aligned with the selected root tiles.
  //
  // Divergence note: upstream LLVMCPU propagates full per-dimension tiling
  // state across the whole dispatch. Here we only inherit coarse maxima from
  // the root config, which is less expressive than the original policy.
  //
  // There is no feedback loop. The upstream CPU propagator adjusts root
  // op tile sizes based on non-root op preferences (e.g. pack ops force aligned
  // distribution tiles) before propagating back. Data tiling is not enabled for
  // Hexagon, and choices regarding this are still up for debate.
  void initPolicyCapsFromRoot() {
    auto rootConfig = getLoweringConfig<LoweringConfigAttr>(rootOperation);
    if (!rootConfig) {
      return;
    }
    if (rootConfig.hasWorkgroupTilingLevel()) {
      int64_t rootDistTile = 0;
      for (int64_t size : rootConfig.getWorkgroupTileSizes()) {
        if (size > 0) {
          rootDistTile = std::max(rootDistTile, size);
        }
      }
      if (rootDistTile > 0) {
        policyCaps.defaultDistributionTile =
            std::min(policyCaps.defaultDistributionTile, rootDistTile);
      }
    }
    if (rootConfig.hasTilingLevel(
            llvm::to_underlying(mlir::iree_compiler::IREE::CPU::TilingLevel::
                                    VectorCommonParallelTiles))) {
      auto attr = mlir::cast<
          mlir::iree_compiler::IREE::Codegen::LoweringConfigTilingLevelAttr>(
          rootConfig.getTilingLevelAttr(
              llvm::to_underlying(mlir::iree_compiler::IREE::CPU::TilingLevel::
                                      VectorCommonParallelTiles)));
      int64_t rootVectorTile = 0;
      for (int64_t size : attr.getSizes()) {
        if (size > 0) {
          rootVectorTile = std::max(rootVectorTile, size);
        }
      }
      if (rootVectorTile > 0) {
        policyCaps.defaultVectorTile =
            std::min(policyCaps.defaultVectorTile, rootVectorTile);
      }
    }
    if (rootConfig.hasTilingLevel(
            llvm::to_underlying(mlir::iree_compiler::IREE::CPU::TilingLevel::
                                    VectorReductionTiles))) {
      auto attr = mlir::cast<
          mlir::iree_compiler::IREE::Codegen::LoweringConfigTilingLevelAttr>(
          rootConfig.getTilingLevelAttr(
              llvm::to_underlying(mlir::iree_compiler::IREE::CPU::TilingLevel::
                                      VectorReductionTiles)));
      int64_t rootReductionTile = 0;
      for (int64_t size : attr.getSizes()) {
        if (size > 0) {
          rootReductionTile = std::max(rootReductionTile, size);
        }
      }
      if (rootReductionTile > 0) {
        policyCaps.defaultReductionTile =
            std::min(policyCaps.defaultReductionTile, rootReductionTile);
      }
    }
  }

  FunctionOpInterface entryPointFn;
  Operation *rootOperation;
  llvm::SmallVector<Operation *> computeOps;
  PolicyCaps policyCaps;
};

// Fetch the effective HVX vector width. The embedded target attribute still
// carries a smaller generic `native_vector_size`, so prefer the explicit HVX
// feature string when it is available.
static int64_t getNativeVectorSizeInBytes(FunctionOpInterface entryPointFn,
                                          const PolicyCaps &policyCaps) {
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
  return policyCaps.nativeVectorBytes;
}

static int64_t ceilDiv(int64_t lhs, int64_t rhs) {
  return (lhs + rhs - 1) / rhs;
}

static int64_t getTypeNativeVectorTile(FunctionOpInterface entryPointFn,
                                       const PolicyCaps &policyCaps,
                                       Type type) {
  Type elementType = mlir::getElementTypeOrSelf(type);
  if (!elementType.isIntOrFloat()) {
    return 1;
  }
  int64_t byteWidth =
      mlir::iree_compiler::IREE::Util::getRoundedElementByteWidth(elementType);
  int64_t naturalWidth = std::max<int64_t>(
      1, getNativeVectorSizeInBytes(entryPointFn, policyCaps) /
             std::max<int64_t>(1, byteWidth));
  if (byteWidth == 1) {
    return std::min<int64_t>(policyCaps.defaultI8VectorTile, naturalWidth);
  }
  if (byteWidth == 2) {
    return std::min<int64_t>(policyCaps.defaultF16VectorTile, naturalWidth);
  }
  if (byteWidth == 4) {
    return std::min<int64_t>(policyCaps.defaultF32VectorTile, naturalWidth);
  }
  return naturalWidth;
}

static int64_t clampToStaticBound(int64_t tileSize, int64_t staticBound) {
  if (tileSize == 0 || ShapedType::isDynamic(staticBound)) {
    return tileSize;
  }
  return std::min(tileSize, staticBound);
}

// Favor factors of the static loop bound so later tiling/vectorization passes
// do not immediately need cleanup for a purely policy-induced remainder.
//
// TODO: (LLM comment) The fallback path (no divisor of staticBound is also a
// multiple of multipleOf) silently returns clampToStaticBound(preferred,
// staticBound), which ignores the multipleOf constraint entirely. In practice
// this causes dist_N < vec_N for f32 matmul: chooseStaticFactor(128, 16,
// multipleOf=32) finds no candidate in [1..16] divisible by 32, so it returns
// 16 even though the N vector tile is 32. The fallback should return a value
// that is still divisible by multipleOf (or signal failure to the caller).
static int64_t chooseStaticFactor(int64_t staticBound, int64_t preferred,
                                  int64_t multipleOf = 1) {
  if (preferred <= 0) {
    return 0;
  }
  if (ShapedType::isDynamic(staticBound)) {
    return preferred;
  }
  int64_t start = std::min(staticBound, preferred);
  int64_t step = std::max<int64_t>(1, multipleOf);
  for (int64_t candidate = start; candidate >= 1; --candidate) {
    if (candidate % step == 0 && staticBound % candidate == 0) {
      return candidate;
    }
  }
  return clampToStaticBound(preferred, staticBound);
}

static int64_t getPreferredVectorTile(FunctionOpInterface entryPointFn,
                                      const PolicyCaps &policyCaps, Type type) {
  return std::max<int64_t>(
      1, std::min<int64_t>(
             policyCaps.defaultVectorTile,
             getTypeNativeVectorTile(entryPointFn, policyCaps, type)));
}

static int64_t chooseThreadAwareDistributionTile(int64_t staticBound,
                                                 int64_t preferredTile,
                                                 const PolicyCaps &policyCaps,
                                                 int64_t multipleOf = 1) {
  if (preferredTile <= 0) {
    return 0;
  }
  if (!ShapedType::isDynamic(staticBound) && staticBound > 0 &&
      policyCaps.targetThreadCount > 0) {
    preferredTile =
        std::min(preferredTile,
                 std::max<int64_t>(
                     1, ceilDiv(staticBound, policyCaps.targetThreadCount)));
  }
  return chooseStaticFactor(staticBound, preferredTile, multipleOf);
}

static FailureOr<mlir::linalg::ContractionDimensions>
getSupportedContractionDims(mlir::linalg::LinalgOp linalgOp) {
  auto contractionDims = mlir::linalg::inferContractionDims(linalgOp);
  if (failed(contractionDims)) {
    return failure();
  }
  if (contractionDims->m.size() != 1 || contractionDims->n.size() != 1 ||
      contractionDims->k.size() != 1 || contractionDims->batch.size() > 1) {
    return failure();
  }
  return contractionDims;
}

static bool isSupportedContraction(mlir::linalg::LinalgOp linalgOp) {
  // Hexagon's dedicated contraction policy is matmul-shaped.
  // Reject dot-like and other generalized contractions here so they reroute to
  // the conservative CPUDefault path until Hexagon grows a dedicated reduction
  // strategy for them.
  return succeeded(getSupportedContractionDims(linalgOp));
}

static bool shouldSetLoweringConfig(Operation *op) {
  if (auto tilingOp = mlir::dyn_cast<TilingInterface>(op)) {
    return !tilingOp.getLoopIteratorTypes().empty();
  }
  return false;
}

// This special case exists to preserve a behavior the simplified propagation
// would otherwise lose: epilogue-style generics that consume the contraction
// root result should tile like the root output/fill path, not like a generic
// producer. Without this, Hexagon can select the same expert pipeline as
// LLVMCPU but still drive it down a different tiling/fusion path.
static bool isRootOutputGeneric(Operation *rootOperation,
                                mlir::linalg::GenericOp genericOp) {
  auto rootLinalgOp =
      mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(rootOperation);
  if (!rootLinalgOp || !isSupportedContraction(rootLinalgOp) ||
      !genericOp.hasPureTensorSemantics()) {
    return false;
  }
  if (!llvm::all_of(genericOp.getIteratorTypesArray(),
                    [](mlir::utils::IteratorType iteratorType) {
                      return iteratorType ==
                             mlir::utils::IteratorType::parallel;
                    })) {
    return false;
  }
  for (mlir::OpOperand *operand : genericOp.getDpsInputOperands()) {
    if (operand->get().getDefiningOp() == rootOperation) {
      return true;
    }
  }
  return false;
}

// Shared config for ops that conceptually write the root/output tile
// (e.g. fills and simple contraction epilogues). This mirrors the upstream
// behavior where these ops usually keep vector-only parallel tiles and do not
// participate in workgroup distribution as separate roots.
static LoweringConfigAttr buildRootOutputParallelConfig(
    FunctionOpInterface entryPointFn, Operation *op,
    llvm::ArrayRef<int64_t> staticLoopRanges,
    llvm::ArrayRef<mlir::utils::IteratorType> iteratorTypes, Type resultType,
    const PolicyCaps &policyCaps) {
  llvm::SmallVector<int64_t> distTileSizes(iteratorTypes.size(), 0);
  llvm::SmallVector<int64_t> vectorTileSizes(iteratorTypes.size(), 1);
  int64_t preferredVectorTile =
      getPreferredVectorTile(entryPointFn, policyCaps, resultType);

  llvm::SmallVector<int64_t> parallelDims;
  for (auto [idx, iteratorType] : llvm::enumerate(iteratorTypes)) {
    if (iteratorType != mlir::utils::IteratorType::parallel) {
      continue;
    }
    parallelDims.push_back(idx);
    vectorTileSizes[idx] = chooseStaticFactor(staticLoopRanges[idx], 1);
  }
  if (parallelDims.size() >= 1) {
    int64_t innermost = parallelDims.back();
    vectorTileSizes[innermost] =
        chooseStaticFactor(staticLoopRanges[innermost], preferredVectorTile);
  }

  LoweringConfigGenerator generator(op);
  generator.setDistributionTileSizes(distTileSizes);
  generator.setVectorTileSizes(vectorTileSizes);
  return generator.generate();
}

// Conservative fallback for ops that Hexagon does not model explicitly yet.
// It prefers legality and predictability over matching LLVMCPU exactly.
//
// Divergence note: if future workloads depend on more precise propagation for
// an unsupported op family, this is the most likely source of behavioral
// drift from upstream.
static LoweringConfigAttr buildFallbackConfig(FunctionOpInterface entryPointFn,
                                              Operation *op,
                                              const PolicyCaps &policyCaps) {
  LoweringConfigGenerator generator(op);
  if (auto linalgOp = mlir::dyn_cast<mlir::linalg::LinalgOp>(op)) {
    llvm::SmallVector<int64_t> distTileSizes(linalgOp.getNumLoops(), 0);
    llvm::SmallVector<int64_t> vectorTileSizes(linalgOp.getNumLoops(), 0);
    auto staticLoopRanges = linalgOp.getStaticLoopRanges();
    auto iteratorTypes = linalgOp.getIteratorTypesArray();
    for (auto [idx, iteratorType] : llvm::enumerate(iteratorTypes)) {
      if (iteratorType != mlir::utils::IteratorType::parallel) {
        continue;
      }
      distTileSizes[idx] = chooseStaticFactor(
          staticLoopRanges[idx], policyCaps.defaultDistributionTile);
    }
    int64_t innermostParallel = -1;
    for (int64_t idx = iteratorTypes.size() - 1; idx >= 0; --idx) {
      if (iteratorTypes[idx] == mlir::utils::IteratorType::parallel) {
        innermostParallel = idx;
        break;
      }
    }
    if (innermostParallel >= 0) {
      Type vectorType = linalgOp.getNumDpsInits()
                            ? linalgOp.getDpsInitOperand(0)->get().getType()
                            : linalgOp->getOperand(0).getType();
      vectorTileSizes[innermostParallel] = chooseStaticFactor(
          staticLoopRanges[innermostParallel],
          getPreferredVectorTile(entryPointFn, policyCaps, vectorType));
    }
    generator.setDistributionTileSizes(distTileSizes);
    generator.setVectorTileSizes(vectorTileSizes);
    return generator.generate();
  }

  if (auto tilingOp = mlir::dyn_cast<TilingInterface>(op)) {
    generator.setDistributionTileSizes(
        llvm::SmallVector<int64_t>(tilingOp.getLoopIteratorTypes().size(), 0));
  }
  return generator.generate();
}

// Hexagon contraction policy: keep batch dimensions conservative, distribute
// only across M/N, and derive vector tiles from the target vector width.
//
// Divergence note: upstream LLVMCPU has additional matmul-specific policy for
// ukernels, mmt4d, cache tiling, pack/unpack interaction, and target-family
// specific shape fixes. None of that is modeled here yet.
static LoweringConfigAttr
buildContractionConfig(FunctionOpInterface entryPointFn,
                       mlir::linalg::LinalgOp op,
                       const PolicyCaps &policyCaps) {
  auto contractionDims = getSupportedContractionDims(op).value();
  int64_t numLoops = op.getNumLoops();
  llvm::SmallVector<int64_t> distTileSizes(numLoops, 0);
  llvm::SmallVector<int64_t> vectorTileSizes(numLoops, 0);
  llvm::SmallVector<int64_t> staticLoopRanges = op.getStaticLoopRanges();

  Type accumulatorType = op.getDpsInitOperand(0)->get().getType();
  int64_t preferredVectorTile =
      getPreferredVectorTile(entryPointFn, policyCaps, accumulatorType);
  int64_t accumulatorVectorWidth = std::max<int64_t>(
      1, getTypeNativeVectorTile(entryPointFn, policyCaps, accumulatorType));
  int64_t nTile = preferredVectorTile;
  int64_t accumulatorRegsPerRow =
      std::max<int64_t>(1, ceilDiv(nTile, accumulatorVectorWidth));
  int64_t mTile = std::max<int64_t>(
      1,
      std::min<int64_t>(policyCaps.defaultContractionMTile,
                        policyCaps.usableVectorRegisterCount /
                            std::max<int64_t>(3, 3 * accumulatorRegsPerRow)));
  // TODO: kTile is not element-type-aware.
  int64_t kTile = std::min<int64_t>(policyCaps.defaultContractionKTile, nTile);

  for (unsigned batchDim : contractionDims.batch) {
    distTileSizes[batchDim] = 1;
    vectorTileSizes[batchDim] = 1;
  }
  int64_t mDim = contractionDims.m.front();
  int64_t nDim = contractionDims.n.front();
  int64_t kDim = contractionDims.k.front();
  distTileSizes[mDim] = chooseThreadAwareDistributionTile(
      staticLoopRanges[mDim], policyCaps.defaultDistributionTile, policyCaps,
      std::max<int64_t>(1, mTile));
  distTileSizes[nDim] = chooseThreadAwareDistributionTile(
      staticLoopRanges[nDim], policyCaps.defaultDistributionTile, policyCaps,
      std::max<int64_t>(1, nTile));
  vectorTileSizes[mDim] = chooseStaticFactor(staticLoopRanges[mDim], mTile);
  vectorTileSizes[nDim] = chooseStaticFactor(staticLoopRanges[nDim], nTile);
  vectorTileSizes[kDim] = chooseStaticFactor(staticLoopRanges[kDim], kTile);

  LoweringConfigGenerator generator(op);
  generator.setDistributionTileSizes(distTileSizes);
  generator.setVectorTileSizes(vectorTileSizes);
  return generator.generate();
}

// Generic-op policy is intentionally shape-driven and local to the op:
// reductions may receive distribution on their parallel dims, while
// elementwise/transposes stay vector-only.
//
// Divergence note: upstream LLVMCPU classifies generic ops with dispatch-wide
// information and can split vector tiles into common vs inner dimensions
// based on all compute ops. This version approximates that behavior from the
// local iterator structure only.
static LoweringConfigAttr buildGenericConfig(FunctionOpInterface entryPointFn,
                                             mlir::linalg::GenericOp genericOp,
                                             const PolicyCaps &policyCaps,
                                             bool enableDistribution) {
  llvm::SmallVector<int64_t> staticLoopRanges = genericOp.getStaticLoopRanges();
  llvm::SmallVector<int64_t> distTileSizes(genericOp.getNumLoops(), 0);
  llvm::SmallVector<int64_t> vectorTileSizes(genericOp.getNumLoops(), 0);
  llvm::SmallVector<int64_t> vectorInnerTileSizes(genericOp.getNumLoops(), 0);
  auto iteratorTypes = genericOp.getIteratorTypesArray();
  // TODO: Think later about whether I want to tile based on input or output
  // operands
  Type resultType = genericOp.getNumDpsInits()
                        ? genericOp.getDpsInitOperand(0)->get().getType()
                        : genericOp->getOperand(0).getType();
  int64_t preferredVectorTile =
      getPreferredVectorTile(entryPointFn, policyCaps, resultType);

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

  if (enableDistribution && innermostReduction >= 0) {
    int64_t reductionDistTile =
        std::max<int64_t>(1, policyCaps.defaultDistributionTile / 2);
    for (int64_t idx : parallelDims) {
      int64_t preferredDist = idx == parallelDims.back()
                                  ? reductionDistTile
                                  : policyCaps.defaultDistributionTile;
      distTileSizes[idx] = chooseThreadAwareDistributionTile(
          staticLoopRanges[idx], preferredDist, policyCaps);
    }
  }

  bool isTranspose = mlir::iree_compiler::isLinalgGeneric2DTranspose(genericOp);
  if (isTranspose && parallelDims.size() >= 2) {
    int64_t secondParallel = parallelDims[parallelDims.size() - 2];
    int64_t innermostParallel = parallelDims.back();
    int64_t transposeTile = std::min<int64_t>(preferredVectorTile, 8);
    vectorTileSizes[secondParallel] =
        chooseStaticFactor(staticLoopRanges[secondParallel], transposeTile);
    vectorTileSizes[innermostParallel] =
        chooseStaticFactor(staticLoopRanges[innermostParallel], transposeTile);
  } else if (parallelDims.size() == 1) {
    int64_t innermostParallel = parallelDims.back();
    vectorTileSizes[innermostParallel] = chooseStaticFactor(
        staticLoopRanges[innermostParallel], preferredVectorTile);
  } else if (!parallelDims.empty()) {
    int64_t innermostParallel = parallelDims.back();
    for (size_t i = 0; i + 1 < parallelDims.size(); ++i) {
      int64_t dim = parallelDims[i];
      vectorTileSizes[dim] = chooseStaticFactor(staticLoopRanges[dim], 1);
    }
    vectorTileSizes[innermostParallel] = chooseStaticFactor(
        staticLoopRanges[innermostParallel], preferredVectorTile);
  }
  if (innermostReduction >= 0) {
    vectorTileSizes[innermostReduction] =
        chooseStaticFactor(staticLoopRanges[innermostReduction],
                           policyCaps.defaultGenericReductionTile);
  }

  LoweringConfigGenerator generator(genericOp);
  generator.setDistributionTileSizes(distTileSizes);
  generator.setVectorTileSizes(vectorTileSizes);
  // Note that this is currently always 0, but applying it is required to avoid
  // crashes later on. Yet again another thing that will probably not be needed
  // when the pipeline is more mature, but we are relying on LLVMCPU passes for
  // now.
  generator.setVectorInnerTileSizes(vectorInnerTileSizes);
  return generator.generate();
}

// Fills use the same parallel-vector policy as root-output-like generics so
// they stay aligned with the destination tiles of the ops they initialize.
static LoweringConfigAttr buildFillConfig(FunctionOpInterface entryPointFn,
                                          mlir::linalg::FillOp fillOp,
                                          const PolicyCaps &policyCaps) {
  return buildRootOutputParallelConfig(
      entryPointFn, fillOp.getOperation(), fillOp.getStaticLoopRanges(),
      fillOp.getIteratorTypesArray(),
      fillOp.getDpsInitOperand(0)->get().getType(), policyCaps);
}

// Compared to upstream LLVMCPU this is still much less global: it selects a
// config from the local op shape plus root-derived caps, instead of
// rebuilding a dispatch-wide tiling solution.
//
// The one intentional root/non-root difference kept here is distribution on
// generic reductions. Only the root path enables it; propagated non-root
// generics stay conservative to avoid changing the downstream tiling/fusion
// behavior.
static std::optional<LoweringConfigAttr>
buildConfigForOp(FunctionOpInterface entryPointFn, Operation *op,
                 const PolicyCaps &policyCaps, bool enableGenericDistribution) {
  return TypeSwitch<Operation *, std::optional<LoweringConfigAttr>>(op)
      .Case<mlir::linalg::FillOp>([&](auto fillOp) {
        return buildFillConfig(entryPointFn, fillOp, policyCaps);
      })
      .Case<mlir::linalg::GenericOp>([&](auto genericOp) {
        return buildGenericConfig(entryPointFn, genericOp, policyCaps,
                                  enableGenericDistribution);
      })
      .Case<mlir::linalg::LinalgOp>([&](auto linalgOp) {
        if (isSupportedContraction(linalgOp)) {
          return buildContractionConfig(entryPointFn, linalgOp, policyCaps);
        }
        return buildFallbackConfig(entryPointFn, linalgOp, policyCaps);
      })
      .Case<TilingInterface>([&](auto tilingOp) {
        return buildFallbackConfig(entryPointFn, tilingOp.getOperation(),
                                   policyCaps);
      })
      .Default([&](Operation *) { return std::nullopt; });
}

// This is mostly used as a fallback in case no other pipeline is selected or
// any failure happens during selection
static LogicalResult
lowerUsingDefaultPipeline(FunctionOpInterface entryPointFn) {
  if (getTranslationInfo(entryPointFn)) {
    return mlir::success();
  }
  auto translationInfo =
      mlir::iree_compiler::IREE::Codegen::TranslationInfoAttr::get(
          entryPointFn.getContext(), DispatchLoweringPassPipeline::CPUDefault);
  return setTranslationInfo(entryPointFn, translationInfo);
}

// Hexagon currently reuses the LLVMCPU expert pipeline names downstream, but
// the policy for choosing/configuring those pipelines lives here.
//
// Divergence note: the pipeline enum may match upstream while the chosen
// tiling config still differs. When debugging mismatches, compare
// `lowering_config` first; matching `translation_info` alone is not enough.
static RootConfigSelection selectRootConfig(FunctionOpInterface entryPointFn,
                                            Operation *rootOperation) {
  PolicyCaps policyCaps;
  RootConfigSelection selection;
  selection.pipeline = DispatchLoweringPassPipeline::CPUDefault;
  selection.usedFallback = true;

  if (auto linalgOp = mlir::dyn_cast<mlir::linalg::LinalgOp>(rootOperation)) {
    if (mlir::linalg::isaContractionOpInterface(linalgOp) &&
        !isSupportedContraction(linalgOp)) {
      return selection;
    }
  }

  if (auto config = buildConfigForOp(entryPointFn, rootOperation, policyCaps,
                                     /*enableGenericDistribution=*/true)) {
    selection.config = *config;
    selection.usedFallback = false;

    // The expert pipeline expects a root lowering config with a workgroup
    // tiling level. Keep pure vector-only fills/broadcasts on CPUDefault; they
    // are valid roots, but they are not expert roots.
    bool hasNonZeroWorkgroupTiling =
        selection.config.hasWorkgroupTilingLevel() &&
        llvm::any_of(selection.config.getWorkgroupTileSizes(),
                     [](int64_t tileSize) { return tileSize != 0; });
    if (auto linalgOp = mlir::dyn_cast<mlir::linalg::LinalgOp>(rootOperation)) {
      if (isSupportedContraction(linalgOp) || hasNonZeroWorkgroupTiling) {
        selection.pipeline =
            DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
      }
    }
  }
  return selection;
}

static LogicalResult
setTranslationInfoAndRootConfig(FunctionOpInterface entryPointFn,
                                llvm::ArrayRef<Operation *> computeOps) {
  // Phase 1: pick a root and assign its translation/config attributes.
  for (Operation *computeOp : computeOps) {
    if (getLoweringConfig(computeOp)) {
      return failure();
    }
  }

  auto rootOp = getRootOperation(computeOps);
  if (mlir::failed(rootOp)) {
    return failure();
  }
  Operation *rootOperation = rootOp.value();
  if (!rootOperation) {
    return lowerUsingDefaultPipeline(entryPointFn);
  }

  // This is where the custom selection for hexagon kicks in
  RootConfigSelection selection = selectRootConfig(entryPointFn, rootOperation);
  if (!selection.config) {
    return lowerUsingDefaultPipeline(entryPointFn);
  }

  if (mlir::failed(setOpConfigAndEntryPointFnTranslation(
          entryPointFn, rootOperation, selection.config, selection.pipeline))) {
    return failure();
  }

  // Phase 2: propagate simplified non-root configs so later IREE tiling passes
  // see a coherent dispatch, even though the propagation policy is lighter than
  // upstream LLVMCPU's dispatch-wide solver.
  MultiLoweringConfigGenerator(entryPointFn, rootOperation, computeOps)
      .propagate();
  return mlir::success();
}

} // namespace

// Top-level entry point used by the Hexagon lowering pipeline.
// After config selection we run the standard ranked-shape cleanup used by the
// upstream launch-config code so later passes see resolved result dimensions.
LogicalResult initHexagonLaunchConfig(FunctionOpInterface funcOp) {
  if (getTranslationInfo(funcOp)) {
    return mlir::success();
  }

  if (funcOp.empty() || !llvm::hasSingleElement(funcOp.getFunctionBody())) {
    return lowerUsingDefaultPipeline(funcOp);
  }

  llvm::SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  if (computeOps.empty()) {
    return lowerUsingDefaultPipeline(funcOp);
  }
  if (mlir::failed(setTranslationInfoAndRootConfig(funcOp, computeOps))) {
    return failure();
  }

  mlir::RewritePatternSet patterns(funcOp.getContext());
  mlir::memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  return mlir::applyPatternsGreedily(funcOp, std::move(patterns));
}

} // namespace mlir::iree_compiler::cellar_hexagon::codegen
