// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/IR/HexagonOps.h"
#include "hexagon/CodeGen/IR/HmxContracts.h"
#include "hexagon/CodeGen/Passes.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerDivisibilityAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONVERIFYHMXRUNTIMEABIPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

// TODO: Think about a better way to setup these constants other than hardcoded
// values.
// The kernels dereference f16 rows as 64-byte half vectors and f32 rows as
// 128-byte HVX vectors. Packed and config buffers go through HMX instructions,
// whose base-address requirement is `IREE::Hexagon::kHmxAlignment`.
constexpr int64_t kHalfVectorAlignment = 64;
constexpr int64_t kHvxAlignment = 128;

// These checks verify the ABI between bufferized HMX IR and the DSP runtime
// kernels. They are required for correctness, not merely optimization hints:
// the kernels use VTCM-only HMX/HVX instructions, issue aligned vector loads,
// and interpret packed buffers using fixed tile shapes and implicit strides.
// They run before the conversion to LLVM, which passes the buffers to the
// kernels as bare pointers: there, the element type, memory space, layout, and
// alignment information would no longer be available.

bool isVtcm(MemRefType type) { return type.getMemorySpaceAsInt() == 1; }

bool hasUnitInnerStride(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(type.getStridesAndOffset(strides, offset)) &&
         !strides.empty() && strides.back() == 1;
}

// The operation verifiers already establish that every buffer this is applied
// to is rank-2, so only the stride relationship is checked here.
bool hasRowMajorLayout(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      strides.back() != 1) {
    return false;
  }
  int64_t rowStride = strides.front();
  int64_t columns = type.getDimSize(1);
  return ShapedType::isDynamic(rowStride) || ShapedType::isDynamic(columns) ||
         rowStride >= columns;
}

// A rank-5 tile grid with an identity layout satisfies this by construction, so
// callers holding a grid should prefer `MemRefLayoutAttrInterface::isIdentity`.
// It is the rank-3 tile carved out of a grid by tiling that needs the explicit
// stride check, because its layout is non-identity but still addressable.
bool hasContiguousTileSuffixLayout(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset))) {
    return false;
  }
  return ArrayRef<int64_t>(strides).take_back(IREE::Hexagon::kHmxTileRank) ==
         ArrayRef<int64_t>(IREE::Hexagon::kHmxTileStrides);
}

LogicalResult verifyRuntimeBufferPlacementAndAlignment(
    Operation *op, Value value, StringRef role, int64_t requiredAlignment,
    DataFlowSolver &solver) {
  // The operation verifiers establish the buffer type, shape, and element type.
  // This pass only checks requirements imposed by the current runtime lowering.
  auto type = cast<MemRefType>(value.getType());
  if (!isVtcm(type)) {
    op->emitOpError() << role << " must be in VTCM memory space 1";
    return failure();
  }
  if (!IREE::Hexagon::providesHmxAlignment(value, requiredAlignment, solver)) {
    op->emitOpError() << role << " must be known to be " << requiredAlignment
                      << "-byte aligned; use an aligned allocation or "
                         "memref.assume_alignment";
    return failure();
  }
  return success();
}

LogicalResult verifyDynamicDimensionFits(Operation *op, Value value,
                                         int64_t dim, int64_t maximum,
                                         StringRef role) {
  auto type = cast<ShapedType>(value.getType());
  if (!type.isDynamicDim(dim)) {
    return success();
  }
  FailureOr<int64_t> upperBound =
      ValueBoundsConstraintSet::computeConstantBound(
          presburger::BoundType::UB, {value, dim}, /*stopCondition=*/nullptr,
          ValueBoundsOptions{/*closedUB=*/true});
  if (failed(upperBound)) {
    return op->emitOpError() << "could not prove an upper bound for " << role
                             << " dimension " << dim;
  }
  if (*upperBound > maximum) {
    return op->emitOpError()
           << role << " dimension " << dim << " has upper bound " << *upperBound
           << ", exceeding the supported maximum " << maximum;
  }
  return success();
}

LogicalResult verifyDynamicGridCoverage(Operation *op, Value grid, Value matrix,
                                        bool transpose, StringRef role) {
  auto gridType = cast<MemRefType>(grid.getType());
  // The op verifier guarantees positive static physical grid dimensions.
  int64_t rowCapacity =
      IREE::Hexagon::getHmxElementCapacity(gridType.getDimSize(0));
  int64_t columnCapacity =
      IREE::Hexagon::getHmxElementCapacity(gridType.getDimSize(1));
  if (failed(verifyDynamicDimensionFits(op, matrix, transpose ? 1 : 0,
                                        rowCapacity, role)) ||
      failed(verifyDynamicDimensionFits(op, matrix, transpose ? 0 : 1,
                                        columnCapacity, role))) {
    return failure();
  }
  return success();
}

LogicalResult verifyPackRuntimePreconditions(IREE::Hexagon::HmxPackOp op,
                                             DataFlowSolver &solver) {
  if (failed(verifyRuntimeBufferPlacementAndAlignment(
          op, op.getSource(), "source", kHalfVectorAlignment, solver)) ||
      failed(verifyRuntimeBufferPlacementAndAlignment(
          op, op.getDest(), "destination", IREE::Hexagon::kHmxAlignment,
          solver))) {
    return failure();
  }
  auto sourceType = cast<MemRefType>(op.getSource().getType());
  auto destType = cast<MemRefType>(op.getDest().getType());
  if (!hasRowMajorLayout(sourceType)) {
    return op.emitOpError(
               "source must have a non-overlapping row-major layout"),
           failure();
  }
  if (!destType.getLayout().isIdentity()) {
    return op.emitOpError(
               "destination must be a contiguous [..., 16, 32, 2] tile grid"),
           failure();
  }
  return verifyDynamicGridCoverage(op, op.getDest(), op.getSource(),
                                   op.getDim() == 1, "source");
}

LogicalResult verifyUnpackRuntimePreconditions(IREE::Hexagon::HmxUnpackOp op,
                                               DataFlowSolver &solver) {
  auto sourceType = cast<MemRefType>(op.getSource().getType());
  auto destType = cast<MemRefType>(op.getDest().getType());
  int64_t destAlignment =
      destType.getElementType().isF16() ? kHalfVectorAlignment : kHvxAlignment;
  if (failed(verifyRuntimeBufferPlacementAndAlignment(
          op, op.getSource(), "source", IREE::Hexagon::kHmxAlignment,
          solver)) ||
      failed(verifyRuntimeBufferPlacementAndAlignment(
          op, op.getDest(), "destination", destAlignment, solver))) {
    return failure();
  }
  // A grid must be fully contiguous, which for the verified [..., 16, 32, 2]
  // shape already implies the per-tile strides. A rank-3 tile is a subview of
  // one, so only its innermost strides can be required.
  bool sourceIsGrid = IREE::Hexagon::isHmxTileGrid(sourceType);
  if (sourceIsGrid ? !sourceType.getLayout().isIdentity()
                   : !hasContiguousTileSuffixLayout(sourceType)) {
    return op.emitOpError(
               "source must use a contiguous [..., 16, 32, 2] tile layout"),
           failure();
  }
  if (!hasRowMajorLayout(destType)) {
    return op.emitOpError(
               "destination must have a non-overlapping row-major layout"),
           failure();
  }
  if (sourceIsGrid) {
    return verifyDynamicGridCoverage(op, op.getSource(), op.getDest(), false,
                                     "source");
  }
  for (unsigned dim : {0u, 1u}) {
    if (failed(verifyDynamicDimensionFits(op, op.getDest(), dim,
                                          IREE::Hexagon::kHmxLogicalTileSize,
                                          "rank-3 unpack destination"))) {
      return failure();
    }
  }
  return success();
}

LogicalResult
verifyAccSetupReadRuntimePreconditions(IREE::Hexagon::HmxAccSetupReadOp op,
                                       DataFlowSolver &solver) {
  if (failed(verifyRuntimeBufferPlacementAndAlignment(
          op, op.getConfig(), "config", IREE::Hexagon::kHmxAlignment,
          solver))) {
    return failure();
  }
  auto type = cast<MemRefType>(op.getConfig().getType());
  if (!hasUnitInnerStride(type)) {
    return op.emitOpError("config must be contiguous"), failure();
  }
  return success();
}

LogicalResult verifyMmaRuntimePreconditions(IREE::Hexagon::HmxMmaOp op,
                                            DataFlowSolver &solver) {
  SmallVector<StringRef> roles = {"lhs", "rhs"};
  for (auto [value, role] :
       llvm::zip_equal(ValueRange{op.getLhs(), op.getRhs()}, roles)) {
    auto type = cast<MemRefType>(value.getType());
    if (failed(verifyRuntimeBufferPlacementAndAlignment(
            op, value, role, IREE::Hexagon::kHmxAlignment, solver))) {
      return failure();
    }
    if (!hasContiguousTileSuffixLayout(type)) {
      return op.emitOpError()
                 << role << " must be one contiguous [16, 32, 2] HMX tile",
             failure();
    }
  }
  return success();
}

LogicalResult verifyAccReadRuntimePreconditions(IREE::Hexagon::HmxAccReadOp op,
                                                DataFlowSolver &solver) {
  if (failed(verifyRuntimeBufferPlacementAndAlignment(
          op, op.getDest(), "destination", IREE::Hexagon::kHmxAlignment,
          solver))) {
    return failure();
  }
  auto type = cast<MemRefType>(op.getDest().getType());
  if (!hasContiguousTileSuffixLayout(type)) {
    return op.emitOpError(
               "destination must be a contiguous [16, 32, 2] HMX tile"),
           failure();
  }
  return success();
}

LogicalResult verifyHmxRuntimeBufferPreconditions(FunctionOpInterface funcOp,
                                                  DataFlowSolver &solver) {
  LogicalResult result = success();
  funcOp.walk([&](Operation *op) {
    LogicalResult opResult =
        llvm::TypeSwitch<Operation *, LogicalResult>(op)
            .Case<IREE::Hexagon::HmxPackOp>([&](auto op) {
              return verifyPackRuntimePreconditions(op, solver);
            })
            .Case<IREE::Hexagon::HmxUnpackOp>([&](auto op) {
              return verifyUnpackRuntimePreconditions(op, solver);
            })
            .Case<IREE::Hexagon::HmxAccSetupReadOp>([&](auto op) {
              return verifyAccSetupReadRuntimePreconditions(op, solver);
            })
            .Case<IREE::Hexagon::HmxMmaOp>([&](auto op) {
              return verifyMmaRuntimePreconditions(op, solver);
            })
            .Case<IREE::Hexagon::HmxAccReadOp>([&](auto op) {
              return verifyAccReadRuntimePreconditions(op, solver);
            })
            .Default([](Operation *) { return success(); });
    if (failed(opResult)) {
      result = failure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

LogicalResult
verifyHmxRuntimeLoweringPreconditions(FunctionOpInterface funcOp) {
  // TODO: I have triggered many bugs between the codegen and the expected
  // shape from the microkernels that were difficult to debug. Therefore, I
  // have added verification between the expected input and the
  // generated one (note that the actual logic of this file is minimal
  // compared to the verification process). I believe this should be
  // generalized to more microkernels and standardized instead of being an
  // ad-hoc check in this file.
  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<dataflow::IntegerDivisibilityAnalysis>();
  if (failed(solver.initializeAndRun(funcOp))) {
    funcOp.emitError("failed to analyze HMX alignment divisibility");
    return failure();
  }
  return verifyHmxRuntimeBufferPreconditions(funcOp, solver);
}

LogicalResult verifyHmxRuntimeABI(ModuleOp moduleOp) {
  for (func::FuncOp funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (failed(verifyHmxRuntimeLoweringPreconditions(funcOp))) {
      return failure();
    }
  }
  return success();
}

struct HexagonVerifyHmxRuntimeABIPass final
    : public impl::HexagonVerifyHmxRuntimeABIPassBase<
          HexagonVerifyHmxRuntimeABIPass> {
  void runOnOperation() override {
    if (failed(verifyHmxRuntimeABI(getOperation()))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::hexagon::codegen
