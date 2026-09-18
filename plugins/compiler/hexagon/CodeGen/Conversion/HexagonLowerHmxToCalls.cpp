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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONLOWERHMXTOCALLSPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kHmxAccClearFn = "iree_hexagon_hmx_acc_clear_f16";
constexpr llvm::StringLiteral kHmxAccSetupReadFn =
    "iree_hexagon_hmx_acc_setup_read_f16";
constexpr llvm::StringLiteral kHmxPackFn = "iree_hexagon_hmx_pack_f16";
constexpr llvm::StringLiteral kHmxPackTransposedFn =
    "iree_hexagon_hmx_pack_transposed_f16";
constexpr llvm::StringLiteral kHmxUnpackFn =
    "iree_hexagon_hmx_unpack_acc_f16_to_f32";
constexpr llvm::StringLiteral kHmxUnpackF16Fn =
    "iree_hexagon_hmx_unpack_acc_f16_to_f16";
constexpr llvm::StringLiteral kHmxMmaFn = "iree_hexagon_hmx_mma_f16";
constexpr llvm::StringLiteral kHmxAccReadFn = "iree_hexagon_hmx_acc_read_f16";

// TODO: Think about a better way to setup these constants other than hardcoded
// values.
// The kernels dereference f16 rows as 64-byte half vectors and f32 rows as
// 128-byte HVX vectors. Packed and config buffers go through HMX instructions,
// whose base-address requirement is `IREE::Hexagon::kHmxAlignment`.
constexpr int64_t kHalfVectorAlignment = 64;
constexpr int64_t kHvxAlignment = 128;

// These checks define the ABI between bufferized HMX IR and the DSP runtime
// kernels. They are required for correctness, not merely optimization hints:
// the kernels use VTCM-only HMX/HVX instructions, issue aligned vector loads,
// and interpret packed buffers using fixed tile shapes and implicit strides.
// Keeping the checks here makes malformed IR fail before it is converted into
// integer pointers, where the element type, memory space, layout, and alignment
// information would no longer be available.

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

func::FuncOp lookupOrCreateFunc(ModuleOp moduleOp, OpBuilder &builder,
                                StringRef name, FunctionType type) {
  if (auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(name)) {
    return funcOp;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());
  auto funcOp = func::FuncOp::create(builder, moduleOp.getLoc(), name, type);
  funcOp.setPrivate();
  return funcOp;
}

Value getDataPointerAsIndex(OpBuilder &builder, Location loc, Value memref) {
  auto memrefType = cast<MemRefType>(memref.getType());
  unsigned elemSize =
      IREE::Hexagon::getHmxElementSizeBytes(memrefType.getElementType());
  assert(elemSize && "unsupported HMX memref element type");

  auto metadata =
      memref::ExtractStridedMetadataOp::create(builder, loc, memref);
  Value basePtr = memref::ExtractAlignedPointerAsIndexOp::create(
      builder, loc, builder.getIndexType(), metadata.getBaseBuffer());
  Value elemOffset = metadata.getOffset();
  if (elemSize == 1) {
    return arith::AddIOp::create(builder, loc, basePtr, elemOffset);
  }
  Value byteScale = arith::ConstantIndexOp::create(builder, loc, elemSize);
  Value byteOffset = arith::MulIOp::create(builder, loc, elemOffset, byteScale);
  return arith::AddIOp::create(builder, loc, basePtr, byteOffset);
}

Value getDataPointerAsI32(OpBuilder &builder, Location loc, Value memref) {
  return arith::IndexCastUIOp::create(
      builder, loc, builder.getI32Type(),
      getDataPointerAsIndex(builder, loc, memref));
}

Value getStrideAsI32(OpBuilder &builder, Location loc, Value memref,
                     int64_t dim) {
  auto metadata =
      memref::ExtractStridedMetadataOp::create(builder, loc, memref);
  return arith::IndexCastUIOp::create(builder, loc, builder.getI32Type(),
                                      metadata.getStrides()[dim]);
}

Value indexToI32(OpBuilder &builder, Location loc, Value value) {
  return arith::IndexCastUIOp::create(builder, loc, builder.getI32Type(),
                                      value);
}

Value constantIndex(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

Value dim(OpBuilder &builder, Location loc, Value memref, int64_t dim) {
  auto type = cast<MemRefType>(memref.getType());
  if (int64_t staticDim = type.getDimSize(dim);
      staticDim != ShapedType::kDynamic) {
    return constantIndex(builder, loc, staticDim);
  }
  return memref::DimOp::create(builder, loc, memref, dim);
}

template <typename OpTy>
func::FuncOp lookupOrCreateFuncFor(OpTy op, OpBuilder &builder, StringRef name,
                                   FunctionType type) {
  ModuleOp moduleOp = op->template getParentOfType<ModuleOp>();
  return lookupOrCreateFunc(moduleOp, builder, name, type);
}

struct LowerPackOp : OpConversionPattern<IREE::Hexagon::HmxPackOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::Hexagon::HmxPackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = adaptor.getSource();
    Value dest = adaptor.getDest();
    int64_t interleaveDim = op.getDim();

    auto i32Type = rewriter.getI32Type();
    auto packType = FunctionType::get(
        rewriter.getContext(),
        {i32Type, i32Type, i32Type, i32Type, i32Type, i32Type, i32Type}, {});

    // The destination tile-major grid is always [interleave_tiles, other_tiles]
    // regardless of source layout. The `dim` attribute names which source
    // dimension carries the interleave (row-pair) axis:
    //   dim == 0: canonical row-major source [interleave, other]. The runtime
    //             `pack_f16` reads it directly (actual rows/cols = source 0/1).
    //   dim == 1: transposed source [other, interleave] (e.g. an N x K weight).
    //             The `pack_transposed_f16` kernel fuses the 32x32 transpose
    //             into the pack; actual interleave/other come from source dims
    //             1/0 and the stride is the between-other-rows stride (source
    //             stride 0).
    if (interleaveDim == 0) {
      func::FuncOp packFn =
          lookupOrCreateFuncFor(op, rewriter, kHmxPackFn, packType);
      func::CallOp::create(
          rewriter, loc, packFn,
          ValueRange{getDataPointerAsI32(rewriter, loc, dest),
                     getDataPointerAsI32(rewriter, loc, source),
                     getStrideAsI32(rewriter, loc, source, 0),
                     indexToI32(rewriter, loc, dim(rewriter, loc, source, 0)),
                     indexToI32(rewriter, loc, dim(rewriter, loc, source, 1)),
                     indexToI32(rewriter, loc, dim(rewriter, loc, dest, 0)),
                     indexToI32(rewriter, loc, dim(rewriter, loc, dest, 1))});
    } else {
      func::FuncOp packFn =
          lookupOrCreateFuncFor(op, rewriter, kHmxPackTransposedFn, packType);
      func::CallOp::create(
          rewriter, loc, packFn,
          ValueRange{getDataPointerAsI32(rewriter, loc, dest),
                     getDataPointerAsI32(rewriter, loc, source),
                     getStrideAsI32(rewriter, loc, source, 0),
                     indexToI32(rewriter, loc, dim(rewriter, loc, source, 1)),
                     indexToI32(rewriter, loc, dim(rewriter, loc, source, 0)),
                     indexToI32(rewriter, loc, dim(rewriter, loc, dest, 0)),
                     indexToI32(rewriter, loc, dim(rewriter, loc, dest, 1))});
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerUnpackOp : OpConversionPattern<IREE::Hexagon::HmxUnpackOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::Hexagon::HmxUnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = adaptor.getSource();
    Value dest = adaptor.getDest();
    auto sourceType = cast<MemRefType>(source.getType());
    auto destType = cast<MemRefType>(dest.getType());

    // The accumulator read-out tile is always f16; the runtime routine to call
    // depends on the destination (output matmul) element type. An f32 dest
    // widens (vcvt) and adds into the init; an f16 dest de-interleaves (vdeal)
    // and adds without widening.
    StringRef unpackFnName =
        destType.getElementType().isF32() ? kHmxUnpackFn : kHmxUnpackF16Fn;

    auto i32Type = rewriter.getI32Type();
    auto unpackType = FunctionType::get(
        rewriter.getContext(),
        {i32Type, i32Type, i32Type, i32Type, i32Type, i32Type, i32Type}, {});
    func::FuncOp unpackFn =
        lookupOrCreateFuncFor(op, rewriter, unpackFnName, unpackType);

    // Tile counts come from the source accumulator grid (rank-5
    // [row_tile, col_tile, ...], or a single tile for rank-3); the actual
    // rows/cols come from the (possibly ragged) destination. The runtime writes
    // only the valid region of each boundary tile, so the destination stays
    // exactly its logical size.
    Value one = constantIndex(rewriter, loc, 1);
    bool sourceIsGrid = IREE::Hexagon::isHmxTileGrid(sourceType);
    Value rowTiles = sourceIsGrid ? dim(rewriter, loc, source, 0) : one;
    Value colTiles = sourceIsGrid ? dim(rewriter, loc, source, 1) : one;
    func::CallOp::create(
        rewriter, loc, unpackFn,
        ValueRange{getDataPointerAsI32(rewriter, loc, dest),
                   getDataPointerAsI32(rewriter, loc, source),
                   getStrideAsI32(rewriter, loc, dest, 0),
                   indexToI32(rewriter, loc, dim(rewriter, loc, dest, 0)),
                   indexToI32(rewriter, loc, dim(rewriter, loc, dest, 1)),
                   indexToI32(rewriter, loc, rowTiles),
                   indexToI32(rewriter, loc, colTiles)});

    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerAccSetupReadOp
    : OpConversionPattern<IREE::Hexagon::HmxAccSetupReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::Hexagon::HmxAccSetupReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto i32Type = rewriter.getI32Type();
    auto setupType = FunctionType::get(rewriter.getContext(), {i32Type}, {});
    func::FuncOp setupFn =
        lookupOrCreateFuncFor(op, rewriter, kHmxAccSetupReadFn, setupType);

    Value configPtr =
        getDataPointerAsI32(rewriter, op.getLoc(), adaptor.getConfig());
    func::CallOp::create(rewriter, op.getLoc(), setupFn, ValueRange{configPtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerAccReadOp : OpConversionPattern<IREE::Hexagon::HmxAccReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::Hexagon::HmxAccReadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(adaptor.getAcc().empty() &&
           "expected the accumulator token to be erased");
    assert(llvm::hasSingleElement(adaptor.getDest()) &&
           "expected the destination memref to remain one value");
    auto i32Type = rewriter.getI32Type();
    auto readType = FunctionType::get(rewriter.getContext(), {i32Type}, {});
    func::FuncOp readFn =
        lookupOrCreateFuncFor(op, rewriter, kHmxAccReadFn, readType);

    Value destPtr = getDataPointerAsI32(
        rewriter, op.getLoc(), llvm::getSingleElement(adaptor.getDest()));
    func::CallOp::create(rewriter, op.getLoc(), readFn, ValueRange{destPtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerMmaOp : OpConversionPattern<IREE::Hexagon::HmxMmaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::Hexagon::HmxMmaOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(llvm::hasSingleElement(adaptor.getLhs()) &&
           llvm::hasSingleElement(adaptor.getRhs()) &&
           "expected HMX tile memrefs to remain one value");
    assert(adaptor.getAccIn().empty() &&
           "expected the accumulator token to be erased");
    auto i32Type = rewriter.getI32Type();
    auto mmaType =
        FunctionType::get(rewriter.getContext(), {i32Type, i32Type}, {});
    func::FuncOp mmaFn =
        lookupOrCreateFuncFor(op, rewriter, kHmxMmaFn, mmaType);

    Value lhsPtr = getDataPointerAsI32(
        rewriter, op.getLoc(), llvm::getSingleElement(adaptor.getLhs()));
    Value rhsPtr = getDataPointerAsI32(
        rewriter, op.getLoc(), llvm::getSingleElement(adaptor.getRhs()));
    func::CallOp::create(rewriter, op.getLoc(), mmaFn,
                         ValueRange{lhsPtr, rhsPtr});

    SmallVector<SmallVector<Value>> replacements(op->getNumResults());
    rewriter.replaceOpWithMultiple(op, std::move(replacements));
    return success();
  }
};

struct LowerAccZeroOp : OpConversionPattern<IREE::Hexagon::HmxAccZeroOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::Hexagon::HmxAccZeroOp op, OneToNOpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto clearType = FunctionType::get(rewriter.getContext(), {}, {});
    func::FuncOp clearFn =
        lookupOrCreateFuncFor(op, rewriter, kHmxAccClearFn, clearType);
    func::CallOp::create(rewriter, op.getLoc(), clearFn, ValueRange{});
    SmallVector<SmallVector<Value>> replacements(op->getNumResults());
    rewriter.replaceOpWithMultiple(op, std::move(replacements));
    return success();
  }
};

LogicalResult verifyNoResidualHmxIR(FunctionOpInterface funcOp) {
  auto containsAccumulator = [](TypeRange types) {
    return llvm::any_of(
        types, [](Type type) { return isa<IREE::Hexagon::HmxAccType>(type); });
  };
  auto functionType = dyn_cast<FunctionType>(funcOp.getFunctionType());
  if (functionType && (containsAccumulator(functionType.getInputs()) ||
                       containsAccumulator(functionType.getResults()))) {
    funcOp.emitError(
        "unexpected HMX accumulator in function signature after lowering");
    return failure();
  }

  LogicalResult result = success();
  funcOp.walk([&](Operation *op) {
    if (op->getName().getStringRef().starts_with("iree_hexagon.hmx")) {
      op->emitOpError("unexpected HMX operation remained after lowering");
      result = failure();
      return WalkResult::interrupt();
    }
    if (containsAccumulator(op->getOperandTypes()) ||
        containsAccumulator(op->getResultTypes())) {
      op->emitOpError(
          "unexpected HMX accumulator token remained after lowering");
      result = failure();
      return WalkResult::interrupt();
    }
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        if (llvm::any_of(block.getArgumentTypes(), [](Type type) {
              return isa<IREE::Hexagon::HmxAccType>(type);
            })) {
          op->emitOpError(
              "unexpected HMX accumulator block argument remained after "
              "lowering");
          result = failure();
          return WalkResult::interrupt();
        }
      }
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

LogicalResult lowerHmxOpsToCalls(ModuleOp moduleOp) {
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  // The accumulator only sequences operations on implicit hardware state. It
  // has no runtime value, so convert it to an empty type range here and avoid
  // an unnecessary placeholder representation.
  typeConverter.addConversion(
      [](IREE::Hexagon::HmxAccType, SmallVectorImpl<Type> &) -> LogicalResult {
        return success();
      });

  RewritePatternSet patterns(moduleOp.getContext());
  patterns.add<LowerPackOp, LowerUnpackOp, LowerAccSetupReadOp, LowerAccReadOp,
               LowerMmaOp, LowerAccZeroOp>(typeConverter,
                                           moduleOp.getContext());

  ConversionTarget target(*moduleOp.getContext());
  target.addIllegalOp<IREE::Hexagon::HmxPackOp, IREE::Hexagon::HmxUnpackOp,
                      IREE::Hexagon::HmxAccSetupReadOp,
                      IREE::Hexagon::HmxAccReadOp, IREE::Hexagon::HmxMmaOp,
                      IREE::Hexagon::HmxAccZeroOp>();
  target.markUnknownOpDynamicallyLegal(
      [&](Operation *op) { return typeConverter.isLegal(op); });
  cf::populateCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                     target);

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

struct HexagonLowerHmxToCallsPass final
    : public impl::HexagonLowerHmxToCallsPassBase<HexagonLowerHmxToCallsPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                cf::ControlFlowDialect, IREE::Hexagon::IREEHexagonDialect>();
  }

  void runOnOperation() override {
    SmallVector<func::FuncOp> funcOps;
    for (func::FuncOp funcOp : getOperation().getOps<func::FuncOp>()) {
      funcOps.push_back(funcOp);
    }
    for (func::FuncOp funcOp : funcOps) {
      if (failed(verifyHmxRuntimeLoweringPreconditions(funcOp))) {
        return signalPassFailure();
      }
    }
    if (failed(lowerHmxOpsToCalls(getOperation()))) {
      return signalPassFailure();
    }
    // The current pipeline only permits the bufferized HMX operations handled
    // above and the accumulator token used to sequence them. Any other HMX op
    // or token carrier here means an earlier pipeline stage violated that
    // contract, so report it.
    for (func::FuncOp funcOp : funcOps) {
      if (failed(verifyNoResidualHmxIR(funcOp))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::hexagon::codegen
