// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Conversion/HexagonHmxToLLVM.h"

#include "hexagon/CodeGen/IR/HexagonOps.h"
#include "hexagon/CodeGen/IR/HmxContracts.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::iree_compiler::hexagon::codegen {

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

// Replaces `op` with a call to the void runtime kernel `name`, declaring the
// kernel on first use. The only results HMX operations have are accumulator
// values, which convert to nothing.
LogicalResult replaceWithKernelCall(ConversionPatternRewriter &rewriter,
                                    Operation *op, StringRef name,
                                    ValueRange args) {
  FailureOr<LLVM::LLVMFuncOp> fn =
      LLVM::lookupOrCreateFn(rewriter, op->getParentOfType<ModuleOp>(), name,
                             llvm::to_vector(args.getTypes()),
                             LLVM::LLVMVoidType::get(rewriter.getContext()));
  if (failed(fn)) {
    return rewriter.notifyMatchFailure(op, "incompatible kernel declaration");
  }
  LLVM::CallOp::create(rewriter, op->getLoc(), *fn, args);
  rewriter.replaceOpWithMultiple(
      op, SmallVector<SmallVector<Value>>(op->getNumResults()));
  return success();
}

// The kernels take every size, count and stride as a 32-bit integer.
Value createI32(OpBuilder &builder, Location loc, int64_t value) {
  return LLVM::ConstantOp::create(builder, loc, builder.getI32Type(),
                                  builder.getI32IntegerAttr(value));
}

Value castToI32(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isInteger(32)) {
    return value;
  }
  return LLVM::TruncOp::create(builder, loc, builder.getI32Type(), value);
}

// Bundles a converted memref with its original type, from which static sizes
// and strides are read directly.
struct Buffer {
  Buffer(Value original, Value converted)
      : type(cast<MemRefType>(original.getType())), descriptor(converted) {}

  // A pointer to the first element: the aligned pointer advanced by the offset.
  Value pointer(OpBuilder &builder, Location loc,
                const LLVMTypeConverter &typeConverter) {
    return descriptor.bufferPtr(builder, loc, typeConverter, type);
  }

  Value size(OpBuilder &builder, Location loc, unsigned dim) {
    if (!type.isDynamicDim(dim)) {
      return createI32(builder, loc, type.getDimSize(dim));
    }
    return castToI32(builder, loc, descriptor.size(builder, loc, dim));
  }

  Value stride(OpBuilder &builder, Location loc, unsigned dim) {
    int64_t staticStride = type.getStridesAndOffset().first[dim];
    if (ShapedType::isStatic(staticStride)) {
      return createI32(builder, loc, staticStride);
    }
    return castToI32(builder, loc, descriptor.stride(builder, loc, dim));
  }

  MemRefType type;
  MemRefDescriptor descriptor;
};

struct ConvertPackOp : ConvertOpToLLVMPattern<IREE::Hexagon::HmxPackOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(IREE::Hexagon::HmxPackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Buffer source(op.getSource(), adaptor.getSource());
    Buffer dest(op.getDest(), adaptor.getDest());

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
    bool transposed = op.getDim() == 1;
    return replaceWithKernelCall(
        rewriter, op, transposed ? kHmxPackTransposedFn : kHmxPackFn,
        {dest.pointer(rewriter, loc, *getTypeConverter()),
         source.pointer(rewriter, loc, *getTypeConverter()),
         source.stride(rewriter, loc, 0),
         source.size(rewriter, loc, transposed ? 1 : 0),
         source.size(rewriter, loc, transposed ? 0 : 1),
         dest.size(rewriter, loc, 0), dest.size(rewriter, loc, 1)});
  }
};

struct ConvertUnpackOp : ConvertOpToLLVMPattern<IREE::Hexagon::HmxUnpackOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(IREE::Hexagon::HmxUnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Buffer source(op.getSource(), adaptor.getSource());
    Buffer dest(op.getDest(), adaptor.getDest());

    // The accumulator read-out tile is always f16; the runtime routine to call
    // depends on the destination (output matmul) element type. An f32 dest
    // widens (vcvt) and adds into the init; an f16 dest de-interleaves (vdeal)
    // and adds without widening.
    StringRef fnName =
        dest.type.getElementType().isF32() ? kHmxUnpackFn : kHmxUnpackF16Fn;

    // Tile counts come from the source accumulator grid (rank-5
    // [row_tile, col_tile, ...], or a single tile for rank-3); the actual
    // rows/cols come from the (possibly ragged) destination. The runtime writes
    // only the valid region of each boundary tile, so the destination stays
    // exactly its logical size.
    bool sourceIsGrid = IREE::Hexagon::isHmxTileGrid(source.type);
    Value rowTiles = sourceIsGrid ? source.size(rewriter, loc, 0)
                                  : createI32(rewriter, loc, 1);
    Value colTiles = sourceIsGrid ? source.size(rewriter, loc, 1)
                                  : createI32(rewriter, loc, 1);
    return replaceWithKernelCall(
        rewriter, op, fnName,
        {dest.pointer(rewriter, loc, *getTypeConverter()),
         source.pointer(rewriter, loc, *getTypeConverter()),
         dest.stride(rewriter, loc, 0), dest.size(rewriter, loc, 0),
         dest.size(rewriter, loc, 1), rowTiles, colTiles});
  }
};

struct ConvertAccSetupReadOp
    : ConvertOpToLLVMPattern<IREE::Hexagon::HmxAccSetupReadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(IREE::Hexagon::HmxAccSetupReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Buffer config(op.getConfig(), adaptor.getConfig());
    return replaceWithKernelCall(
        rewriter, op, kHmxAccSetupReadFn,
        {config.pointer(rewriter, op.getLoc(), *getTypeConverter())});
  }
};

struct ConvertAccZeroOp : ConvertOpToLLVMPattern<IREE::Hexagon::HmxAccZeroOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(IREE::Hexagon::HmxAccZeroOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return replaceWithKernelCall(rewriter, op, kHmxAccClearFn, {});
  }
};

struct ConvertMmaOp : ConvertOpToLLVMPattern<IREE::Hexagon::HmxMmaOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(IREE::Hexagon::HmxMmaOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Buffer lhs(op.getLhs(), llvm::getSingleElement(adaptor.getLhs()));
    Buffer rhs(op.getRhs(), llvm::getSingleElement(adaptor.getRhs()));
    return replaceWithKernelCall(
        rewriter, op, kHmxMmaFn,
        {lhs.pointer(rewriter, loc, *getTypeConverter()),
         rhs.pointer(rewriter, loc, *getTypeConverter())});
  }
};

struct ConvertAccReadOp : ConvertOpToLLVMPattern<IREE::Hexagon::HmxAccReadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(IREE::Hexagon::HmxAccReadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Buffer dest(op.getDest(), llvm::getSingleElement(adaptor.getDest()));
    return replaceWithKernelCall(
        rewriter, op, kHmxAccReadFn,
        {dest.pointer(rewriter, op.getLoc(), *getTypeConverter())});
  }
};

} // namespace

void populateHexagonHmxToLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  typeConverter.addConversion(
      [](IREE::Hexagon::HmxAccType, SmallVectorImpl<Type> &) {
        return success();
      });
  patterns.addWithLabel<ConvertPackOp, ConvertUnpackOp, ConvertAccSetupReadOp,
                        ConvertAccZeroOp, ConvertMmaOp, ConvertAccReadOp>(
      {"hexagon-hmx-to-llvm"}, typeConverter);
}

} // namespace mlir::iree_compiler::hexagon::codegen
