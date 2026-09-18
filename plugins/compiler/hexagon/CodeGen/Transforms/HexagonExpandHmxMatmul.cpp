// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Expands the bufferized `iree_hexagon.hmx.matmul` into the real HMX op
// sequence.
//
// After bufferization the op is a whole-K reduction for one 32x32 output tile,
// operating on VTCM memrefs:
//
//   hmx.matmul ins(%lhs : memref<1x(K/32)x16x32x2xf16>,
//                  %rhs : memref<(K/32)x1x16x32x2xf16>)
//              outs(%grid : memref<1x1x16x32x2xf16>)
//   %acc = subview %grid[0, 0, 0, 0, 0] [1, 1, 16, 32, 2]
//       : memref<1x1x16x32x2xf16> to memref<16x32x2xf16>
//
// The HMX accumulator is register state cleared/accumulated/read (no load), so
// this expands to:
//
//   %cfg = hexagonmem.alloc {alignment = 2048} : memref<2048xi8, 1>
//   hmx.acc.setup_read %cfg
//   %acc_read = hexagonmem.alloc {alignment = 2048} : memref<16x32x2xf16, 1>
//   %a0 = hmx.acc.zero
//   %aK = scf.for %k = 0 to K step 32 iter_args(%it = %a0) {
//     %lt = subview %lhs[0, %k, 0]      [16, 32, 2]
//     %rt = subview %rhs[%k/2, 0, 0]    [16, 32, 2]
//     %n  = hmx.mma %lt, %rt, %it
//     scf.yield %n
//   }
//   hmx.acc.read %aK, %acc_read
//
// The selected HMX pipeline must tile matmul to one output tile before this
// pass. Expansion creates its own rank-reduced view for the accumulator write;
// downstream consumers may use separate views that alias the same tile grid.

#include "hexagon/CodeGen/IR/HexagonOps.h"
#include "hexagon/CodeGen/IR/HmxContracts.h"
#include "hexagon/CodeGen/Passes.h"

#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <functional>

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONEXPANDHMXMATMULPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

// hexKL programs the FP16 accumulator-read bias table from a 256-byte block at
// config+12288. A full tile-sized scratch keeps the allocation and future
// config layouts naturally aligned to `IREE::Hexagon::kHmxAlignment`, which is
// what `HexagonLowerHmxToCallsPass` requires of every buffer reaching the
// runtime kernels.
constexpr int64_t kHmxAccReadConfigBytes = 2048;

static MemRefType getIdentityLayoutType(MemRefType type) {
  return MemRefType::get(type.getShape(), type.getElementType(),
                         MemRefLayoutAttrInterface{}, type.getMemorySpace());
}

/// Creates the physical accumulator-read tile for a singleton rank-5 grid.
static Value createAccumulatorReadSubview(OpBuilder &builder, Location loc,
                                          Value grid) {
  auto gridType = cast<MemRefType>(grid.getType());
  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  SmallVector<OpFoldResult> offsets(IREE::Hexagon::kHmxTileGridRank, zero);
  SmallVector<OpFoldResult> sizes = {
      one, one, builder.getIndexAttr(IREE::Hexagon::kHmxTileRows),
      builder.getIndexAttr(IREE::Hexagon::kHmxTileColumns),
      builder.getIndexAttr(IREE::Hexagon::kHmxTileInterleave)};
  SmallVector<OpFoldResult> strides(IREE::Hexagon::kHmxTileGridRank, one);
  SmallVector<int64_t> resultShape = {IREE::Hexagon::kHmxTileRows,
                                      IREE::Hexagon::kHmxTileColumns,
                                      IREE::Hexagon::kHmxTileInterleave};
  MemRefType resultType = memref::SubViewOp::inferRankReducedResultType(
      resultShape, gridType, offsets, sizes, strides);
  return memref::SubViewOp::create(builder, loc, resultType, grid, offsets,
                                   sizes, strides)
      .getResult();
}

struct HexagonExpandHmxMatmulPass final
    : public impl::HexagonExpandHmxMatmulPassBase<HexagonExpandHmxMatmulPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect,
                    hexagonmem::HexagonMemDialect,
                    IREE::Hexagon::IREEHexagonDialect>();
  }

  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();

    SmallVector<IREE::Hexagon::HmxMatmulOp> matmuls;
    funcOp.walk([&](IREE::Hexagon::HmxMatmulOp op) { matmuls.push_back(op); });
    if (matmuls.empty()) {
      return;
    }

    IRRewriter rewriter(&getContext());

    auto createEntryAlloc = [&](MemRefType type, int64_t alignment) -> Value {
      OpBuilder::InsertionGuard guard(rewriter);
      Block &entryBlock = funcOp.getFunctionBody().front();
      rewriter.setInsertionPointToStart(&entryBlock);
      return hexagonmem::AllocOp::create(rewriter, funcOp.getLoc(), type,
                                         ValueRange{},
                                         rewriter.getI64IntegerAttr(alignment))
          .getBuffer();
    };

    auto configTy =
        MemRefType::get({kHmxAccReadConfigBytes}, rewriter.getI8Type(),
                        MemRefLayoutAttrInterface{},
                        IntegerAttr::get(rewriter.getI32Type(), 1));
    Value config = createEntryAlloc(configTy, IREE::Hexagon::kHmxAlignment);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfterValue(config);
      IREE::Hexagon::HmxAccSetupReadOp::create(rewriter, funcOp.getLoc(),
                                               config);
    }

    SmallVector<std::pair<MemRefType, Value>> accReadScratch;
    auto getAccReadScratch = [&](MemRefType type) -> Value {
      type = getIdentityLayoutType(type);
      for (auto [scratchType, scratch] : accReadScratch) {
        if (scratchType == type) {
          return scratch;
        }
      }
      Value scratch = createEntryAlloc(type, IREE::Hexagon::kHmxAlignment);
      accReadScratch.push_back({type, scratch});
      return scratch;
    };

    SmallVector<std::pair<Value, Value>> alignedPackedScratch;
    std::function<FailureOr<Value>(Value)> getAlignedPackedScratch =
        [&](Value original) -> FailureOr<Value> {
      // Tiled HMX operands reach this pass through views of the packed tile
      // grid. A view derives its base address from its source, so align the
      // underlying allocation and keep the view: substituting a fresh buffer
      // for the view itself would both lose the packed data and fail
      // HexagonMem-to-LLVM lowering.
      if (Value viewSource = IREE::Hexagon::getHmxMemRefViewSource(original)) {
        if (failed(getAlignedPackedScratch(viewSource))) {
          return failure();
        }
        return original;
      }

      for (auto [oldValue, scratch] : alignedPackedScratch) {
        if (oldValue == original || scratch == original) {
          return scratch;
        }
      }
      // Any stronger alignment is good enough.
      std::optional<int64_t> declared =
          IREE::Hexagon::getDeclaredHmxAlignment(original);
      if (declared && *declared != 0 &&
          *declared % IREE::Hexagon::kHmxAlignment == 0) {
        return original;
      }

      // Re-allocating is only sound for a buffer this pipeline allocated:
      // replacing an allocation moves every writer with it, so whatever fills
      // the packed grid follows. A buffer entering the function from outside --
      // a block argument, a binding subspan, a global -- has its producer
      // beyond this rewrite, and substituting a fresh allocation would silently
      // drop its contents. Report those instead, so the missing alignment gets
      // fixed where the buffer is created.
      Operation *defOp = original.getDefiningOp();
      if (!isa_and_nonnull<hexagonmem::AllocOp, memref::AllocOp,
                           memref::AllocaOp>(defOp)) {
        return emitError(original.getLoc())
               << "HMX operand must be backed by an allocation this pass can "
                  "align to "
               << IREE::Hexagon::kHmxAlignment
               << " bytes, but it comes from outside the function or from an "
                  "operation with no alignment guarantee";
      }

      auto type = cast<MemRefType>(original.getType());
      type = getIdentityLayoutType(type);
      Value scratch = createEntryAlloc(type, IREE::Hexagon::kHmxAlignment);
      original.replaceUsesWithIf(scratch, [&](OpOperand &) { return true; });
      if (original.use_empty()) {
        rewriter.eraseOp(defOp);
      }
      alignedPackedScratch.push_back({original, scratch});
      return scratch;
    };

    for (IREE::Hexagon::HmxMatmulOp op : matmuls) {
      if (failed(expand(rewriter, op, getAccReadScratch,
                        getAlignedPackedScratch))) {
        return signalPassFailure();
      }
    }

    // The configuration buffer is introduced after buffer deallocation has
    // already run, so give it an explicit function-scoped lifetime. Keep it
    // alive through all HMX accumulator reads and release it on every function
    // exit. Note that this is not strictly necessary and deallocation could
    // happen just afterregister accumulation configuration has finished.
    for (Block &block : funcOp.getFunctionBody()) {
      Operation *terminator = block.getTerminator();
      if (!terminator->hasTrait<OpTrait::ReturnLike>()) {
        continue;
      }
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(terminator);
      hexagonmem::DeallocOp::create(rewriter, terminator->getLoc(), config);
    }
  }

  LogicalResult
  expand(IRRewriter &rewriter, IREE::Hexagon::HmxMatmulOp op,
         llvm::function_ref<Value(MemRefType)> getAccReadScratch,
         llvm::function_ref<FailureOr<Value>(Value)> getAlignedPackedScratch) {
    Location loc = op.getLoc();
    Value originalAcc = op.getAcc();
    auto lhsTy = cast<MemRefType>(op.getLhs().getType());
    auto rhsTy = cast<MemRefType>(op.getRhs().getType());
    auto accTy = cast<MemRefType>(originalAcc.getType());

    // The verifier guarantees rank-5 inputs, static matching K grids, and a
    // rank-3 tile or rank-5 accumulator grid. It also already requires
    // singleton M and N grids whenever the accumulator is the rank-3 tile; this
    // check extends that to the rank-5 accumulator form, because expansion
    // supports only one M/N output tile.
    if (lhsTy.getDimSize(0) != 1 || rhsTy.getDimSize(1) != 1) {
      return op.emitError(
          "hmx.matmul expansion expects singleton M and N tile grids");
    }

    // Alignment repair preserves the operand shapes, so the checks above hold
    // for the returned buffers too.
    FailureOr<Value> lhs = getAlignedPackedScratch(op.getLhs());
    FailureOr<Value> rhs = getAlignedPackedScratch(op.getRhs());
    if (failed(lhs) || failed(rhs)) {
      return failure();
    }

    // K comes from the number of activation K tiles.
    int64_t kTiles = lhsTy.getShape()[1];
    int64_t k = IREE::Hexagon::getHmxElementCapacity(kTiles);

    auto accType =
        IREE::Hexagon::HmxAccType::get(rewriter.getContext(),
                                       {IREE::Hexagon::kHmxLogicalTileSize,
                                        IREE::Hexagon::kHmxLogicalTileSize},
                                       rewriter.getF32Type());

    rewriter.setInsertionPoint(op);
    Value acc;
    Value obsoleteAcc;
    if (IREE::Hexagon::isHmxTileGrid(accTy)) {
      FailureOr<Value> alignedGrid = getAlignedPackedScratch(originalAcc);
      if (failed(alignedGrid)) {
        return failure();
      }
      acc = createAccumulatorReadSubview(rewriter, loc, *alignedGrid);
    } else {
      obsoleteAcc = originalAcc;
      acc = getAccReadScratch(accTy);
      originalAcc.replaceUsesWithIf(acc, [&](OpOperand &use) {
        return use.getOwner() != op.getOperation();
      });
    }

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value cK = arith::ConstantIndexOp::create(rewriter, loc, k);
    Value cStep = arith::ConstantIndexOp::create(
        rewriter, loc, IREE::Hexagon::kHmxLogicalTileSize);

    auto expandOneTile = [&](OpBuilder &builder, Location loc, Value lhsBase,
                             Value rhsBase, Value accTile) {
      Value accZero =
          IREE::Hexagon::HmxAccZeroOp::create(builder, loc, accType).getAcc();
      auto forOp = scf::ForOp::create(
          builder, loc, c0, cK, cStep, ValueRange{accZero},
          [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
            Value accIn = iterArgs[0];
            Value kTile = arith::DivUIOp::create(
                b, loc, iv,
                arith::ConstantIndexOp::create(
                    b, loc, IREE::Hexagon::kHmxLogicalTileSize));

            SmallVector<OpFoldResult> unitStrides = {
                b.getIndexAttr(1), b.getIndexAttr(1), b.getIndexAttr(1),
                b.getIndexAttr(1), b.getIndexAttr(1)};
            SmallVector<OpFoldResult> lhsOffsets = {
                b.getIndexAttr(0), kTile, b.getIndexAttr(0), b.getIndexAttr(0),
                b.getIndexAttr(0)};
            SmallVector<OpFoldResult> lhsSizes = {
                b.getIndexAttr(1), b.getIndexAttr(1),
                b.getIndexAttr(IREE::Hexagon::kHmxTileRows),
                b.getIndexAttr(IREE::Hexagon::kHmxTileColumns),
                b.getIndexAttr(IREE::Hexagon::kHmxTileInterleave)};
            Value lhsTile = memref::SubViewOp::create(
                b, loc, lhsBase, lhsOffsets, lhsSizes, unitStrides);

            SmallVector<OpFoldResult> rhsOffsets = {
                kTile, b.getIndexAttr(0), b.getIndexAttr(0), b.getIndexAttr(0),
                b.getIndexAttr(0)};
            SmallVector<OpFoldResult> rhsSizes = {
                b.getIndexAttr(1), b.getIndexAttr(1),
                b.getIndexAttr(IREE::Hexagon::kHmxTileRows),
                b.getIndexAttr(IREE::Hexagon::kHmxTileColumns),
                b.getIndexAttr(IREE::Hexagon::kHmxTileInterleave)};
            Value rhsTile = memref::SubViewOp::create(
                b, loc, rhsBase, rhsOffsets, rhsSizes, unitStrides);

            Value next = IREE::Hexagon::HmxMmaOp::create(
                             b, loc, accType, lhsTile, rhsTile, accIn)
                             .getAccOut();
            scf::YieldOp::create(b, loc, next);
          });
      IREE::Hexagon::HmxAccReadOp::create(builder, loc, forOp.getResult(0),
                                          accTile);
    };

    expandOneTile(rewriter, loc, *lhs, *rhs, acc);

    rewriter.eraseOp(op);
    if (obsoleteAcc && obsoleteAcc.use_empty()) {
      if (Operation *defOp = obsoleteAcc.getDefiningOp();
          isa_and_nonnull<hexagonmem::AllocOp, memref::AllocOp>(defOp)) {
        rewriter.eraseOp(defOp);
      }
    }
    return success();
  }
};

} // namespace

} // namespace mlir::iree_compiler::hexagon::codegen
