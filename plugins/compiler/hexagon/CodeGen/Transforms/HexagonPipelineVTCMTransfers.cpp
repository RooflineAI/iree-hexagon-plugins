// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/CodeGen/Passes.h"
#include "hexagon/Common/Common.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-hexagon-pipeline-vtcm-transfers"

namespace mlir::iree_compiler::hexagon::codegen {

#define GEN_PASS_DEF_HEXAGONPIPELINEVTCMTRANSFERSPASS
#include "hexagon/CodeGen/Passes.h.inc"

namespace {

using ::mlir::hexagon::DEFAULT_DDR_ADDRESS_SPACE;
using ::mlir::hexagon::VTCM_ADDRESS_SPACE;

// Limits of the UserDMA descriptors (UserDMADescriptors.h): the length of a
// 1D transfer has 24 bits; the width, height and strides of a 2D transfer
// have 16 bits each. The runtime rejects longer 1D transfers and copies wider
// 2D transfers row by row, synchronously.
constexpr int64_t kMax1DTransferBytes = 0xFFFFFF;
constexpr int64_t kMax2DFieldValue = 0xFFFF;

constexpr int64_t kNumBuffers = 2;

/// A copy between DDR and VTCM as one memref.dma_start, in the form
/// hexagon-mlir's DMAToLLVM lowers (see createDMAStartOp in
/// DMATransferUtil.cpp): either one contiguous transfer, or rows of `width`
/// elements that are `stride` elements apart in DDR and dense in VTCM. The
/// number of rows of a ragged tile is only known at runtime; `maxElements`
/// bounds the number of elements by the size of the VTCM buffer.
struct DMAShape {
  int64_t maxElements = 0;
  std::optional<int64_t> stride;
  int64_t width = 0;
};

/// A DDR <-> VTCM copy in the body of a tile loop, staged through `buffer`,
/// a VTCM allocation outside of the loop. The VTCM side of the copy is the
/// buffer or a view of it.
struct Transfer {
  memref::CopyOp copy;
  memref::AllocOp buffer;
  DMAShape shape;
  Value tag;
};

struct TileLoopTransfers {
  SmallVector<Transfer> loads;
  SmallVector<Transfer> stores;
};

bool isInMemorySpace(Type type, unsigned memorySpace) {
  auto memrefType = dyn_cast<MemRefType>(type);
  int memorySpaceId;
  return memrefType &&
         ::mlir::hexagon::isMemorySpaceIntTypeOrDefault(memrefType,
                                                        memorySpaceId) &&
         memorySpaceId == static_cast<int>(memorySpace);
}

bool isLoad(memref::CopyOp copy) {
  return isInMemorySpace(copy.getSource().getType(),
                         DEFAULT_DDR_ADDRESS_SPACE) &&
         isInMemorySpace(copy.getTarget().getType(), VTCM_ADDRESS_SPACE);
}

bool isStore(memref::CopyOp copy) {
  return isInMemorySpace(copy.getSource().getType(), VTCM_ADDRESS_SPACE) &&
         isInMemorySpace(copy.getTarget().getType(), DEFAULT_DDR_ADDRESS_SPACE);
}

/// Returns the VTCM buffer `vtcmValue` is a view of through memref.subview
/// ops, or nullptr.
memref::AllocOp getStagingBuffer(Value vtcmValue) {
  while (auto subview = vtcmValue.getDefiningOp<memref::SubViewOp>())
    vtcmValue = subview.getSource();
  return vtcmValue.getDefiningOp<memref::AllocOp>();
}

/// Returns the static row stride of a rank 1 or 2 memref whose rows are dense
/// (i.e. whose innermost stride is 1), or nullopt. A rank 1 memref is one
/// row. A single row has no row stride either; its rows are dense.
std::optional<int64_t> getRowStride(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)) || strides.back() != 1)
    return std::nullopt;
  int64_t width = type.getShape().back();
  if (type.getRank() == 1 || type.getDimSize(0) == 1)
    return width;
  if (ShapedType::isDynamic(strides.front()))
    return std::nullopt;
  return strides.front();
}

/// Returns how the copy between `ddrType` and `vtcmType`, a view of
/// `bufferType`, is issued as a single asynchronous DMA, or nullopt if it
/// can't be one. Only the number of rows may be dynamic: it is then bounded
/// by the size of the buffer.
std::optional<DMAShape> getDMAShape(MemRefType ddrType, MemRefType vtcmType,
                                    MemRefType bufferType) {
  for (MemRefType type : {ddrType, vtcmType}) {
    if (type.getRank() < 1 || type.getRank() > 2 ||
        ShapedType::isDynamic(type.getShape().back()))
      return std::nullopt;
  }
  if (!bufferType.hasStaticShape())
    return std::nullopt;
  Type elementType = ddrType.getElementType();
  if (elementType != vtcmType.getElementType() || !elementType.isIntOrFloat() ||
      elementType.getIntOrFloatBitWidth() % 8 != 0)
    return std::nullopt;
  int64_t elementBytes = elementType.getIntOrFloatBitWidth() / 8;
  int64_t maxElements = vtcmType.hasStaticShape() ? vtcmType.getNumElements()
                                                  : bufferType.getNumElements();
  int64_t width = vtcmType.getShape().back();
  if (maxElements == 0 || ddrType.getShape().back() != width)
    return std::nullopt;
  // The lowering takes the row stride in VTCM to be the width.
  if (getRowStride(vtcmType) != width)
    return std::nullopt;

  std::optional<int64_t> stride = getRowStride(ddrType);
  if (!stride || *stride < width)
    return std::nullopt;
  if (*stride == width) {
    if (maxElements * elementBytes > kMax1DTransferBytes)
      return std::nullopt;
    return DMAShape{maxElements, std::nullopt, 0};
  }
  if (width * elementBytes > kMax2DFieldValue ||
      maxElements / width > kMax2DFieldValue ||
      *stride * elementBytes > kMax2DFieldValue)
    return std::nullopt;
  return DMAShape{maxElements, *stride, width};
}

/// Returns the ops of `body` that compute `values`, in program order.
SmallVector<Operation *> getComputationInBody(ValueRange values, Block *body) {
  BackwardSliceOptions options;
  options.inclusive = true;
  options.filter = [&](Operation *op) { return op->getBlock() == body; };
  llvm::SetVector<Operation *> slice;
  for (Value value : values) {
    Operation *def = value.getDefiningOp();
    if (def && def->getBlock() == body)
      (void)getBackwardSlice(def, &slice, options);
  }
  SmallVector<Operation *> ops = slice.takeVector();
  llvm::sort(ops,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  return ops;
}

/// Whether the computation of `value` in `body` can be cloned for another
/// iteration: it must be free of side effects and regions.
bool isRecomputable(Value value, Block *body) {
  return llvm::all_of(getComputationInBody(value, body), [](Operation *op) {
    return op->getNumRegions() == 0 && isMemoryEffectFree(op);
  });
}

/// Collects the users of `value` and, through view-like ops, of its views.
void collectUsersThroughViews(Value value,
                              SmallVectorImpl<Operation *> &users) {
  for (Operation *user : value.getUsers()) {
    users.push_back(user);
    auto viewOp = dyn_cast<ViewLikeOpInterface>(user);
    if (viewOp && viewOp.getViewSource() == value) {
      for (Value result : user->getResults())
        collectUsersThroughViews(result, users);
    }
  }
}

/// Whether `buffer` holds the tile only between its load and its store: all
/// its uses are inside `loop`, the load (if any) comes before all other uses
/// in the iteration and the store (if any) after all of them. Subviews of the
/// buffer do not access it and are not uses: their own users are checked.
/// sinkViewsIntoLoop moves the ones outside of the loop into it.
bool isStagedPerIteration(scf::ForOp loop, memref::AllocOp buffer,
                          memref::CopyOp load, memref::CopyOp store) {
  Block *body = loop.getBody();
  SmallVector<Operation *> users;
  collectUsersThroughViews(buffer.getResult(), users);
  for (Operation *user : users) {
    if (isa<memref::SubViewOp>(user) ||
        (isa<memref::DeallocOp>(user) && !loop->isAncestor(user)))
      continue;
    if (!loop->isProperAncestor(user))
      return false;
    Operation *ancestor = body->findAncestorOpInBlock(*user);
    if (load && ancestor != load && ancestor->isBeforeInBlock(load))
      return false;
    if (store && ancestor != store && store->isBeforeInBlock(ancestor))
      return false;
  }
  return true;
}

/// Collects the DDR <-> VTCM transfers of `loop`. Fails if they can not all
/// become asynchronous DMAs: on the in-order DMA engine a synchronous
/// transfer left in the loop would wait for the prefetch issued before it.
FailureOr<TileLoopTransfers> analyzeTileLoop(scf::ForOp loop) {
  Block *body = loop.getBody();
  TileLoopTransfers transfers;
  llvm::SmallDenseMap<Operation *, memref::CopyOp> loadOf, storeOf;
  for (auto copy : body->getOps<memref::CopyOp>()) {
    bool load = isLoad(copy);
    if (!load && !isStore(copy))
      continue;
    Value vtcmValue = load ? copy.getTarget() : copy.getSource();
    Value ddrValue = load ? copy.getSource() : copy.getTarget();
    memref::AllocOp buffer = getStagingBuffer(vtcmValue);
    if (!buffer || loop->isAncestor(buffer)) {
      LLVM_DEBUG(llvm::dbgs() << "not staged through a buffer allocated "
                                 "outside of the loop: "
                              << copy << "\n");
      return failure();
    }
    std::optional<DMAShape> shape =
        getDMAShape(cast<MemRefType>(ddrValue.getType()),
                    cast<MemRefType>(vtcmValue.getType()), buffer.getType());
    if (!shape) {
      LLVM_DEBUG(llvm::dbgs()
                 << "not a single asynchronous DMA: " << copy << "\n");
      return failure();
    }
    // The prefetch recomputes both views of a load for the next iteration.
    if (!isRecomputable(ddrValue, body) ||
        (load && !isRecomputable(vtcmValue, body))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "view can not be recomputed: " << copy << "\n");
      return failure();
    }
    auto &copyOf = load ? loadOf : storeOf;
    if (!copyOf.try_emplace(buffer, copy).second)
      return failure();
    (load ? transfers.loads : transfers.stores)
        .push_back(Transfer{copy, buffer, *shape, Value()});
  }

  // Each buffer must hold its tile for one iteration only.
  llvm::SetVector<Operation *> buffers;
  for (auto &[buffer, copy] : loadOf)
    buffers.insert(buffer);
  for (auto &[buffer, copy] : storeOf)
    buffers.insert(buffer);
  for (Operation *buffer : buffers) {
    if (!isStagedPerIteration(loop, cast<memref::AllocOp>(buffer),
                              loadOf.lookup(buffer), storeOf.lookup(buffer))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "buffer is not staged per iteration: " << *buffer << "\n");
      return failure();
    }
  }

  // No other DDR <-> VTCM copy may remain anywhere in the loop.
  WalkResult walkResult = loop.getBody()->walk([&](memref::CopyOp copy) {
    if (!isLoad(copy) && !isStore(copy))
      return WalkResult::advance();
    Value vtcmValue = isLoad(copy) ? copy.getTarget() : copy.getSource();
    Operation *buffer = getStagingBuffer(vtcmValue);
    auto &copyOf = isLoad(copy) ? loadOf : storeOf;
    if (buffer && copyOf.lookup(buffer) == copy)
      return WalkResult::advance();
    LLVM_DEBUG(llvm::dbgs()
               << "synchronous transfer would remain: " << copy << "\n");
    return WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted())
    return failure();
  return transfers;
}

/// Returns the buffer slot of the iteration `iv`: ((iv - lb) / step) mod 2.
Value createSlot(OpBuilder &builder, Location loc, Value iv, int64_t lb,
                 int64_t step) {
  AffineExpr d0 = builder.getAffineDimExpr(0);
  AffineMap map =
      AffineMap::get(1, 0, ((d0 - lb).floorDiv(step)) % kNumBuffers);
  OpFoldResult slot =
      affine::makeComposedFoldedAffineApply(builder, loc, map, {iv});
  return getValueOrCreateConstantIndexOp(builder, loc, slot);
}

/// Returns the number of elements of `vtcmView`, the VTCM side of a transfer.
/// Only the number of rows may be dynamic (see getDMAShape).
Value createNumElements(OpBuilder &builder, Location loc, Value vtcmView,
                        const DMAShape &shape) {
  auto type = cast<MemRefType>(vtcmView.getType());
  if (type.hasStaticShape())
    return arith::ConstantIndexOp::create(builder, loc, shape.maxElements);
  Value rows = memref::DimOp::create(builder, loc, vtcmView, 0);
  if (type.getRank() == 1)
    return rows;
  Value width =
      arith::ConstantIndexOp::create(builder, loc, type.getShape().back());
  return arith::MulIOp::create(builder, loc, rows, width);
}

void createDMAStart(OpBuilder &builder, Location loc, Value source,
                    Value target, Value tag, Value slot,
                    const DMAShape &shape) {
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  SmallVector<Value> sourceIndices(cast<MemRefType>(source.getType()).getRank(),
                                   zero);
  SmallVector<Value> targetIndices(cast<MemRefType>(target.getType()).getRank(),
                                   zero);
  Value vtcmView =
      isInMemorySpace(target.getType(), VTCM_ADDRESS_SPACE) ? target : source;
  Value numElements = createNumElements(builder, loc, vtcmView, shape);
  Value stride, width;
  if (shape.stride) {
    stride = arith::ConstantIndexOp::create(builder, loc, *shape.stride);
    width = arith::ConstantIndexOp::create(builder, loc, shape.width);
  }
  memref::DmaStartOp::create(builder, loc, source, sourceIndices, target,
                             targetIndices, numElements, tag, ValueRange{slot},
                             stride, width);
}

/// The wait identifies the transfer by its tag only; for a tile with a
/// dynamic number of rows, the size of the full buffer is given as its number
/// of elements.
void createDMAWait(OpBuilder &builder, Location loc, const Transfer &transfer,
                   Value slot) {
  Value numElements =
      arith::ConstantIndexOp::create(builder, loc, transfer.shape.maxElements);
  memref::DmaWaitOp::create(builder, loc, transfer.tag, ValueRange{slot},
                            numElements);
}

/// Moves the subviews of `value` that are computed in front of `loop`, but
/// only used inside of it, to the start of the loop body. Then all uses of a
/// buffer checked by isStagedPerIteration are inside the loop, as
/// memref::multiBuffer requires, and the views take the slot of the
/// iteration. A subview used inside of the loop dominates it, and so do its
/// operands.
void sinkViewsIntoLoop(RewriterBase &rewriter, Value value, scf::ForOp loop) {
  for (Operation *user : llvm::make_early_inc_range(value.getUsers())) {
    auto subview = dyn_cast<memref::SubViewOp>(user);
    if (!subview || loop->isAncestor(subview))
      continue;
    sinkViewsIntoLoop(rewriter, subview.getResult(), loop);
    if (!subview->use_empty()) {
      rewriter.setInsertionPointToStart(loop.getBody());
      Operation *clone = rewriter.clone(*subview);
      rewriter.replaceAllUsesWith(subview.getResult(), clone->getResult(0));
    }
    rewriter.eraseOp(subview);
  }
}

/// Double-buffers the transfers of `loop` and rotates it (see the pass
/// description). The compute of the loop body stays in place; only the
/// DMA issue and wait operations are added outside of it.
LogicalResult pipelineTileLoop(RewriterBase &rewriter, scf::ForOp loop,
                               TileLoopTransfers &transfers, int64_t lb,
                               int64_t ub, int64_t step) {
  Location loc = loop.getLoc();
  Block *body = loop.getBody();
  Value iv = loop.getInductionVar();

  // A second buffer for every staged buffer. Users of the buffer then take
  // the slot of the iteration. The override analysis of multiBuffer is not
  // needed: analyzeTileLoop checked that every use of a buffer is inside the
  // loop, once its views are moved there, and between its load and store, and
  // iree-hexagon-coalesce-vtcm-loops that no buffer carries data from one tile
  // to the next.
  llvm::SetVector<Operation *> buffers;
  for (const Transfer &transfer :
       llvm::concat<Transfer>(transfers.loads, transfers.stores))
    buffers.insert(transfer.buffer);
  for (Operation *buffer : buffers) {
    sinkViewsIntoLoop(rewriter, buffer->getResult(0), loop);
    if (failed(memref::multiBuffer(rewriter, cast<memref::AllocOp>(buffer),
                                   kNumBuffers,
                                   /*skipOverrideAnalysis=*/true)))
      return buffer->emitOpError("failed to double-buffer a VTCM buffer");
  }

  // One tag per buffer slot and transfer, at the function entry.
  Block &entryBlock =
      loop->getParentOfType<FunctionOpInterface>().getFunctionBody().front();
  rewriter.setInsertionPointToStart(&entryBlock);
  auto tagType = MemRefType::get({kNumBuffers}, rewriter.getI32Type());
  for (Transfer &transfer :
       llvm::concat<Transfer>(transfers.loads, transfers.stores))
    transfer.tag = memref::AllocaOp::create(rewriter, loc, tagType);

  // Issues the loads of the iteration `loadIv` by recomputing the views of
  // their DDR source and VTCM slot.
  SmallVector<Value> loadViews;
  for (Transfer &transfer : transfers.loads) {
    loadViews.push_back(transfer.copy.getSource());
    loadViews.push_back(transfer.copy.getTarget());
  }
  SmallVector<Operation *> loadViewOps = getComputationInBody(loadViews, body);
  auto issueLoads = [&](Value loadIv) {
    IRMapping mapping;
    mapping.map(iv, loadIv);
    for (Operation *op : loadViewOps)
      rewriter.clone(*op, mapping);
    Value slot = createSlot(rewriter, loc, loadIv, lb, step);
    for (auto [index, transfer] : llvm::enumerate(transfers.loads)) {
      createDMAStart(rewriter, loc,
                     mapping.lookupOrDefault(loadViews[2 * index]),
                     mapping.lookupOrDefault(loadViews[2 * index + 1]),
                     transfer.tag, slot, transfer.shape);
    }
  };

  // Prologue: the loads of the first iteration.
  rewriter.setInsertionPoint(loop);
  if (!transfers.loads.empty())
    issueLoads(arith::ConstantIndexOp::create(rewriter, loc, lb));

  // Head: prefetch the next tile, then wait for the current one.
  //
  // A buffer that is loaded and stored (in/out) is prefetched into the slot
  // whose store from the previous iteration is only waited for at the end of
  // this one. This relies on the DMA engine processing descriptors in the
  // order they are issued: that store was issued before the prefetch, so it
  // has read the slot before the prefetch writes it.
  rewriter.setInsertionPointToStart(body);
  if (!transfers.loads.empty()) {
    Value nextIv = arith::AddIOp::create(rewriter, loc, iv, loop.getStep());
    Value hasNext = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, nextIv, loop.getUpperBound());
    auto prefetchOp = scf::IfOp::create(rewriter, loc, hasNext,
                                        /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(prefetchOp.thenBlock());
    issueLoads(nextIv);
    rewriter.setInsertionPointAfter(prefetchOp);
  }
  Value slot = createSlot(rewriter, loc, iv, lb, step);
  for (const Transfer &transfer : transfers.loads) {
    createDMAWait(rewriter, loc, transfer, slot);
    rewriter.eraseOp(transfer.copy);
  }

  // Stores: issued where the copy was, waited for one iteration later.
  for (Transfer &transfer : transfers.stores) {
    rewriter.setInsertionPoint(transfer.copy);
    createDMAStart(rewriter, loc, transfer.copy.getSource(),
                   transfer.copy.getTarget(), transfer.tag, slot,
                   transfer.shape);
    rewriter.eraseOp(transfer.copy);
  }
  if (!transfers.stores.empty()) {
    rewriter.setInsertionPoint(body->getTerminator());
    Value hasPrevious = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sgt, iv, loop.getLowerBound());
    auto waitOp = scf::IfOp::create(rewriter, loc, hasPrevious,
                                    /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(waitOp.thenBlock());
    Value previousIv = arith::SubIOp::create(rewriter, loc, iv, loop.getStep());
    Value previousSlot = createSlot(rewriter, loc, previousIv, lb, step);
    for (const Transfer &transfer : transfers.stores)
      createDMAWait(rewriter, loc, transfer, previousSlot);

    // Epilogue: the stores of the last iteration.
    rewriter.setInsertionPointAfter(loop);
    int64_t lastIv = lb + (llvm::divideCeil(ub - lb, step) - 1) * step;
    Value lastSlot = createSlot(
        rewriter, loc, arith::ConstantIndexOp::create(rewriter, loc, lastIv),
        lb, step);
    for (const Transfer &transfer : transfers.stores)
      createDMAWait(rewriter, loc, transfer, lastSlot);
  }

  // The DDR views of the loads are now only computed in the clones.
  for (Operation &op : llvm::make_early_inc_range(llvm::reverse(*body))) {
    if (isOpTriviallyDead(&op))
      rewriter.eraseOp(&op);
  }

  // LLVM would otherwise peel the first or last iteration to fold the guards,
  // duplicating the compute again.
  MLIRContext *context = rewriter.getContext();
  auto unroll = LLVM::LoopUnrollAttr::get(
      context, /*disable=*/rewriter.getBoolAttr(true), /*count=*/{},
      /*runtimeDisable=*/{}, /*full=*/{}, /*followupUnrolled=*/{},
      /*followupRemainder=*/{}, /*followupAll=*/{});
  auto annotation = LLVM::LoopAnnotationAttr::get(
      context, /*id=*/{}, /*disableNonforced=*/{}, /*vectorize=*/{},
      /*interleave=*/{}, unroll, /*unrollAndJam=*/{}, /*licm=*/{},
      /*distribute=*/{}, /*pipeline=*/{}, /*peeled=*/{}, /*unswitch=*/{},
      /*mustProgress=*/{}, /*isVectorized=*/{}, /*align=*/0,
      /*startLoc=*/{}, /*endLoc=*/{}, /*parallelAccesses=*/{});
  loop->setAttr(LLVM::getLoopAnnotationAttrName(), annotation);
  return success();
}

/// Whether `loop` runs on the thread that runs the dispatch. DMA tokens are
/// only valid on the thread that issued them.
bool runsOnDispatchThread(scf::ForOp loop) {
  for (Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isThreadForall(parent) || isa<async::ExecuteOp>(parent))
      return false;
  }
  return true;
}

struct HexagonPipelineVTCMTransfersPass
    : impl::HexagonPipelineVTCMTransfersPassBase<
          HexagonPipelineVTCMTransfersPass> {
  using Base::Base;

  void runOnOperation() override {
    // Post-order: inner tile loops are pipelined first and are then part of
    // the compute of an outer one.
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kVTCMTileLoopAttrName))
        loops.push_back(loop);
    });

    llvm::SmallPtrSet<Operation *, 4> outerLoops;
    for (scf::ForOp loop : loops) {
      for (Operation *parent = loop->getParentOp(); parent;
           parent = parent->getParentOp()) {
        if (parent->hasAttr(kVTCMTileLoopAttrName))
          outerLoops.insert(parent);
      }
    }

    IRRewriter rewriter(&getContext());
    for (scf::ForOp loop : loops) {
      loop->removeAttr(kVTCMTileLoopAttrName);
      if (!pipelineOuterTileLoops && outerLoops.contains(loop))
        continue;
      std::optional<int64_t> lb = getConstantIntValue(loop.getLowerBound());
      std::optional<int64_t> ub = getConstantIntValue(loop.getUpperBound());
      std::optional<int64_t> step = getConstantIntValue(loop.getStep());
      if (!lb || !ub || !step || *step <= 0 || !loop.getInitArgs().empty() ||
          !runsOnDispatchThread(loop))
        continue;
      // A single iteration has nothing to overlap.
      if (llvm::divideCeil(*ub - *lb, *step) < 2)
        continue;
      FailureOr<TileLoopTransfers> transfers = analyzeTileLoop(loop);
      if (failed(transfers) ||
          (transfers->loads.empty() && transfers->stores.empty()))
        continue;
      if (failed(pipelineTileLoop(rewriter, loop, *transfers, *lb, *ub, *step)))
        return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::hexagon::codegen
