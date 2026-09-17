// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_CODEGEN_IR_HMXCONTRACTS_H_
#define ROOF_HEXAGON_CODEGEN_IR_HMXCONTRACTS_H_

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include <array>
#include <optional>

namespace mlir {
class Operation;

namespace iree_compiler::IREE::Hexagon {

// TODO: Think about a better way to setup these constants other than hardcoded
// values.
inline constexpr int64_t kHmxLogicalTileSize = 32;
inline constexpr int64_t kHmxTileRows = 16;
inline constexpr int64_t kHmxTileColumns = 32;
inline constexpr int64_t kHmxTileInterleave = 2;
inline constexpr int64_t kHmxConfigBytes = 256;

/// Rank of one physical HMX tile ([16, 32, 2]) and of a tile grid
/// ([row_tiles, column_tiles, 16, 32, 2]). Both forms appear on the
/// buffer-level operations: tiling reduces a grid to a single tile without
/// changing rank, while expansion and unpacking also accept the already
/// rank-reduced tile.
inline constexpr int64_t kHmxTileRank = 3;
inline constexpr int64_t kHmxTileGridRank = 5;

/// Base-address alignment, in bytes, that every HMX instruction requires of the
/// VTCM buffers it addresses: packed tile grids, accumulator read-out tiles,
/// and the accumulator-read config block. `HexagonExpandHmxMatmulPass` creates
/// the allocations that must satisfy this, and `HexagonLowerHmxToCallsPass`
/// rejects the ones it cannot prove, so the two must agree on a single value.
inline constexpr int64_t kHmxAlignment = 2048;

/// Innermost three strides, in elements, of a contiguous physical [16, 32, 2]
/// HMX tile. The runtime kernels address tiles through these implicit strides
/// rather than reading them from a descriptor.
inline constexpr std::array<int64_t, 3> kHmxTileStrides = {
    kHmxTileColumns * kHmxTileInterleave, kHmxTileInterleave, 1};

/// Number of whole 32x32 logical tiles needed to cover `elements`.
constexpr int64_t getHmxTileCount(int64_t elements) {
  return (elements + kHmxLogicalTileSize - 1) / kHmxLogicalTileSize;
}

/// Number of logical elements covered by `tiles` whole 32x32 tiles.
constexpr int64_t getHmxElementCapacity(int64_t tiles) {
  return tiles * kHmxLogicalTileSize;
}

/// Returns whether `type` has the statically shaped physical HMX tile suffix
/// [16, 32, 2]. This does not validate memory layout or alignment.
bool hasStaticHmxTileSuffix(ShapedType type);

/// Returns whether `type` is a rank-5 HMX tile grid rather than a rank-3 single
/// physical tile. Only meaningful once the rank is known to be one of the two.
bool isHmxTileGrid(ShapedType type);

/// Returns whether the leading two tile-grid dimensions of `type` are static
/// and positive. Runtime lowering derives the grid capacity from them, so they
/// may not be dynamic even when the logical extents they cover are.
bool hasPositiveStaticTileGrid(ShapedType type);

/// Returns whether `type` is an f16 HMX tile: a rank-3 physical [16, 32, 2]
/// tile when `requireGrid` is false, or a rank-5 grid of them with positive
/// static tile counts when it is true. This is the single definition of the
/// tile shape contract; the `verify*` entry points below only choose how to
/// report it.
bool isHmxF16Tile(ShapedType type, bool requireGrid);

/// Returns whether `type` is exactly !iree_hexagon.hmx.acc<32x32xf32>.
bool isHmxAccumulator(Type type);

/// Verifies `isHmxF16Tile`, reporting which part of the contract failed. Used
/// where several tile forms are legal and the distinction helps the reader.
LogicalResult verifyHmxF16Tile(Operation *op, ShapedType type, StringRef role,
                               bool requireGrid);

/// Verifies that `type` holds exactly one f16 HMX tile, reporting a single
/// combined diagnostic. When `allowSingletonGrid` is true a rank-5 grid of one
/// tile is also accepted. Used by the primitives that have only one legal
/// operand shape, where naming it outright beats a staged diagnostic.
LogicalResult verifyHmxSingleF16Tile(Operation *op, ShapedType type,
                                     StringRef role, bool allowSingletonGrid);

/// Verifies the shape and element-type contract shared by tensor and buffer HMX
/// pack operations. Dynamic logical dimensions are accepted; proving that they
/// fit the static physical grid is deferred to the runtime-lowering boundary.
LogicalResult verifyHmxPackContract(Operation *op, ShapedType sourceType,
                                    ShapedType destType, int64_t dim);

/// Verifies the shape, element-type, and M/K/N grid relationships of an HMX
/// matmul. When `allowSingleTileAccumulator` is true, `accType` may be the
/// rank-3 physical accumulator tile used by the buffer form.
LogicalResult verifyHmxMatmulContract(Operation *op, ShapedType lhsType,
                                      ShapedType rhsType, ShapedType accType,
                                      bool allowSingleTileAccumulator);

/// Verifies the shape and element-type contract shared by tensor and buffer HMX
/// unpack operations. Dynamic logical dimensions are accepted; proving that
/// they fit the static physical grid is deferred to the runtime-lowering
/// boundary.
LogicalResult verifyHmxUnpackContract(Operation *op, ShapedType sourceType,
                                      ShapedType destType, int64_t dim);

//===----------------------------------------------------------------------===//
// Buffer placement contract.
//
// Where a VTCM buffer's base address comes from. Two passes reason about this:
// `HexagonExpandHmxMatmulPass` has to *produce* HMX
// operands whose base address is `kHmxAlignment`-aligned, and
// `HexagonLowerHmxToCallsPass` has to *prove* that they are before handing raw
// pointers to the runtime kernels.
//
// Unlike the shape contracts above, these are not usable from an operation
// verifier: they inspect the surrounding IR and should not be run at any time.
//===----------------------------------------------------------------------===//

/// If `value` is produced by a memref operation that only reshapes or restricts
/// a buffer (cast, collapse_shape, expand_shape, subview), returns the buffer
/// it views. Returns a null Value otherwise, including for allocations,
/// `memref.assume_alignment` (which is an alignment source, not a transparent
/// view), block arguments, and operations this model does not cover.
///
/// A view's base address is derived from its source, so every caller must
/// follow the same chain: proving alignment means walking to the allocation,
/// and repairing alignment means replacing that allocation rather than the
/// view.
Value getHmxMemRefViewSource(Value value);

/// Walks `getHmxMemRefViewSource` to the buffer that actually owns the storage
/// behind `value`. Returns `value` itself when it is not a view.
Value getHmxAllocationBase(Value value);

/// Returns the alignment guaranteed by an allocation-like operation
/// (`memref.alloc`, `memref.alloca`, `hexagonmem.alloc`, or an explicit
/// `memref.assume_alignment`). Returns nullopt when `value` is not defined by
/// such an operation at all; an allocation that declares no alignment yields 0.
/// This looks only at the defining operation; use `getHmxAllocationBase` first
/// to see through views.
std::optional<int64_t> getDeclaredHmxAlignment(Value value);

/// Size in bytes of an element type the HMX runtime ABI can address, or 0 for
/// any other type. Pointer arithmetic and alignment reasoning both fail closed
/// on 0 rather than guessing a width.
unsigned getHmxElementSizeBytes(Type elementType);

/// Proves that the base address of `value` is a multiple of
/// `requiredAlignment`. Follows the same view chain as `getHmxAllocationBase`,
/// additionally accounting for the byte offset that a `memref.subview` adds;
/// dynamic offsets are proven through the integer-divisibility analysis carried
/// by `solver`, and anything unproven fails closed.
bool providesHmxAlignment(Value value, int64_t requiredAlignment,
                          DataFlowSolver &solver);

} // namespace iree_compiler::IREE::Hexagon
} // namespace mlir

#endif // ROOF_HEXAGON_CODEGEN_IR_HMXCONTRACTS_H_
