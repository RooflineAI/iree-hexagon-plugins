// Copyright 2024 The IREE Authors
// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Derived from Codegen/Dialect/CPU/IR/IREECPUAttrs.cpp at IREE revision
// a45adeaa6115e446c898e6eb21fb6edc0e65ddc4.

#include "hexagon/CodeGen/IR/HexagonAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"

#include <array>

namespace mlir::iree_compiler::IREE::Hexagon {

static constexpr std::array<StringLiteral, 6> tilingLevelNames = {
    "distribution",           "cache_parallel",   "cache_reduction",
    "vector_common_parallel", "vector_reduction", "vector_inner_parallel"};

SmallVector<int> getTilingLevelsAsInts() {
  return llvm::to_vector(llvm::seq<int>(0, tilingLevelNames.size()));
}

StringRef getTilingLevelName(TilingLevel level) {
  unsigned index = static_cast<unsigned>(level);
  return index < tilingLevelNames.size() ? StringRef(tilingLevelNames[index])
                                         : StringRef();
}

LogicalResult
LoweringConfigAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           DictionaryAttr config) {
  std::optional<size_t> rank;
  for (NamedAttribute item : config) {
    if (!llvm::is_contained(tilingLevelNames, item.getName().getValue()))
      return emitError() << "unknown Hexagon tiling stage " << item.getName();
    auto level =
        dyn_cast<Codegen::LoweringConfigTilingLevelAttr>(item.getValue());
    if (!level)
      return emitError() << item.getName()
                         << " must contain a Codegen tiling level";
    if (llvm::any_of(level.getSizes(), [](int64_t size) { return size < 0; }))
      return emitError() << "expected nonnegative tile sizes for "
                         << item.getName();
    if (llvm::is_contained(level.getScalableFlags(), true) ||
        !level.getInterchange().empty())
      return emitError() << "expected fixed tile sizes without interchange for "
                         << item.getName();
    if (rank && *rank != level.getSizes().size())
      return emitError()
             << "expected all Hexagon tiling stages to have the same rank";
    rank = level.getSizes().size();
  }
  return success();
}

Attribute LoweringConfigAttr::parse(AsmParser &parser, Type) {
  SmallVector<NamedAttribute> items;
  if (parser.parseCommaSeparatedList(
          AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
            std::string key;
            SmallVector<int64_t> sizes;
            if (parser.parseKeywordOrString(&key) || parser.parseEqual() ||
                parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                               [&]() {
                                                 int64_t size;
                                                 if (parser.parseInteger(size))
                                                   return failure();
                                                 sizes.push_back(size);
                                                 return success();
                                               }))
              return failure();
            if (llvm::any_of(items, [&](NamedAttribute item) {
                  return item.getName().getValue() == key;
                }))
              return parser.emitError(parser.getCurrentLocation(),
                                      "duplicate Hexagon tiling stage ")
                     << key;
            items.emplace_back(StringAttr::get(parser.getContext(), key),
                               getTilingLevelAttr(parser.getContext(), sizes));
            return success();
          }))
    return {};
  return parser.getChecked<LoweringConfigAttr>(
      parser.getContext(), DictionaryAttr::get(parser.getContext(), items));
}

void LoweringConfigAttr::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::interleaveComma(getConfig(), printer, [&](NamedAttribute item) {
    printer << item.getName().getValue() << " = [";
    llvm::interleaveComma(
        cast<Codegen::LoweringConfigTilingLevelAttr>(item.getValue())
            .getSizes(),
        printer);
    printer << "]";
  });
  printer << ">";
}

LoweringConfigAttr LoweringConfigAttr::get(MLIRContext *ctx,
                                           SmallVector<NamedAttribute> items) {
  return get(ctx, DictionaryAttr::get(ctx, items));
}

Attribute LoweringConfigAttr::getTilingLevelAttr(MLIRContext *ctx,
                                                 ArrayRef<int64_t> tileSizes) {
  return Codegen::LoweringConfigTilingLevelAttr::get(ctx, tileSizes, {}, {});
}

Attribute LoweringConfigAttr::getTilingLevelAttr(unsigned level) const {
  StringRef key = getTilingLevelName(static_cast<TilingLevel>(level));
  return key.empty() ? Attribute() : getConfig().get(key);
}

bool LoweringConfigAttr::hasTilingLevel(unsigned level) const {
  return bool(getTilingLevelAttr(level));
}

bool LoweringConfigAttr::hasWorkgroupTilingLevel() const {
  return hasTilingLevel(static_cast<unsigned>(TilingLevel::DistributionTiles));
}

std::optional<unsigned> LoweringConfigAttr::getNumTilingLevels() const {
  return getConfig().size();
}

SmallVector<int64_t>
LoweringConfigAttr::getStaticTilingLevelSizes(unsigned level,
                                              Operation *) const {
  auto attr = dyn_cast_or_null<Codegen::LoweringConfigTilingLevelAttr>(
      getTilingLevelAttr(level));
  return attr ? SmallVector<int64_t>(attr.getSizes()) : SmallVector<int64_t>();
}

SmallVector<int64_t> LoweringConfigAttr::getWorkgroupTileSizes() const {
  return getStaticTilingLevelSizes(
      static_cast<unsigned>(TilingLevel::DistributionTiles), nullptr);
}

SmallVector<OpFoldResult>
LoweringConfigAttr::getTilingLevelSizes(OpBuilder &builder, unsigned level,
                                        Operation *op) const {
  return llvm::map_to_vector(
      getStaticTilingLevelSizes(level, op),
      [&](int64_t size) -> OpFoldResult { return builder.getIndexAttr(size); });
}

std::optional<SmallVector<int64_t>> LoweringConfigAttr::getVectorSizes() const {
  SmallVector<int64_t> result;
  constexpr std::array vectorLevels{TilingLevel::VectorCommonParallelTiles,
                                    TilingLevel::VectorReductionTiles,
                                    TilingLevel::VectorInnerParallelTiles};
  for (TilingLevel level : vectorLevels) {
    auto sizes =
        getStaticTilingLevelSizes(static_cast<unsigned>(level), nullptr);
    if (sizes.empty())
      continue;
    if (result.empty())
      result.resize(sizes.size(), 0);
    if (result.size() != sizes.size())
      return std::nullopt;
    for (auto [index, size] : llvm::enumerate(sizes)) {
      if (size == 0)
        continue;
      if (result[index] != 0)
        return std::nullopt;
      result[index] = size;
    }
  }
  return result;
}

} // namespace mlir::iree_compiler::IREE::Hexagon
