// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/Target/HexagonTargetBackend.h"

#include "cellar-hexagon/CodeGen/IR/HexagonAttrs.h"
#include "cellar-hexagon/CodeGen/IR/HexagonDialect.h"
#include "cellar-hexagon/CodeGen/Pipelines/ConfigurationPipeline.h"
#include "cellar-hexagon/CodeGen/Pipelines/LinkingPipeline.h"
#include "cellar-hexagon/CodeGen/Pipelines/TranslationPipeline.h"
#include "cellar-hexagon/Target/HexagonExecutableSerialization.h"
#include "cellar-hexagon/Target/HexagonLLVMTarget.h"
#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include <optional>
#include <string>

namespace mlir::iree_compiler::cellar_hexagon::target {
namespace IREE = mlir::iree_compiler::IREE;
namespace HAL = mlir::iree_compiler::IREE::HAL;

class HexagonTargetBackend final : public HAL::TargetBackend {
public:
  explicit HexagonTargetBackend(const HexagonOptions &options)
      : hexagonOptions(options) {}

  std::string getLegacyDefaultDeviceID() const override { return "hexagon"; }

  void getDefaultExecutableTargets(
      mlir::MLIRContext *context, llvm::StringRef deviceID,
      mlir::DictionaryAttr deviceConfigAttr,
      llvm::SmallVectorImpl<HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {
    executableTargetAttrs.push_back(getExecutableTarget(context));
  }

  // This generates an attr that will be embedded in the IR and that
  // is used by some of the passes
  //
  // One of the informations we have to embed into the IR is the encoding.
  // It consists of metadata describing layout and tiling information.
  //
  // This works in tandem with the resolver, who is in charge of using this
  // additional information to generate the tiling.
  //
  // This is needed, even for prototyping simple scalar backends, because
  // otherwise linalg operations do not get lowered in passes such as
  // materializeEncodings.
  HAL::ExecutableTargetAttr
  getExecutableTarget(mlir::MLIRContext *context) const {
    mlir::Builder builder(context);

    auto target = createLLVMTargetForHexagon(hexagonOptions);

    llvm::SmallVector<mlir::NamedAttribute> configItems;
    target.storeToConfigAttrs(context, configItems);

    configItems.emplace_back(builder.getNamedAttr(
        "hexagon.version", builder.getStringAttr(hexagonOptions.version)));

    // I tried using the already existing CPUEncodingResolverAttr. This does not
    // play properly with the custom target backend, so creating a custom one is
    // needed, even for prototyping
    configItems.emplace_back(
        builder.getStringAttr(IREE::Encoding::kEncodingResolverAttrName),
        IREE::Hexagon::HexagonEncodingResolverAttr::get(context, {}));

    // The first two attributes are only identifiers if I am not wrong.
    // The second one interacts with the HAL. It follows the pattern
    // [loader]-[format]-[arch]. Check LLVMCPUTarget:188 for the example this is
    // based on.
    // Might be changed in the future to qurt-elf-hexagon
    return builder.getAttr<HAL::ExecutableTargetAttr>(
        builder.getStringAttr("hexagon"),
        builder.getStringAttr("embedded-elf-hexagon"),
        builder.getDictionaryAttr(configItems));
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, IREE::Hexagon::IREEHexagonDialect,
                    mlir::hexkl::HexKLDialect,
                    mlir::hexagonmem::HexagonMemDialect>();
    registerBuiltinDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
  }

  // These are hooks into the HAL managed pipelines.
  void
  buildConfigurationPassPipeline(HAL::ExecutableTargetAttr targetAttr,
                                 mlir::OpPassManager &passManager) override {
    (void)targetAttr;
    mlir::iree_compiler::cellar_hexagon::codegen::
        buildHexagonConfigurationPassPipeline(passManager);
  }

  void buildTranslationPassPipeline(HAL::ExecutableTargetAttr targetAttr,
                                    mlir::OpPassManager &passManager) override {
    (void)targetAttr;
    mlir::iree_compiler::cellar_hexagon::codegen::
        buildHexagonTranslationPassPipeline(passManager);
  }

  void buildLinkingPassPipeline(mlir::OpPassManager &passManager) override {
    // Passing the backend name ensures the link pass actually gathers
    // Hexagon executables instead of skipping with an empty target filter.
    mlir::iree_compiler::cellar_hexagon::codegen::
        buildHexagonLinkingPassPipeline(
            passManager, std::make_optional<std::string>("hexagon"));
  }

  mlir::LogicalResult
  serializeExecutable(const SerializationOptions &options,
                      HAL::ExecutableVariantOp variantOp,
                      mlir::OpBuilder &executableBuilder) override {
    return serializeHexagonExecutable(hexagonOptions, options, variantOp,
                                      executableBuilder);
  }

private:
  const HexagonOptions &hexagonOptions;
};

std::shared_ptr<HAL::TargetBackend>
createHexagonTargetBackend(const HexagonOptions &options) {
  return std::make_shared<HexagonTargetBackend>(options);
}

} // namespace mlir::iree_compiler::cellar_hexagon::target
