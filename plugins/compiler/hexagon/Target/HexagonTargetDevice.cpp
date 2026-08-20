// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/Target/HexagonTargetDevice.h"

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler::cellar_hexagon::target {
namespace HAL = mlir::iree_compiler::IREE::HAL;

// This is a custom TargetDevice for Hexagon.
// This embeds an attribute into the IR containing information for the runtime.
// Right now, this is modeled after LLVMCPUTarget and only contains a reference
// to the TargetBackend attribute and the name of the target device.
class HexagonTargetDevice final : public HAL::TargetDevice {
public:
  explicit HexagonTargetDevice(const HexagonOptions &options)
      : options(options) {}

  HAL::DeviceTargetAttr getDefaultDeviceTarget(
      mlir::MLIRContext *context,
      const HAL::TargetRegistry &targetRegistry) const override {
    mlir::Builder builder(context);
    auto executableConfigAttr = builder.getDictionaryAttr({});
    llvm::SmallVector<HAL::ExecutableTargetAttr> executableTargetAttrs;

    if (auto targetBackend = targetRegistry.getTargetBackend("hexagon")) {
      targetBackend->getDefaultExecutableTargets(
          context, "hexagon", executableConfigAttr, executableTargetAttrs);
    } else {
      llvm::errs() << "Hexagon target backend not registered; unable to build "
                      "default device target\n";
      return {};
    }

    // Note that we currently have no additional configuration information.
    return HAL::DeviceTargetAttr::get(context, builder.getStringAttr("hexagon"),
                                      {}, executableTargetAttrs);
  }

  // TODO: This is also a placeholder and is currently incomplete.
  // This is just meant for verification so I should be able to ignore it for
  // now...
  // Should: "Build an expression that returns an i1 indicating whether
  // the given |device| matches the |targetAttr| requirements."
  mlir::Value buildDeviceTargetMatch(mlir::Location loc, mlir::Value device,
                                     HAL::DeviceTargetAttr targetAttr,
                                     mlir::OpBuilder &builder) const override {
    return HAL::DeviceTargetAttr::buildDeviceIDAndExecutableFormatsMatch(
        loc, device, "hexagon*", targetAttr.getExecutableTargets(), builder);
  }

private:
  const HexagonOptions &options;
};

std::shared_ptr<HAL::TargetDevice>
createHexagonTargetDevice(const HexagonOptions &options) {
  return std::make_shared<HexagonTargetDevice>(options);
}

} // namespace mlir::iree_compiler::cellar_hexagon::target
