// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/Target/HexagonTargetDevice.h"

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler::hexagon::target {
namespace HAL = mlir::iree_compiler::IREE::HAL;

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

} // namespace mlir::iree_compiler::hexagon::target
