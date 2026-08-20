// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ROOF_HEXAGON_HAL_HEXAGONTARGETDEVICE_H_
#define ROOF_HEXAGON_HAL_HEXAGONTARGETDEVICE_H_

#include "hexagon/Target/HexagonOptions.h"

#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"

#include <memory>
namespace mlir::iree_compiler::hexagon::target {

std::shared_ptr<mlir::iree_compiler::IREE::HAL::TargetDevice>
createHexagonTargetDevice(const HexagonOptions &options);

} // namespace mlir::iree_compiler::hexagon::target

#endif // ROOF_HEXAGON_HAL_HEXAGONTARGETDEVICE_H_
