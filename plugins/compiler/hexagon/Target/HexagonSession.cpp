// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/Target/HexagonSession.h"

#include "hexagon/CodeGen/Encoding/HexagonEncodingExternalModels.h"
#include "hexagon/CodeGen/IR/HexagonDialect.h"
#include "hexagon/CodeGen/Passes.h"
#include "hexagon/Target/HexagonOptions.h"
#include "hexagon/Target/HexagonTargetBackend.h"
#include "hexagon/Target/HexagonTargetDevice.h"

#include "hexagon/Dialect/HexKL/Transforms/BufferizableOpInterfaceImpl.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/OptionUtils.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"

namespace mlir::iree_compiler::hexagon::target {
namespace {
namespace HAL = mlir::iree_compiler::IREE::HAL;
namespace IREE = mlir::iree_compiler::IREE;

// The plugin session simply takes care of registering all the extensions from
// the plugin
struct HexagonSession
    : public mlir::iree_compiler::PluginSession<
          HexagonSession, HexagonOptions,
          mlir::iree_compiler::PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    mlir::iree_compiler::hexagon::codegen::
        registerHexagonCodeGenPasses();
  };

  // For some reason, trying to override registerGlobalDialects through the CRTP
  // does not work, so this is an alternative. Not sure about the difference.
  void onRegisterDialects(mlir::DialectRegistry &registry) override {
    // MLIR hooks
    registry.insert<IREE::Hexagon::IREEHexagonDialect,
                    mlir::hexagonmem::HexagonMemDialect>();

    // Hexagon-mlir hooks
    // Reusing hexagon-mlir external models for bufferization
    mlir::hexkl::registerBufferizableOpInterfaceExternalModels(registry);
    memref::registerAllocationOpInterfaceExternalModels(registry);
    arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferDeallocationOpInterfaceExternalModels(registry);

    // IREE hooks
    mlir::iree_compiler::hexagon::codegen::
        registerHexagonEncodingExternalModels(registry);
  }

  void populateHALTargetDevices(HAL::TargetDeviceList &targets) override {
    targets.add("hexagon",
                [=, this]() { return createHexagonTargetDevice(options); });
  }

  void populateHALTargetBackends(HAL::TargetBackendList &targets) override {
    targets.add("hexagon",
                [=, this]() { return createHexagonTargetBackend(options); });
  }
};

} // namespace

bool registerHexagonPlugin(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<HexagonSession>("hexagon");
  return true;
}

} // namespace mlir::iree_compiler::hexagon::target

IREE_DEFINE_COMPILER_OPTION_FLAGS(
    ::mlir::iree_compiler::hexagon::target::HexagonOptions);
