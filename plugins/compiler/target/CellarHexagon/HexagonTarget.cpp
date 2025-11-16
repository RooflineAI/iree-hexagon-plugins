// Copyright 2020 The IREE Authors
//
// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CodeGen/HexagonCodeGenPipeline.h"

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/OptionUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;
using namespace mlir::iree_compiler::IREE::HAL;

namespace hexagon::HAL {

namespace {

// Debugging function
static void dumpModuleToPath(StringRef path, StringRef baseName,
                             StringRef suffix, StringRef extPrefix,
                             mlir::ModuleOp module) {
  // TODO: Another placeholder to help with debugging later on
  if (path.empty() || !module)
    module->emitError("Tried dumping a null module or to an empty path. How "
                      "did you mess up the debugging code dude...");

  std::string extension = (extPrefix + ".mlir").str();
  std::string textData;
  llvm::raw_string_ostream os(textData);
  mlir::OpPrintingFlags printingFlags;
  printingFlags.enableDebugInfo(true);
  module.print(os, printingFlags);

  dumpDataToPath(path, baseName, suffix, extension, textData);
}

// These options describe the available fields that are reused in the Hexagon
// TargetDevice and TargetBackend. They are available as CLI arguments obviously
struct HexagonOptions {
  std::string architecture = "79";

  void bindOptions(OptionsBinder &binder) {
    // TODO: Placeholder option for now.
    // I think we still do not know what architecture we are going to be working
    // with, placeholder option to determine it
    static llvm::cl::OptionCategory category("Hexagon HAL Target");
    binder.opt<std::string>(
        "iree-hexagon-v", architecture, llvm::cl::cat(category),
        llvm::cl::desc("Hexagon ISA version to target (e.g. 68, 69, 73, 79)."));
  }
};

// This is a custom TargetDevice for Hexagon.
// If I understand correctly, this should embed an attribute into the IR to be
// consumed by the runtime. This code is copied from the CUDATarget.cpp plugin.
// TODO: This means that I should synchronize with Stefan on what would be
// needed here, what information should I pass the runtime from the codegen side
class HexagonTargetDevice final : public TargetDevice {
public:
  explicit HexagonTargetDevice(const HexagonOptions &options)
      : options(options) {}

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const override {
    Builder builder(context);
    // Example of how to pass an additional config attr
    SmallVector<NamedAttribute> deviceConfigItems = {
        builder.getNamedAttr("hexagon.arch",
                             builder.getStringAttr(options.architecture)),
    };
    // The deviceTargetAttr requires deviceConfigAttr and executableTargetAttrs,
    // I am not asking questions about it right now.
    auto deviceConfigAttr = builder.getDictionaryAttr(deviceConfigItems);
    auto executableConfigAttr = builder.getDictionaryAttr({});

    // TODO: This is just querying the TargetBackend below. I am unsure why this
    // has to be managed here instead of querying it directly (since it is
    // embedded in the IR right?), but I am copy pasting right now. Answer this
    // question later.
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    if (auto targetBackend = targetRegistry.getTargetBackend("hexagon")) {
      targetBackend->getDefaultExecutableTargets(
          context, "hexagon", executableConfigAttr, executableTargetAttrs);
    } else {
      llvm::errs() << "Hexagon target backend not registered; unable to build "
                      "default device target\n";
      return {};
    }

    return IREE::HAL::DeviceTargetAttr::get(
        context, builder.getStringAttr("hexagon"), deviceConfigAttr,
        executableTargetAttrs);
  }

  // TODO: This is also a placeholder and is currently incomplete.
  // This is just meant for verification so I should be able to ignore it for
  // now...
  // Should: "Builds an expression that returns an i1 indicating whether
  // the given |device| matches the |targetAttr| requirements."
  Value buildDeviceTargetMatch(Location loc, Value device,
                               IREE::HAL::DeviceTargetAttr targetAttr,
                               OpBuilder &builder) const override {
    return IREE::HAL::DeviceTargetAttr::buildDeviceIDAndExecutableFormatsMatch(
        loc, device, "hexagon*", targetAttr.getExecutableTargets(), builder);
  }

private:
  const HexagonOptions &options;
};

class HexagonTargetBackend final : public TargetBackend {
public:
  HexagonTargetBackend(const HexagonOptions &options) : options(options) {}

  // TODO: Ask Stefan about this
  // from his draft pr, I think this is the expected name
  // https://app.graphite.dev/github/pr/RooflineAI/roof-mlir/847/feat(hexagon)-runtime-skeleton-for-Hexagon#file-plugins/runtime/hexagon/registration/driver_module.c
  // Can currently be tested with :
  // iree-compile --iree-plugin=hal_target_hexagon --iree-hexagon-v=79
  // --iree-hal-target-backends=hexagon --iree-hal-target-device=hexagon
  // ./check_correct_registration.mlir -o erase.vmfb
  std::string getLegacyDefaultDeviceID() const override { return "hexagon"; }

  // TODO: Decide what is the necessary information that should be embedded in
  // the IR
  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {
    executableTargetAttrs.push_back(getExecutableTarget(context));
  }

  // TODO: Check claim:
  // This is supposed to be extracting information embedded in the IR under
  // hexagon.arch and is meant to provide information for the HAL
  IREE::HAL::ExecutableTargetAttr
  getExecutableTarget(MLIRContext *context) const {
    Builder builder(context);
    SmallVector<NamedAttribute> configItems = {
        builder.getNamedAttr("hexagon.arch",
                             builder.getStringAttr(options.architecture)),
    };

    return builder.getAttr<IREE::HAL::ExecutableTargetAttr>(
        builder.getStringAttr("hexagon"), builder.getStringAttr("hexagon-llvm"),
        builder.getDictionaryAttr(configItems));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // TODO: Decide on the needed dialects for hexagon
    // This can be entirely removed though
  }

  void
  buildConfigurationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                 OpPassManager &passManager) override {
    (void)targetAttr;
    cellar::target::hexagon::buildHexagonConfigurationPassPipeline(passManager);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                    OpPassManager &passManager) override {
    (void)targetAttr;
    cellar::target::hexagon::buildHexagonTranslationPassPipeline(passManager);
  }

  void buildLinkingPassPipeline(OpPassManager &passManager) override {
    cellar::target::hexagon::buildHexagonLinkingPassPipeline(passManager);
  }

  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    // TODO: Here is the main function from TargetBackend
    // Here is where we are supposed to create our output
    // We are supposed to create a .vmfb that should contain:
    // .so, constants.bin and .fb
    // For more info, check here:
    // https://linear.app/roofline/document/luis-meeting-notes-fec2bd974e4a

    // Discussion on how to finish the plumbing with the SDK should also be
    // managed here

    // TODO: Placeholder for debugging later on
    if (!options.dumpIntermediatesPath.empty()) {
      dumpModuleToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                       variantOp.getName(), ".codegen",
                       variantOp.getInnerModule());
    }

    // Right now, trying to execute this function and returning a success
    // results in some incomprehensible error. I tried passing dummy binaries
    // (handwritten) or linalg operations, but both yielded different errors.
    // Maybe I will not be able to test this until I have an actual working
    // lowering pipeline.

    return variantOp.emitOpError("Hexagon serialization not implemented yet");
  }

private:
  const HexagonOptions &options;
};

// The plugin session simply takes care of registering all the extensions from
// the plugin
struct HexagonSession
    : public PluginSession<HexagonSession, HexagonOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    cellar::target::hexagon::registerHexagonCodeGenPasses();
  };

  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) override {
    targets.add("hexagon", [=, this]() {
      return std::make_shared<HexagonTargetDevice>(options);
    });
  }

  void
  populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) override {
    targets.add("hexagon", [=, this]() {
      return std::make_shared<HexagonTargetBackend>(options);
    });
  }
};

} // namespace

} // namespace hexagon::HAL

IREE_DEFINE_COMPILER_OPTION_FLAGS(hexagon::HAL::HexagonOptions);

extern "C" bool iree_register_compiler_plugin_hal_target_hexagon(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<hexagon::HAL::HexagonSession>("hal_target_hexagon");
  return true;
}
