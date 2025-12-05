// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "CodeGen/HexagonCodeGenPipeline.h"
#include "CodeGen/HexagonEncodingExternalModels.h"
#include "CodeGen/HexagonLinkerTool.h"
#include "CodeGen/IR/HexagonEncodingAttrs.h"
#include "CodeGen/IR/HexagonEncodingDialect.h"

#include "compiler/plugins/target/LLVMCPU/LLVMTargetOptions.h"
#include "compiler/plugins/target/LLVMCPU/LibraryBuilder.h"
#include "compiler/plugins/target/LLVMCPU/LinkerTool.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/OptionUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"

// TODO: There is a lot of code that is calling on functions from the
// LLVMCPUTarget plugin, especially during linking. This also includes other
// files inside the hexagon plugin. This will have to be revisited in the future
// since it is not particularly clean...

// Please do not add the llvm namespace or the code becomes illegible
using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE::HAL;

namespace hexagon::HAL {

static constexpr char kQueryFunctionName[] =
    "iree_hal_executable_library_query";

static void dumpMLIRModuleToPath(StringRef path, StringRef baseName,
                                 StringRef suffix, StringRef extPrefix,
                                 ModuleOp module) {
  std::string extension = (extPrefix + ".mlir").str();
  std::string textData;
  llvm::raw_string_ostream os(textData);
  OpPrintingFlags printingFlags;
  printingFlags.enableDebugInfo(true);
  // These two flags are meant to avoid heisenbugs related to MLIR's module
  // printing.
  // MLIR recommends printing only with multi-threading disabled, see
  // https://mlir.llvm.org/docs/PassManagement/#ir-printing
  //
  // When multi-threading is enabled the class AsmPrinter.cpp walks through the
  // the same data in parallel or may alternatively trigger multiple
  // verifications at the same time. These flags avoid any traversal through
  // shared state between threads. It is unclear to me where data is modified
  // and not only read in the AsmPrinter.cpp, but this looks like it makes the
  // bug disappear. If you find the line modifying the data, please ping for my
  // curiosity.
  printingFlags.assumeVerified(true);
  printingFlags.useLocalScope(true);
  module.print(os, printingFlags);

  dumpDataToPath(path, baseName, suffix, extension, textData);
}

static void dumpLLVMModuleToPath(StringRef path, StringRef baseName,
                                 StringRef suffix, llvm::Module &module) {
  llvm::SmallVector<char, 0> textData;
  llvm::raw_svector_ostream ostream(textData);
  module.print(ostream, nullptr);
  dumpDataToPath(path, baseName, suffix, ".ll",
                 StringRef(textData.data(), textData.size()));

  // Dump bitcode to path.
  llvm::SmallVector<char> binaryData;
  llvm::raw_svector_ostream binaryOstream(binaryData);
  // Write the specified module to the specified output stream.
  llvm::WriteBitcodeToFile(module, binaryOstream);
  dumpDataToPath(path, baseName, suffix, ".bc",
                 StringRef(binaryData.data(), binaryData.size()));
}

static void dumpAssemblyFromLLVMModule(ExecutableVariantOp variantOp,
                                       llvm::Module &llvmModule,
                                       llvm::TargetMachine &targetMachine,
                                       StringRef path, StringRef baseName) {
  llvm::SmallVector<char, 0> asmDataStorage;
  llvm::raw_svector_ostream asmStream(asmDataStorage);
  llvm::legacy::PassManager asmPassManager;
  auto asmModule = llvm::CloneModule(llvmModule);
  if (targetMachine.addPassesToEmitFile(asmPassManager, asmStream, nullptr,
                                        llvm::CodeGenFileType::AssemblyFile)) {
    variantOp.emitOpError()
        << "Hexagon target machine cannot emit assembly files";
  }
  asmPassManager.run(*asmModule);
  std::vector<int8_t> asmData(asmDataStorage.begin(), asmDataStorage.end());
  dumpDataToPath<int8_t>(path, baseName, variantOp.getName(), ".s", asmData);
}

// Registers all LLVM components required for Hexagon code generation.
static void initializeHexagonTarget() {
  static std::once_flag init;
  std::call_once(init, []() {
    LLVMInitializeHexagonTargetInfo();
    LLVMInitializeHexagonTarget();
    LLVMInitializeHexagonTargetMC();
    LLVMInitializeHexagonAsmPrinter();
    LLVMInitializeHexagonAsmParser();
    LLVMInitializeHexagonDisassembler();
  });
}

// These options describe the available fields that are reused in the Hexagon
// TargetDevice and TargetBackend. They are available as CLI arguments obviously
struct HexagonOptions {
  std::string version = "79";
  std::string features = "";
  std::string linker = "";

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("Hexagon HAL Target");

    binder.opt<std::string>(
        "iree-hexagon-v", version, llvm::cl::cat(category),
        llvm::cl::desc("Hexagon ISA version to target (e.g. 68, 69, 73, 79)."));

    // This is just passed raw to the LLVM backend for now
    binder.opt<std::string>(
        "iree-hexagon-features", features, llvm::cl::cat(category),
        llvm::cl::desc(
            "Hexagon features supported to be passed to the llvm backend (e.g. "
            "+hvxv79,+hvx-length128b). Use llc to determine other options."));

    binder.opt<std::string>(
        "iree-hexagon-linker-path", linker, llvm::cl::cat(category),
        llvm::cl::desc("Hexagon linker tool path to use during serialization. "
                       "Currently supported linkers are lld and hexagon-clang "
                       "(available in Hexagon's SDK)"));
  }
};

// This is a custom TargetDevice for Hexagon.
// This embeds an attribute into the IR containing information for the runtime.
// Right now, this is modeled after LLVMCPUTarget and only contains a reference
// to the TargetBackend attribute and the name of the target device.
class HexagonTargetDevice final : public TargetDevice {
public:
  explicit HexagonTargetDevice(const HexagonOptions &options)
      : options(options) {}

  DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const override {
    Builder builder(context);
    auto executableConfigAttr = builder.getDictionaryAttr({});
    SmallVector<ExecutableTargetAttr> executableTargetAttrs;

    if (auto targetBackend = targetRegistry.getTargetBackend("hexagon")) {
      targetBackend->getDefaultExecutableTargets(
          context, "hexagon", executableConfigAttr, executableTargetAttrs);
    } else {
      llvm::errs() << "Hexagon target backend not registered; unable to build "
                      "default device target\n";
      return {};
    }

    // Note that we currently have no additional configuration information.
    return DeviceTargetAttr::get(context, builder.getStringAttr("hexagon"), {},
                                 executableTargetAttrs);
  }

  // TODO: This is also a placeholder and is currently incomplete.
  // This is just meant for verification so I should be able to ignore it for
  // now...
  // Should: "Build an expression that returns an i1 indicating whether
  // the given |device| matches the |targetAttr| requirements."
  Value buildDeviceTargetMatch(Location loc, Value device,
                               DeviceTargetAttr targetAttr,
                               OpBuilder &builder) const override {
    return DeviceTargetAttr::buildDeviceIDAndExecutableFormatsMatch(
        loc, device, "hexagon*", targetAttr.getExecutableTargets(), builder);
  }

private:
  const HexagonOptions &options;
};

// This function creates a LLVM target for Hexagon. This is different
// from LLVMCPU since this logic is usually managed by multiple
// classes that do necessary adjustments depending on host machine or
// cross-compilation options. The cleanest way of implementing this would be to
// extend the LLVMCPUTarget functions managing this to support Hexagon, but this
// wraps around some of those calls instead to avoid modifying code outside this
// plugin
static LLVMTarget createLLVMTargetForHexagon(const HexagonOptions &options) {
  constexpr StringRef triple = "hexagon-unknown-unknown-elf";
  std::string cpuName = std::string("hexagonv") + options.version;
  ResolveCPUAndCPUFeaturesStatus status;

  // FIXME: This calls resolveCPUAndCPUFeatures that will fail because hexagon
  // is not registered as an LLVMTarget in IREE. Since I am just prototyping, I
  // will ignore the failed status and manually input the necessary info
  // (hardcoded) in the target (dataLayout and vectorWidth). The correct way
  // of doing this would be to update IREE I guess.
  auto targetOption =
      LLVMTarget::create(triple, cpuName, options.features, false, status);
  if (!targetOption)
    llvm::errs() << "Failed to define default LLVMTarget for Hexagon";
  auto target = targetOption.value();

  target.dataLayout =
      "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:"
      "32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-"
      "v2048:2048:2048";

  // TODO: This something expected by CPU passes, but does not really make
  // that much sense in the context of Hexagon. Passes dependent on it should
  // be later revisited.
  target.vectorWidthInBytes = 32;

  return target;
}

class HexagonTargetBackend final : public TargetBackend {
public:
  HexagonTargetBackend(const HexagonOptions &options)
      : hexagonOptions(options) {}

  std::string getLegacyDefaultDeviceID() const override { return "hexagon"; }

  void getDefaultExecutableTargets(MLIRContext *context, StringRef deviceID,
                                   DictionaryAttr deviceConfigAttr,
                                   SmallVectorImpl<ExecutableTargetAttr>
                                       &executableTargetAttrs) const override {
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
  ExecutableTargetAttr getExecutableTarget(MLIRContext *context) const {
    Builder builder(context);

    auto target = createLLVMTargetForHexagon(hexagonOptions);

    SmallVector<NamedAttribute> configItems;
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
    return builder.getAttr<ExecutableTargetAttr>(
        builder.getStringAttr("hexagon"),
        builder.getStringAttr("embedded-elf-hexagon"),
        builder.getDictionaryAttr(configItems));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registerBuiltinDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
  }

  // These are hooks into the HAL managed pipelines.
  void buildConfigurationPassPipeline(ExecutableTargetAttr targetAttr,
                                      OpPassManager &passManager) override {
    (void)targetAttr;
    cellar::target::hexagon::buildHexagonConfigurationPassPipeline(passManager);
  }

  void buildTranslationPassPipeline(ExecutableTargetAttr targetAttr,
                                    OpPassManager &passManager) override {
    (void)targetAttr;
    cellar::target::hexagon::buildHexagonTranslationPassPipeline(passManager);
  }

  void buildLinkingPassPipeline(OpPassManager &passManager) override {
    // Passing the backend name ensures the link pass actually gathers
    // Hexagon executables instead of skipping with an empty target filter.
    cellar::target::hexagon::buildHexagonLinkingPassPipeline(
        passManager, std::make_optional<std::string>("hexagon"));
  }

  // Build the IREE HAL executable library metadata. The runtime uses this
  // to find the entry point functions and their information. More details:
  //
  // The linking pipeline assigns ordinals to the functions and associates
  // exported symbols to them, by creating hal.executable.export attributes
  // (also does the same for imports btw). The hal instructions use these
  // ordinals to call on the functions. Therefore, we must expose a "query"
  // function that should allow to retrieve the addresses of the functions from
  // their assigned ordinals and add it to the executable so that the hal can
  // call on it.
  //
  // Also note that this is not the solution implemented for other targets. The
  // GPUs embed this information into the flatbuffer instead. This must be
  // synchronized with the hal. We are copying the pattern from the CPU for the
  // time being though.
  //
  // Another pattern that we are not reusing from LLVMCPUTarget is that all
  // functions except the query function have their visibility changed and are
  // hidden. We are not currently doing the same for Hexagon and all functions
  // are exposed.
  void buildExecutableMetadata(const LLVMTarget &target,
                               llvm::Module &llvmModule,
                               ExecutableVariantOp &variantOp) {
    LibraryBuilder::Mode libraryBuilderMode =
        target.debugSymbols ? LibraryBuilder::Mode::INCLUDE_REFLECTION_ATTRS
                            : LibraryBuilder::Mode::NONE;
    LibraryBuilder libraryBuilder(&llvmModule, libraryBuilderMode,
                                  LibraryBuilder::Version::LATEST);

    // The LLVMCPUTarget has support for multiple sanitizer kinds, defined
    // in target.sanitizerKind. For simplicity, let's not add any for now.
    libraryBuilder.setSanitizerKind(LibraryBuilder::SanitizerKind::NONE);

    // Declare dynamically imported functions (currently unused by Hexagon, so
    // this has not been checked).
    auto importsAttrName =
        StringAttr::get(variantOp.getContext(), "hal.executable.imports");
    if (auto importsAttr =
            variantOp->getAttrOfType<ArrayAttr>(importsAttrName)) {
      for (auto importAttr : importsAttr.getAsValueRange<ArrayAttr>()) {
        auto nameAttr = llvm::cast<StringAttr>(importAttr[0]);
        auto weakAttr = llvm::cast<BoolAttr>(importAttr[1]);
        libraryBuilder.addImport(nameAttr.getValue(), weakAttr.getValue());
      }
      variantOp->removeAttr(importsAttrName);
    }

    // Declare exported entry points.
    auto align16 = llvm::Attribute::getWithAlignment(llvmModule.getContext(),
                                                     llvm::Align(16));
    for (auto exportOp : variantOp.getBlock().getOps<ExecutableExportOp>()) {
      // Find the matching function in the LLVM module.
      auto *llvmFunc = llvmModule.getFunction(exportOp.getName());
      if (!llvmFunc)
        continue;

      // We do not want to hide the other functions like LLVMCPUTarget does,
      // easier debugging for now
      // llvmFunc->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
      // llvmFunc->setDSOLocal(true);

      // Tag the function parameters in case they got removed during conversion.
      // (%arg0: environment, %arg1: dispatch_state, %arg2: workgroup_state)
      for (unsigned i = 0; i <= 2; ++i) {
        llvmFunc->addParamAttr(i, llvm::Attribute::NonNull);
        llvmFunc->addParamAttr(i, llvm::Attribute::NoAlias);
        llvmFunc->addParamAttr(i, align16);
      }

      LibraryBuilder::DispatchAttrs dispatchAttrs = {};

      // Entry points may optionally specify that they require workgroup local
      // memory. We fetch that value here and plumb it through so the runtime
      // knows how much memory to reserve and pass in.
      dispatchAttrs.localMemorySize = exportOp.getWorkgroupLocalMemory()
                                          .value_or(APInt(64, 0))
                                          .getSExtValue();

      // Specify the constant and binding information used to validate
      // dispatches.
      if (auto layoutAttr = exportOp.getLayout()) {
        dispatchAttrs.constantCount = layoutAttr.getConstants();
        dispatchAttrs.bindingCount = layoutAttr.getBindings().size();
      }

      LibraryBuilder::SourceLocation sourceLocation;
      SmallVector<LibraryBuilder::SourceLocation> stageLocations;
      libraryBuilder.addExport(exportOp.getName(), std::move(sourceLocation),
                               std::move(stageLocations), /*tag=*/"",
                               dispatchAttrs, llvmFunc);
    }

    // TODO: LLVMCPUTarget performs an additional step here to embed source
    // files. I do not know exactly if this is needed, nor do I understand
    // its purpose in the LLVMCPUTarget yet

    // TODO: This is where we are actually performing the work. Note that
    // this is reusing a lot of machinery from LLVMCPUTarget through a call to
    // the that plugin's functions.
    auto queryFunctionName = std::string(kQueryFunctionName);
    auto *queryLibraryFunc = libraryBuilder.build(queryFunctionName);

    // The query function must be exported for dynamic libraries.
    queryLibraryFunc->setDSOLocal(false);
    queryLibraryFunc->setVisibility(
        llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
    queryLibraryFunc->setLinkage(
        llvm::GlobalValue::LinkageTypes::ExternalLinkage);
  }

  // Run the target backend codegen pipeline to produce an ELF object
  SmallVector<Artifact> generateObjectFiles(llvm::Module &llvmModule,
                                            llvm::TargetMachine &targetMachine,
                                            ExecutableVariantOp &variantOp,
                                            const SerializationOptions &options,
                                            StringRef libraryName) {
    llvm::SmallVector<char, 0> objectDataStorage;
    llvm::raw_svector_ostream objectStream(objectDataStorage);
    llvm::legacy::PassManager passManager;
    if (targetMachine.addPassesToEmitFile(passManager, objectStream, nullptr,
                                          llvm::CodeGenFileType::ObjectFile)) {
      variantOp.emitOpError()
          << "Hexagon target machine cannot emit object files";
    }
    passManager.run(llvmModule);
    std::vector<int8_t> objectData(objectDataStorage.begin(),
                                   objectDataStorage.end());

    if (!options.dumpBinariesPath.empty()) {
      dumpDataToPath<int8_t>(options.dumpBinariesPath, options.dumpBaseName,
                             variantOp.getName(), ".o", objectData);
    }

    // Persist the temporary object to disk so the linker can turn it into an
    // ET_DYN shared object
    SmallVector<Artifact> objectFiles;
    {
      Artifact objectFile = Artifact::createTemporary(libraryName, "o");
      auto &os = objectFile.outputFile->os();
      os.write(reinterpret_cast<const char *>(objectData.data()),
               objectData.size());
      os.flush();
      os.close();
      objectFiles.push_back(std::move(objectFile));
    }

    return objectFiles;
  }

  std::optional<Artifacts>
  linkArtifacts(const SmallVector<Artifact> &objectFiles,
                const LLVMTarget &llvmIreeTarget,
                const llvm::TargetMachine &targetMachine,
                ExecutableVariantOp &variantOp, const StringRef libraryName) {
    // FIXME: I can optionally pass more arguments here, but any other options
    // would not be useful given my current custom implementation of the linker.
    // This type is yet another example of reused code from LLVMCPUTarget that
    // is meant for more complex logic
    LLVMTargetOptions linkerOptions;
    linkerOptions.target.copy(llvmIreeTarget);
    linkerOptions.embeddedLinkerPath = this->hexagonOptions.linker;

    auto linkerTool =
        createHexagonLinkerTool(targetMachine.getTargetTriple(), linkerOptions);

    auto linkedArtifactsOption =
        linkerTool->linkDynamicLibrary(libraryName, objectFiles);
    if (!linkedArtifactsOption) {
      variantOp.emitOpError() << "failed to link Hexagon shared object "
                                 "(see linker output above)";
    }

    return linkedArtifactsOption;
  }

  // Here we are creating our output .vmfb that should contain:
  // .so, constants.bin and .fb
  // For more info, check here:
  // https://linear.app/roofline/document/luis-meeting-notes-fec2bd974e4a
  //
  // Takes charge of translating to LLVMIR, calling the LLVM hexagon
  // target, linking the files and calling the emitFile passes to finally
  // create the executable.
  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {

    if (!options.dumpIntermediatesPath.empty()) {
      dumpMLIRModuleToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                           variantOp.getName(), ".codegen",
                           variantOp.getInnerModule());
    }

    initializeHexagonTarget();

    // Conversions between IREE and LLVM types
    // Note that the LLVM Target type and its related functions are reusing part
    // of IREE's LLVMCPUTarget plugin
    // Retrieve IREE's LLVM target and create the LLVM's TargetMachine from it.
    auto targetAttr = variantOp.getTarget();
    DictionaryAttr configAttr = targetAttr.getConfiguration();
    if (!configAttr)
      variantOp->emitError("Failed to retrieve target attribute configuration");

    auto llvmTargetOption = LLVMTarget::loadFromConfigAttr(
        variantOp->getLoc(), configAttr,
        createLLVMTargetForHexagon(this->hexagonOptions));
    if (!llvmTargetOption)
      variantOp->emitError(
          "Failed to load LLVMTarget from configuration attributes");

    llvm::LLVMContext context;
    auto libraryName =
        variantOp->getParentOfType<ExecutableOp>().getName().str();

    // Convert the MLIR LLVM dialect module to an llvm::Module for codegen.
    auto llvmModule = translateModuleToLLVMIR(variantOp.getInnerModule(),
                                              context, libraryName);
    if (!llvmModule)
      return variantOp.emitOpError()
             << "failed to translate module to LLVM IR for Hexagon";

    const auto &llvmIreeTarget = llvmTargetOption.value();

    buildExecutableMetadata(llvmIreeTarget, *llvmModule, variantOp);

    auto targetMachine = createTargetMachine(llvmIreeTarget);
    if (!targetMachine) {
      return variantOp->emitError(
          "failed to create target machine for target triple '" +
          llvmIreeTarget.getTriple() + "'");
    }

    // This information is embedded into each one of the dispatches. When
    // linking all dispatches together through the linking pipeline, a new
    // module is created that does not copy this information, so let's add
    // it again in case it is needed at some other point. Might be a bug in
    // the linking pipeline? LLVMCPUTarget also needs this dirty fix
    llvmModule->setTargetTriple(targetMachine->getTargetTriple());
    llvmModule->setDataLayout(targetMachine->createDataLayout());

    // Note that we are dumping the ll after the fixes above, but LLVMCPUTarget
    // outputs multiple ll's for each stage though. Be careful if you are
    // comparing them to each other.
    if (!options.dumpIntermediatesPath.empty()) {
      dumpLLVMModuleToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                           variantOp.getName(), *llvmModule);
    }

    // Dump assembly
    if (!options.dumpBinariesPath.empty()) {
      dumpAssemblyFromLLVMModule(variantOp, *llvmModule, *targetMachine,
                                 options.dumpBinariesPath,
                                 options.dumpBaseName);
    }

    SmallVector<Artifact> objectFiles = generateObjectFiles(
        *llvmModule, *targetMachine, variantOp, options, libraryName);

    auto linkedArtifacts = linkArtifacts(
        objectFiles, llvmIreeTarget, *targetMachine, variantOp, libraryName);

    auto libraryFileOption = linkedArtifacts->libraryFile.read();
    if (!libraryFileOption) {
      return variantOp.emitOpError()
             << "failed to read back linked Hexagon library from "
             << linkedArtifacts->libraryFile.path;
    }
    if (!options.dumpBinariesPath.empty()) {
      dumpDataToPath<int8_t>(options.dumpBinariesPath, options.dumpBaseName,
                             variantOp.getName(), ".so",
                             libraryFileOption.value());
    }

    // Embed the resulting executable binary into the IR
    auto bufferAttr = DenseIntElementsAttr::get(
        VectorType::get({static_cast<int64_t>(libraryFileOption->size())},
                        IntegerType::get(executableBuilder.getContext(), 8)),
        std::move(libraryFileOption.value()));
    auto binaryOp = ExecutableBinaryOp::create(
        executableBuilder, variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(), bufferAttr);
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-elf"));

    return success();
  }

private:
  const HexagonOptions &hexagonOptions;
};

// The plugin session simply takes care of registering all the extensions from
// the plugin
struct HexagonSession
    : public PluginSession<HexagonSession, HexagonOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    cellar::target::hexagon::registerHexagonCodeGenPasses();
  };

  // For some reason, trying to override registerGlobalDialects through the CRTP
  // does not work, so this is an alternative. Not sure about the difference.
  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<IREE::Hexagon::IREEHexagonEncodingDialect>();
    iree_compiler::hexagon::registerHexagonEncodingExternalModels(registry);
  }

  void populateHALTargetDevices(TargetDeviceList &targets) override {
    targets.add("hexagon", [=, this]() {
      return std::make_shared<HexagonTargetDevice>(options);
    });
  }

  void populateHALTargetBackends(TargetBackendList &targets) override {
    targets.add("hexagon", [=, this]() {
      return std::make_shared<HexagonTargetBackend>(options);
    });
  }
};

} // namespace hexagon::HAL

IREE_DEFINE_COMPILER_OPTION_FLAGS(::hexagon::HAL::HexagonOptions);

extern "C" bool iree_register_compiler_plugin_hal_target_hexagon(
    iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::hexagon::HAL::HexagonSession>(
      "hal_target_hexagon");
  return true;
}
