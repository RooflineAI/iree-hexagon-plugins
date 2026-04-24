// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cellar-hexagon/Target/HexagonExecutableSerialization.h"

#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "cellar-hexagon/Target/HexagonLLVMTarget.h"
#include "cellar-hexagon/Target/Linking/HexagonLinkerTool.h"
#include "compiler/plugins/target/LLVMCPU/LLVMTargetOptions.h"
#include "compiler/plugins/target/LLVMCPU/LibraryBuilder.h"
#include "compiler/plugins/target/LLVMCPU/LinkerTool.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"

// TODO: There is a lot of code that is calling on functions from the
// LLVMCPUTarget plugin, especially during linking. This also includes other
// files inside the hexagon plugin. This will have to be revisited in the future
// since it is not particularly clean...

namespace mlir::iree_compiler::cellar_hexagon::target {
namespace HAL = mlir::iree_compiler::IREE::HAL;
using HAL::Artifact;
using HAL::Artifacts;
using HAL::dumpDataToPath;
using HAL::LibraryBuilder;
using HAL::LLVMTarget;
using HAL::LLVMTargetOptions;
using HAL::TargetBackend;
namespace {

static constexpr char kQueryFunctionName[] =
    "iree_hal_executable_library_query";

static void dumpMLIRModuleToPath(llvm::StringRef path, llvm::StringRef baseName,
                                 llvm::StringRef suffix,
                                 llvm::StringRef extPrefix,
                                 mlir::ModuleOp module) {
  std::string extension = (extPrefix + ".mlir").str();
  std::string textData;
  llvm::raw_string_ostream os(textData);
  mlir::OpPrintingFlags printingFlags;
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

static void dumpLLVMModuleToPath(llvm::StringRef path, llvm::StringRef baseName,
                                 llvm::StringRef suffix, llvm::Module &module) {
  llvm::SmallVector<char, 0> textData;
  llvm::raw_svector_ostream ostream(textData);
  module.print(ostream, nullptr);
  dumpDataToPath(path, baseName, suffix, ".ll",
                 llvm::StringRef(textData.data(), textData.size()));

  // Dump bitcode to path.
  llvm::SmallVector<char> binaryData;
  llvm::raw_svector_ostream binaryOstream(binaryData);
  // Write the specified module to the specified output stream.
  llvm::WriteBitcodeToFile(module, binaryOstream);
  dumpDataToPath(path, baseName, suffix, ".bc",
                 llvm::StringRef(binaryData.data(), binaryData.size()));
}

static void dumpAssemblyFromLLVMModule(HAL::ExecutableVariantOp variantOp,
                                       llvm::Module &llvmModule,
                                       llvm::TargetMachine &targetMachine,
                                       llvm::StringRef path,
                                       llvm::StringRef baseName) {
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
static void buildExecutableMetadata(const LLVMTarget &target,
                                    llvm::Module &llvmModule,
                                    HAL::ExecutableVariantOp &variantOp) {
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
      mlir::StringAttr::get(variantOp.getContext(), "hal.executable.imports");
  if (auto importsAttr =
          variantOp->getAttrOfType<mlir::ArrayAttr>(importsAttrName)) {
    for (auto importAttr : importsAttr.getAsValueRange<mlir::ArrayAttr>()) {
      auto nameAttr = mlir::cast<mlir::StringAttr>(importAttr[0]);
      auto weakAttr = mlir::cast<mlir::BoolAttr>(importAttr[1]);
      libraryBuilder.addImport(nameAttr.getValue(), weakAttr.getValue());
    }
    variantOp->removeAttr(importsAttrName);
  }

  // Declare exported entry points.
  auto align16 = llvm::Attribute::getWithAlignment(llvmModule.getContext(),
                                                   llvm::Align(16));
  for (auto exportOp : variantOp.getBlock().getOps<HAL::ExecutableExportOp>()) {
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
                                        .value_or(llvm::APInt(64, 0))
                                        .getSExtValue();

    // Specify the constant and binding information used to validate
    // dispatches.
    if (auto layoutAttr = exportOp.getLayout()) {
      dispatchAttrs.constantCount = layoutAttr.getConstants();
      dispatchAttrs.bindingCount = layoutAttr.getBindings().size();
    }

    // Extract workgroup size if specified at compile time.
    if (auto workgroupSizeAttr = exportOp.getWorkgroupSize()) {
      auto workgroupSizeValues = workgroupSizeAttr->getValue();
      dispatchAttrs.workgroupSize[0] = static_cast<uint32_t>(
          mlir::cast<mlir::IntegerAttr>(workgroupSizeValues[0]).getInt());
      dispatchAttrs.workgroupSize[1] = static_cast<uint32_t>(
          mlir::cast<mlir::IntegerAttr>(workgroupSizeValues[1]).getInt());
      dispatchAttrs.workgroupSize[2] = static_cast<uint32_t>(
          mlir::cast<mlir::IntegerAttr>(workgroupSizeValues[2]).getInt());
    }

    LibraryBuilder::SourceLocation sourceLocation;
    SmallVector<LibraryBuilder::SourceLocation> stageLocations;
    libraryBuilder.addExport(exportOp.getName(), std::move(sourceLocation),
                             std::move(stageLocations), /*tag=*/"",
                             dispatchAttrs, llvmFunc);
  }

  // TODO: This is where we are actually performing the work. Note that
  // this is reusing a lot of machinery from LLVMCPUTarget through a call to
  // the that plugin's functions and this should be fixed.
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
static llvm::SmallVector<Artifact>
generateObjectFiles(llvm::Module &llvmModule,
                    llvm::TargetMachine &targetMachine,
                    HAL::ExecutableVariantOp &variantOp,
                    const TargetBackend::SerializationOptions &options,
                    llvm::StringRef libraryName) {
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
  llvm::SmallVector<Artifact> objectFiles;
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

static mlir::LogicalResult
appendLinkerObjects(HAL::ExecutableVariantOp &variantOp,
                    llvm::StringRef libraryName,
                    llvm::SmallVector<Artifact> &objectFiles) {
  llvm::SmallVector<HAL::ExecutableObjectAttr> linkerObjectAttrs;
  HAL::ExecutableObjectAttr::filterObjects(variantOp.getObjectsAttr(),
                                           {".o", ".obj", ".a", ".lib"},
                                           linkerObjectAttrs);

  for (auto [index, objectAttr] : llvm::enumerate(linkerObjectAttrs)) {
    if (objectAttr.getData()) {
      auto objectData = objectAttr.loadData();
      if (!objectData) {
        return variantOp.emitOpError()
               << "failed to load inline executable object data for "
               << objectAttr;
      }

      auto pathAttr = objectAttr.getPath();
      auto extension = pathAttr
                           ? llvm::sys::path::extension(pathAttr.getValue())
                           : llvm::StringRef("o");
      if (extension.empty())
        extension = "o";

      Artifact objectFile = Artifact::createTemporary(
          libraryName.str() + "_object_" + std::to_string(index), extension);
      auto &os = objectFile.outputFile->os();
      os.write(objectData->data(), objectData->size());
      os.close();
      objectFiles.push_back(std::move(objectFile));
      continue;
    }

    auto absolutePath = objectAttr.getAbsolutePath();
    if (failed(absolutePath)) {
      return variantOp.emitOpError()
             << "referenced executable object file not found; use "
                "--iree-hal-executable-object-search-path= to add search "
                "paths: "
             << objectAttr;
    }
    objectFiles.push_back(Artifact::fromFile(*absolutePath));
  }

  return success();
}

static std::optional<Artifacts> linkArtifacts(
    const HexagonOptions &options,
    const llvm::SmallVector<Artifact> &objectFiles,
    const LLVMTarget &llvmIreeTarget, const llvm::TargetMachine &targetMachine,
    HAL::ExecutableVariantOp &variantOp, const llvm::StringRef libraryName) {
  // FIXME: I can optionally pass more arguments here, but any other options
  // would not be useful given my current custom implementation of the linker.
  // This type is yet another example of reused code from LLVMCPUTarget that
  // is meant for more complex logic
  LLVMTargetOptions linkerOptions;
  linkerOptions.target.copy(llvmIreeTarget);
  linkerOptions.embeddedLinkerPath = options.linker;

  auto linkerTool = mlir::iree_compiler::cellar_hexagon::target::linking::
      createHexagonLinkerTool(targetMachine.getTargetTriple(), linkerOptions);

  auto linkedArtifactsOption =
      linkerTool->linkDynamicLibrary(libraryName, objectFiles);
  if (!linkedArtifactsOption) {
    variantOp.emitOpError()
        << "failed to link Hexagon shared object (see linker output above)";
  }

  return linkedArtifactsOption;
}

} // namespace

// Here we are creating our output .vmfb that should contain:
// .so, constants.bin and .fb
// For more info, check here:
// https://linear.app/roofline/document/luis-meeting-notes-fec2bd974e4a
//
// Takes charge of translating to LLVMIR, calling the LLVM hexagon
// target, linking the files and calling the emitFile passes to finally
// create the executable.
mlir::LogicalResult serializeHexagonExecutable(
    const HexagonOptions &options,
    const TargetBackend::SerializationOptions &serializationOptions,
    HAL::ExecutableVariantOp variantOp, mlir::OpBuilder &executableBuilder) {
  if (!serializationOptions.dumpIntermediatesPath.empty()) {
    dumpMLIRModuleToPath(serializationOptions.dumpIntermediatesPath,
                         serializationOptions.dumpBaseName, variantOp.getName(),
                         ".codegen", variantOp.getInnerModule());
  }

  initializeHexagonTarget();

  // Conversions between IREE and LLVM types
  // Note that the LLVM Target type and its related functions are reusing part
  // of IREE's LLVMCPUTarget plugin
  // Retrieve IREE's LLVM target and create the LLVM's TargetMachine from it.
  auto targetAttr = variantOp.getTarget();
  mlir::DictionaryAttr configAttr = targetAttr.getConfiguration();
  if (!configAttr)
    variantOp->emitError("Failed to retrieve target attribute configuration");

  auto llvmTargetOption = LLVMTarget::loadFromConfigAttr(
      variantOp->getLoc(), configAttr, createLLVMTargetForHexagon(options));
  if (!llvmTargetOption)
    variantOp->emitError(
        "Failed to load LLVMTarget from configuration attributes");

  llvm::LLVMContext context;
  auto libraryName =
      variantOp->getParentOfType<HAL::ExecutableOp>().getName().str();

  // Convert the MLIR LLVM dialect module to an llvm::Module for codegen.
  auto llvmModule =
      translateModuleToLLVMIR(variantOp.getInnerModule(), context, libraryName);
  if (!llvmModule)
    return variantOp.emitOpError()
           << "failed to translate module to LLVM IR for Hexagon";

  const auto &llvmIreeTarget = llvmTargetOption.value();

  buildExecutableMetadata(llvmIreeTarget, *llvmModule, variantOp);

  auto targetMachine = createTargetMachine(llvmIreeTarget);
  if (!targetMachine) {
    return variantOp->emitError("failed to create target machine for target "
                                "triple '" +
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
  if (!serializationOptions.dumpIntermediatesPath.empty()) {
    dumpLLVMModuleToPath(serializationOptions.dumpIntermediatesPath,
                         serializationOptions.dumpBaseName, variantOp.getName(),
                         *llvmModule);
  }

  // Dump assembly
  if (!serializationOptions.dumpBinariesPath.empty()) {
    dumpAssemblyFromLLVMModule(variantOp, *llvmModule, *targetMachine,
                               serializationOptions.dumpBinariesPath,
                               serializationOptions.dumpBaseName);
  }

  llvm::SmallVector<Artifact> objectFiles =
      generateObjectFiles(*llvmModule, *targetMachine, variantOp,
                          serializationOptions, libraryName);

  // Here we are linking any objects defined as a hal.executable.objects in
  // the IR
  // These are controlled through the --iree-hal-executable-object-search-path
  if (failed(appendLinkerObjects(variantOp, libraryName, objectFiles))) {
    return failure();
  }

  auto linkedArtifacts = linkArtifacts(options, objectFiles, llvmIreeTarget,
                                       *targetMachine, variantOp, libraryName);
  if (!linkedArtifacts)
    return failure();

  auto libraryFileOption = linkedArtifacts->libraryFile.read();
  if (!libraryFileOption) {
    return variantOp.emitOpError()
           << "failed to read back linked Hexagon library from "
           << linkedArtifacts->libraryFile.path;
  }
  if (!serializationOptions.dumpBinariesPath.empty()) {
    dumpDataToPath<int8_t>(serializationOptions.dumpBinariesPath,
                           serializationOptions.dumpBaseName,
                           variantOp.getName(), ".so",
                           libraryFileOption.value());
  }

  // Embed the resulting executable binary into the IR
  auto bufferAttr = mlir::DenseIntElementsAttr::get(
      mlir::VectorType::get(
          {static_cast<int64_t>(libraryFileOption->size())},
          mlir::IntegerType::get(executableBuilder.getContext(), 8)),
      std::move(libraryFileOption.value()));
  auto binaryOp = HAL::ExecutableBinaryOp::create(
      executableBuilder, variantOp.getLoc(), variantOp.getSymName(),
      variantOp.getTarget().getFormat(), bufferAttr);
  binaryOp.setMimeTypeAttr(
      executableBuilder.getStringAttr("application/x-elf"));

  return mlir::success();
}

} // namespace mlir::iree_compiler::cellar_hexagon::target
