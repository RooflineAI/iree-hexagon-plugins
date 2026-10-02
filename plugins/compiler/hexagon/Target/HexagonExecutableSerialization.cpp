// Copyright 2025 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hexagon/Target/HexagonExecutableSerialization.h"
#include "hexagon/CodeGen/Conversion/HexagonRuntimeLinking.h"

#include "hexagon/Target/HexagonLLVMTarget.h"
#include "hexagon/Target/LibraryBuilder.h"
#include "hexagon/Target/Linking/HexagonLinker.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "iree/hal/local/executable_library.h"
// Generated flatcc builder for the Hexagon executable-def flatbuffer (host-
// readable export names + ELF). See serialize/hexagon_executable_def.fbs.
#include "hexagon/schemas/hexagon_executable_def_builder.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <memory>
#include <string>
#include <vector>

namespace mlir::iree_compiler::hexagon::target {
namespace HAL = mlir::iree_compiler::IREE::HAL;
using HAL::dumpDataToPath;
using HAL::TargetBackend;
namespace {

// The vendored LibraryBuilder emits the library tables the DSP runtime loads.
// The runtime checks the version at load time (see dsp/executable.c); this
// catches an IREE bump that changes the ABI at build time instead.
static_assert(static_cast<uint32_t>(LibraryBuilder::Version::LATEST) ==
                  IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
              "vendored Target/LibraryBuilder is out of sync with "
              "iree/hal/local/executable_library.h; re-sync it from upstream");

// Temporary files handed to the linker. Each one is deleted when its remover is
// destroyed at the end of serialization.
using TemporaryFiles = llvm::SmallVector<std::unique_ptr<llvm::FileRemover>>;

static constexpr char kQueryFunctionName[] =
    "iree_hal_executable_library_query";

// LLVM only knows a function's final frame size, including register spills,
// after register allocation. Use its `warn-stack-size` diagnostic to enforce a
// per-function frame limit during object generation.
//
// This is not a whole-call-chain stack analysis: it does not add the frames of
// callers and callees that can be live at the same time. Module-local call
// chains can still exceed the DSP stack while every individual frame is below
// the limit.
static llvm::cl::opt<unsigned> clHexagonFailOnStackFramesLargerThan(
    "iree-hexagon-fail-on-stack-frames-larger-than",
    llvm::cl::desc(
        "Fail compilation if any individual Hexagon function's final stack "
        "frame (locals and register spills measured after register allocation) "
        "exceeds the specified byte count. This is a per-frame check; it does "
        "not sum frames across a call chain. Set to 0 to disable."),
    llvm::cl::init(12 * 1024));

// Records the per-function stack-size diagnostics LLVM emits during codegen.
// Non stack diagnostics fall through (return false) to LLVM's default printing.
class StackSizeDiagnosticHandler : public llvm::DiagnosticHandler {
public:
  struct Violation {
    std::string function;
    uint64_t stackSize;
  };

  explicit StackSizeDiagnosticHandler(std::vector<Violation> &violations)
      : violations(violations) {}

  bool handleDiagnostics(const llvm::DiagnosticInfo &info) override {
    if (info.getKind() != llvm::DK_StackSize) {
      return false;
    }
    const auto &stackDiag = llvm::cast<llvm::DiagnosticInfoStackSize>(info);
    violations.push_back(
        {stackDiag.getFunction().getName().str(), stackDiag.getStackSize()});
    return true;
  }

private:
  std::vector<Violation> &violations;
};

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
  // bug disappear.
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
  targetMachine.Options.MCOptions.AsmVerbose = true;
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
static void
buildExecutableMetadata(llvm::Module &llvmModule,
                        HAL::ExecutableVariantOp &variantOp,
                        llvm::SmallVectorImpl<std::string> &entryPointNames) {
  // LLVMCPU also supports sanitizers and dropping the reflection attributes.
  // Hexagon supports neither, so the defaults (no sanitizer) are kept.
  LibraryBuilder libraryBuilder(&llvmModule,
                                LibraryBuilder::Mode::INCLUDE_REFLECTION_ATTRS,
                                LibraryBuilder::Version::LATEST);

  // Declare dynamically imported functions if present. Hexagon currently
  // expects runtime/helper symbols to be resolved via native DSP linking, so
  // imports here are unexpected.
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

    // Value-initialized: `line` is otherwise left uninitialized.
    LibraryBuilder::SourceLocation sourceLocation{};
    SmallVector<LibraryBuilder::SourceLocation> stageLocations;
    libraryBuilder.addExport(exportOp.getName(), std::move(sourceLocation),
                             std::move(stageLocations), /*tag=*/"",
                             dispatchAttrs, llvmFunc);

    // Collect the export name in the SAME order it is added to the library so
    // the host-readable flatbuffer's entry_points[i] lines up with the DSP's
    // exports->ptrs[i] (i.e. index == dispatch ordinal).
    entryPointNames.push_back(exportOp.getName().str());
  }

  auto queryFunctionName = std::string(kQueryFunctionName);
  auto *queryLibraryFunc = libraryBuilder.build(queryFunctionName);

  // The query function must be exported for dynamic libraries.
  queryLibraryFunc->setDSOLocal(false);
  queryLibraryFunc->setVisibility(
      llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
  queryLibraryFunc->setLinkage(
      llvm::GlobalValue::LinkageTypes::ExternalLinkage);
}

// Runs the LLVM middle-end optimization pipeline at O2. Loop interleaving,
// loop vectorization, loop unrolling and SLP vectorization stay disabled,
// matching the defaults of the LLVMCPU backend this was derived from.
static LogicalResult
runLLVMOptimizationPasses(llvm::TargetMachine &targetMachine,
                          llvm::Module &llvmModule) {
  llvm::LoopAnalysisManager loopAnalysisManager;
  llvm::FunctionAnalysisManager functionAnalysisManager;
  llvm::CGSCCAnalysisManager cgsccAnalysisManager;
  llvm::ModuleAnalysisManager moduleAnalysisManager;

  // Honors LLVM's debugging flags, such as -print-after-all.
  llvm::PassInstrumentationCallbacks passInstrumentationCallbacks;
  llvm::StandardInstrumentations standardInstrumentations(
      llvmModule.getContext(), /*DebugLogging=*/false);
  standardInstrumentations.registerCallbacks(passInstrumentationCallbacks);

  llvm::PipelineTuningOptions tuningOptions;
  tuningOptions.LoopInterleaving = false;
  tuningOptions.LoopVectorization = false;
  tuningOptions.LoopUnrolling = false;
  tuningOptions.SLPVectorization = false;

  llvm::PassBuilder passBuilder(&targetMachine, tuningOptions,
                                /*PGOOpt=*/std::nullopt,
                                &passInstrumentationCallbacks);
  passBuilder.registerModuleAnalyses(moduleAnalysisManager);
  passBuilder.registerCGSCCAnalyses(cgsccAnalysisManager);
  passBuilder.registerFunctionAnalyses(functionAnalysisManager);
  passBuilder.registerLoopAnalyses(loopAnalysisManager);
  passBuilder.crossRegisterProxies(loopAnalysisManager, functionAnalysisManager,
                                   cgsccAnalysisManager, moduleAnalysisManager);

  llvm::ModulePassManager modulePassManager =
      passBuilder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
  modulePassManager.run(llvmModule, moduleAnalysisManager);

  return failure(llvm::verifyModule(llvmModule, &llvm::errs()));
}

// Writes `data` to a new temporary file and returns its path.
static FailureOr<std::string> writeTemporaryFile(llvm::StringRef prefix,
                                                 llvm::StringRef extension,
                                                 llvm::StringRef data,
                                                 TemporaryFiles &tempFiles) {
  llvm::SmallString<128> path;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          sanitizeFileName(prefix), sanitizeFileName(extension), path)) {
    llvm::errs() << "failed to create temporary file: " << error.message()
                 << "\n";
    return failure();
  }
  tempFiles.push_back(std::make_unique<llvm::FileRemover>(path));

  std::error_code error;
  llvm::raw_fd_ostream os(path, error);
  if (error) {
    llvm::errs() << "failed to open temporary file '" << path
                 << "': " << error.message() << "\n";
    return failure();
  }
  os << data;
  os.close();
  if (os.has_error()) {
    llvm::errs() << "failed to write temporary file '" << path
                 << "': " << os.error().message() << "\n";
    os.clear_error();
    return failure();
  }
  return path.str().str();
}

// Run the target backend codegen pipeline to produce an ELF object and write
// it to a temporary file for the linker.
static LogicalResult
generateObjectFile(llvm::Module &llvmModule, llvm::TargetMachine &targetMachine,
                   HAL::ExecutableVariantOp &variantOp,
                   const TargetBackend::SerializationOptions &options,
                   llvm::StringRef libraryName, TemporaryFiles &tempFiles,
                   llvm::SmallVectorImpl<std::string> &objectPaths) {
  llvm::SmallVector<char, 0> objectData;
  llvm::raw_svector_ostream objectStream(objectData);
  llvm::legacy::PassManager passManager;
  if (targetMachine.addPassesToEmitFile(passManager, objectStream, nullptr,
                                        llvm::CodeGenFileType::ObjectFile)) {
    return variantOp.emitOpError()
           << "Hexagon target machine cannot emit object files";
  }
  passManager.run(llvmModule);
  llvm::StringRef objectRef(objectData.data(), objectData.size());

  if (!options.dumpBinariesPath.empty()) {
    dumpDataToPath(options.dumpBinariesPath, options.dumpBaseName,
                   variantOp.getName(), ".o", objectRef);
  }

  // Persist the temporary object to disk so the linker can turn it into an
  // ET_DYN shared object
  FailureOr<std::string> objectPath =
      writeTemporaryFile(libraryName, "o", objectRef, tempFiles);
  if (failed(objectPath))
    return failure();
  objectPaths.push_back(*objectPath);
  return success();
}

static mlir::LogicalResult
appendLinkerObjects(HAL::ExecutableVariantOp &variantOp,
                    llvm::StringRef libraryName, TemporaryFiles &tempFiles,
                    llvm::SmallVectorImpl<std::string> &objectPaths) {
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

      FailureOr<std::string> objectPath = writeTemporaryFile(
          libraryName.str() + "_object_" + std::to_string(index), extension,
          *objectData, tempFiles);
      if (failed(objectPath))
        return failure();
      objectPaths.push_back(*objectPath);
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
    objectPaths.push_back(*absolutePath);
  }

  return success();
}

// Links the objects into a shared object and returns its contents.
static FailureOr<std::unique_ptr<llvm::MemoryBuffer>>
linkSharedObject(const HexagonOptions &options,
                 llvm::ArrayRef<std::string> objectPaths,
                 HAL::ExecutableVariantOp &variantOp,
                 llvm::StringRef libraryName, TemporaryFiles &tempFiles) {
  FailureOr<std::string> libraryPath =
      writeTemporaryFile(libraryName, "so", "", tempFiles);
  if (failed(libraryPath))
    return failure();

  // Allow undefined symbols that will be resolved from the runtime when the
  // variant is tagged.
  const bool allowNativeUndefinedSymbols =
      variantOp->hasAttr(codegen::kNativeRuntimeLinkVariantAttrName);
  if (failed(linking::linkHexagonSharedObject(options.linker, objectPaths,
                                              *libraryPath,
                                              allowNativeUndefinedSymbols))) {
    return variantOp.emitOpError()
           << "failed to link Hexagon shared object (see linker output above)";
  }

  auto library = llvm::MemoryBuffer::getFile(*libraryPath);
  if (!library) {
    return variantOp.emitOpError()
           << "failed to read back linked Hexagon library from "
           << *libraryPath;
  }
  return std::move(*library);
}

} // namespace

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

  // The executable target configuration written by HexagonTargetBackend
  // describes the LLVM target machine to generate code for.
  mlir::DictionaryAttr configAttr = variantOp.getTarget().getConfiguration();
  if (!configAttr)
    return variantOp.emitOpError()
           << "missing executable target attribute configuration";

  llvm::LLVMContext context;
  auto libraryName =
      variantOp->getParentOfType<HAL::ExecutableOp>().getName().str();

  // Convert the MLIR LLVM dialect module to an llvm::Module for codegen.
  auto llvmModule =
      translateModuleToLLVMIR(variantOp.getInnerModule(), context, libraryName);
  if (!llvmModule)
    return variantOp.emitOpError()
           << "failed to translate module to LLVM IR for Hexagon";

  llvm::SmallVector<std::string> entryPointNames;
  buildExecutableMetadata(*llvmModule, variantOp, entryPointNames);

  auto targetMachine = createHexagonTargetMachine(configAttr);
  if (!targetMachine) {
    return variantOp.emitOpError()
           << "failed to create Hexagon target machine from " << configAttr;
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

  // Run the LLVM IR middle-end optimization pipeline before instruction
  // selection.
  if (failed(runLLVMOptimizationPasses(*targetMachine, *llvmModule))) {
    return variantOp.emitOpError()
           << "failed to run LLVM IR optimization passes for Hexagon";
  }

  if (!serializationOptions.dumpIntermediatesPath.empty()) {
    dumpLLVMModuleToPath(serializationOptions.dumpIntermediatesPath,
                         serializationOptions.dumpBaseName,
                         llvm::StringRef(variantOp.getName().str() + ".opt"),
                         *llvmModule);
  }

  // Dump assembly
  if (!serializationOptions.dumpBinariesPath.empty()) {
    dumpAssemblyFromLLVMModule(variantOp, *llvmModule, *targetMachine,
                               serializationOptions.dumpBinariesPath,
                               serializationOptions.dumpBaseName);
  }

  // Ask LLVM to report defined functions whose final frame exceeds the
  // configured per-frame limit. PrologEpilogInserter evaluates this attribute
  // after register allocation, when both fixed stack objects and spills are
  // known. The diagnostic handler records those reports so serialization can
  // return an MLIR error after object generation.
  std::vector<StackSizeDiagnosticHandler::Violation> stackViolations;
  const unsigned stackFrameLimit = clHexagonFailOnStackFramesLargerThan;
  if (stackFrameLimit > 0) {
    std::string thresholdStr = std::to_string(stackFrameLimit);
    for (llvm::Function &func : llvmModule->functions()) {
      if (func.isDeclaration()) {
        continue;
      }
      func.addFnAttr("warn-stack-size", thresholdStr);
    }
    context.setDiagnosticHandler(
        std::make_unique<StackSizeDiagnosticHandler>(stackViolations),
        /*RespectFilters=*/false);
  }

  TemporaryFiles tempFiles;
  llvm::SmallVector<std::string> objectPaths;
  if (failed(generateObjectFile(*llvmModule, *targetMachine, variantOp,
                                serializationOptions, libraryName, tempFiles,
                                objectPaths))) {
    return failure();
  }

  if (!stackViolations.empty()) {
    InFlightDiagnostic diag =
        variantOp.emitOpError()
        << "Hexagon function stack frame exceeds the configured limit of "
        << stackFrameLimit << " B. Offending "
        << (stackViolations.size() == 1 ? "function:" : "functions:");
    for (const auto &violation : stackViolations) {
      diag.attachNote() << violation.function << ": " << violation.stackSize
                        << " B frame (includes register spills)";
    }
    return failure();
  }

  // Here we are linking any objects defined as a hal.executable.objects in
  // the IR
  // These are controlled through the --iree-hal-executable-object-search-path
  if (failed(appendLinkerObjects(variantOp, libraryName, tempFiles,
                                 objectPaths))) {
    return failure();
  }

  FailureOr<std::unique_ptr<llvm::MemoryBuffer>> library =
      linkSharedObject(options, objectPaths, variantOp, libraryName, tempFiles);
  if (failed(library))
    return failure();
  llvm::StringRef libraryData = (*library)->getBuffer();
  if (!serializationOptions.dumpBinariesPath.empty()) {
    dumpDataToPath(serializationOptions.dumpBinariesPath,
                   serializationOptions.dumpBaseName, variantOp.getName(),
                   ".so", libraryData);
  }

  // Wrap the export names (host-readable, in dense ordinal order) plus the
  // linked ELF into the executable-def flatbuffer, mirroring the GPU targets.
  // This lets the HAL resolve the by-name dispatch references introduced by
  // IREE #24036 to ordinals on the host; the DSP still loads the ELF and
  // dispatches by that same ordinal (exports->ptrs[ordinal]).
  FlatbufferBuilder builder;
  iree_hal_hexagon_ExecutableDef_start_as_root(builder);
  auto entryPointsRef = builder.createStringVec(entryPointNames);
  auto elfRef = flatbuffers_uint8_vec_create(
      builder, reinterpret_cast<const uint8_t *>(libraryData.data()),
      libraryData.size());
  iree_hal_hexagon_ExecutableDef_entry_points_add(builder, entryPointsRef);
  iree_hal_hexagon_ExecutableDef_elf_add(builder, elfRef);
  iree_hal_hexagon_ExecutableDef_end_as_root(builder);

  auto binaryOp = HAL::ExecutableBinaryOp::create(
      executableBuilder, variantOp.getLoc(), variantOp.getSymName(),
      variantOp.getTarget().getFormat(),
      builder.getHeaderPrefixedBufferAttr(
          executableBuilder.getContext(),
          iree_hal_hexagon_ExecutableDef_file_identifier, /*version=*/0));
  binaryOp.setMimeTypeAttr(
      executableBuilder.getStringAttr("application/x-flatbuffers"));

  return mlir::success();
}

} // namespace mlir::iree_compiler::hexagon::target
