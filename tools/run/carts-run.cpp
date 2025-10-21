///==========================================================================
/// File: carts-run.cpp
/// Main entry point for the CARTS runtime execution tool.
///==========================================================================

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "polygeist/Dialect.h"
#include "polygeist/Passes/Passes.h"

#include "arts/Analysis/ArtsAnalysisManager.h"
#include "arts/ArtsDialect.h"
#include "arts/Passes/ArtsPasses.h"
#include "arts/Utils/ArtsDebug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

ARTS_DEBUG_SETUP(carts_run)

//===----------------------------------------------------------------------===//
// Interface Attachments
//===----------------------------------------------------------------------===//
/// Use the original models for attaching type interfaces.
class MemRefInsider
    : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefInsider> {};

template <typename T>
struct PtrElementModel
    : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<
          PtrElementModel<T>, T> {};

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//
static cl::opt<std::string>
    InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool> Opt("O3", cl::desc("Apply Optimizations"),
                         cl::init(false));

static cl::opt<bool> EmitLLVM("emit-llvm", cl::desc("Emit LLVM IR output"),
                              cl::init(false));

static cl::opt<bool> Debug("g", cl::desc("Enable debug mode"), cl::init(false));

static cl::opt<bool> ExportJson(
    "export-json",
    cl::desc("Export DB analysis to JSON files in output/ directory"),
    cl::init(false));

static cl::opt<std::string> ArtsConfig("arts-config",
                                       cl::desc("ARTS configuration file path"),
                                       cl::value_desc("config_file"),
                                       cl::init(""));

//===----------------------------------------------------------------------===//
// Pipeline Stop Options
//===----------------------------------------------------------------------===//
enum class PipelineStage {
  Simplify,         // After initial cleanup and simplification
  ArtsInliner,      // After ARTS inliner
  OpenMPToArts,     // After OpenMPToArts
  EdtTransforms,    // After EDT transformations
  EdtOptimizations, // After EDT optimizations
  PreLowering,      // After pre-lowering transformations
  Db,               // After DB creation and optimization
  Epochs,           // After epoch creation
  ArtsToLLVM,       // After ConvertArtsToLLVM
  Complete          // Full pipeline (default)
};

static cl::opt<PipelineStage>
    StopAt(cl::desc("Stop pipeline at specified stage:"),
           cl::values(clEnumValN(PipelineStage::Simplify, "simplify",
                                 "Stop after simplification"),
                      clEnumValN(PipelineStage::ArtsInliner, "arts-inliner",
                                 "Stop after Arts inliner"),
                      clEnumValN(PipelineStage::OpenMPToArts, "openmp-to-arts",
                                 "Stop after OpenMP to Arts conversion"),
                      clEnumValN(PipelineStage::EdtTransforms, "edt-transforms",
                                 "Stop after EDT transformations"),
                      clEnumValN(PipelineStage::Db, "db",
                                 "Stop after db optimization"),
                      clEnumValN(PipelineStage::Epochs, "epochs",
                                 "Stop after epoch creation"),
                      clEnumValN(PipelineStage::EdtOptimizations, "edt",
                                 "Stop after EDT optimizations"),
                      clEnumValN(PipelineStage::PreLowering, "pre-lowering",
                                 "Stop after pre-lowering transformations"),
                      clEnumValN(PipelineStage::ArtsToLLVM, "arts-to-llvm",
                                 "Stop after Arts to LLVM conversion"),
                      clEnumValN(PipelineStage::Complete, "complete",
                                 "Run complete pipeline (default)")),
           cl::init(PipelineStage::Complete));

//===----------------------------------------------------------------------===//
// Helper Functions for Initialization and Pass Setup
//===----------------------------------------------------------------------===//
/// Register standard MLIR dialects, passes, and translations.
void registerDialects(DialectRegistry &registry) {
  registry.insert<mlir::polygeist::PolygeistDialect, mlir::arts::ArtsDialect>();
  mlir::registerAllPasses();
  mlir::registerAllTranslations();
  mlir::registerpolygeistPasses();
  mlir::func::registerInlinerExtension(registry);
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllFromLLVMIRTranslations(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
}

/// Initialize the MLIR context by loading necessary dialects and attaching
/// type interfaces.
void initializeContext(MLIRContext &context) {
  context.disableMultithreading(true);
  context.getOrLoadDialect<affine::AffineDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<DLTIDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::async::AsyncDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::NVVM::NVVMDialect>();
  context.getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  context.getOrLoadDialect<mlir::gpu::GPUDialect>();
  context.getOrLoadDialect<mlir::omp::OpenMPDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::polygeist::PolygeistDialect>();
  context.getOrLoadDialect<mlir::arts::ArtsDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

  /// Register all necessary interfaces for LLVM conversion
  LLVM::LLVMFunctionType::attachInterface<MemRefInsider>(context);
  LLVM::LLVMArrayType::attachInterface<MemRefInsider>(context);
  LLVM::LLVMPointerType::attachInterface<MemRefInsider>(context);
  LLVM::LLVMStructType::attachInterface<MemRefInsider>(context);
  MemRefType::attachInterface<PtrElementModel<MemRefType>>(context);
  LLVM::LLVMStructType::attachInterface<PtrElementModel<LLVM::LLVMStructType>>(
      context);
  LLVM::LLVMPointerType::attachInterface<
      PtrElementModel<LLVM::LLVMPointerType>>(context);
  LLVM::LLVMArrayType::attachInterface<PtrElementModel<LLVM::LLVMArrayType>>(
      context);
  LLVM::LLVMArrayType::attachInterface<MemRefInsider>(context);
  LLVM::LLVMStructType::attachInterface<MemRefInsider>(context);
  MemRefType::attachInterface<PtrElementModel<MemRefType>>(context);
  IndexType::attachInterface<PtrElementModel<IndexType>>(context);
  LLVM::LLVMStructType::attachInterface<PtrElementModel<LLVM::LLVMStructType>>(
      context);
  LLVM::LLVMPointerType::attachInterface<
      PtrElementModel<LLVM::LLVMPointerType>>(context);
  LLVM::LLVMArrayType::attachInterface<PtrElementModel<LLVM::LLVMArrayType>>(
      context);
}

/// Setup initial cleanup and simplification passes.
void setupInitialCleanup(mlir::OpPassManager &optPM, bool optEnabled) {
  optPM.addPass(createLowerAffinePass());
  optPM.addPass(polygeist::createCanonicalizeForPass());

  if (optEnabled) {
    optPM.addPass(createCSEPass());
    optPM.addPass(polygeist::createPolygeistCanonicalizePass());
    optPM.addPass(createMem2Reg());
    optPM.addPass(polygeist::createPolygeistCanonicalizePass());
    optPM.addPass(polygeist::replaceAffineCFGPass());
    optPM.addPass(affine::createAffineScalarReplacementPass());
    optPM.addPass(polygeist::replaceAffineCFGPass());
    optPM.addPass(createLoopInvariantCodeMotionPass());
    optPM.addPass(createCSEPass());
    optPM.addPass(polygeist::createPolygeistCanonicalizePass());
    optPM.addPass(createLowerAffinePass());
  }

  optPM.addPass(createCSEPass());
  optPM.addPass(polygeist::createPolygeistCanonicalizePass());
}

/// Setup ARTS inliner pass.
void setupArtsInliner(PassManager &pm) {
  pm.addPass(arts::createArtsInlinerPass());
}

/// Setup OpenMP to ARTS conversion passes.
void setupOpenMPToArts(PassManager &pm) {
  pm.addPass(arts::createConvertOpenMPtoArtsPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(createSymbolDCEPass());
}

/// Setup EDT transformation passes.
void setupEdtTransforms(PassManager &pm, arts::ArtsAnalysisManager *AM) {
  pm.addPass(arts::createEdtPass(AM, false));
  pm.addPass(arts::createEdtInvariantCodeMotionPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
}

/// Setup db creation and optimization passes.
void setupDb(PassManager &pm, arts::ArtsAnalysisManager *AM, bool identifyDbs,
             bool exportJson) {
  pm.addPass(arts::createCanonicalizeMemrefsPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(createCSEPass());
  pm.addPass(arts::createEdtPtrRematerializationPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(arts::createCreateDbsPass(identifyDbs));
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(createCSEPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createMem2Reg());
  pm.addPass(arts::createCanonicalizeDbsPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(arts::createDbPass(AM, exportJson));
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(createCSEPass());
  pm.addPass(createMem2Reg());
}

/// Setup EDT optimization passes.
void setupEdtOptimizations(PassManager &pm, arts::ArtsAnalysisManager *AM) {
  pm.addPass(arts::createEdtPass(AM, true));
  pm.addPass(arts::createConcurrencyPass(AM));
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
}

/// Setup epoch creation passes.
void setupEpochs(PassManager &pm, arts::ArtsAnalysisManager *AM) {
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(arts::createCreateEpochsPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
}

/// Setup pre-lowering passes.
void setupPreLowering(PassManager &pm, arts::ArtsAnalysisManager *AM) {
  pm.addPass(arts::createParallelEdtLoweringPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(arts::createDbLoweringPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(arts::createEdtLoweringPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(arts::createEpochLoweringPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(createCSEPass());
  pm.addPass(createMem2Reg());
}

/// Setup ARTS to LLVM conversion passes.
void setupArtsToLLVM(PassManager &pm, bool debug) {
  pm.addPass(arts::createConvertArtsToLLVMPass(debug));
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(createCSEPass());
  pm.addPass(createMem2Reg());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(createControlFlowSinkPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
}

/// Setup additional optimizations (post-ARTS pipeline).
void setupAdditionalOptimizations(mlir::OpPassManager &optPM) {
  optPM.addPass(polygeist::createPolygeistCanonicalizePass());
  optPM.addPass(createControlFlowSinkPass());
  optPM.addPass(polygeist::createPolygeistCanonicalizePass());
  optPM.addPass(createLoopInvariantCodeMotionPass());
  optPM.addPass(createCSEPass());
  optPM.addPass(polygeist::createPolygeistCanonicalizePass());
}

/// Setup LLVM IR emission passes.
void setupLLVMIREmission(PassManager &pm) {
  pm.addPass(createCSEPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(polygeist::createConvertPolygeistToLLVMPass());
  pm.addPass(polygeist::createPolygeistCanonicalizePass());
  pm.addPass(createCSEPass());
}

/// Configure the pass manager with the optimization passes.
void setupPassManager(mlir::ModuleOp module, MLIRContext &context,
                      PipelineStage stopAt = PipelineStage::Complete) {
  // Create module-level analysis manager for caching across functions
  std::unique_ptr<arts::ArtsAnalysisManager> AM =
      std::make_unique<arts::ArtsAnalysisManager>(module, ArtsConfig);

  /// Complete pipeline
  if (stopAt == PipelineStage::Complete) {
    PassManager pm(&context);

    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    setupInitialCleanup(optPM, Opt);

    /// Complete pipeline
    setupArtsInliner(pm);
    setupOpenMPToArts(pm);
    setupEdtTransforms(pm, AM.get());
    setupDb(pm, AM.get(), true, ExportJson);
    setupEdtOptimizations(pm, AM.get());
    setupEpochs(pm, AM.get());
    setupPreLowering(pm, AM.get());
    setupArtsToLLVM(pm, Debug);

    if (mlir::failed(pm.run(module))) {
      ARTS_ERROR("Error running CARTS pipeline");
      module->dump();
      return;
    }

    /// Optimizations (if enabled)
    if (Opt) {
      PassManager pm2(&context);
      mlir::OpPassManager &optPM = pm2.nest<mlir::func::FuncOp>();
      setupAdditionalOptimizations(optPM);

      if (mlir::failed(pm2.run(module))) {
        ARTS_ERROR("Error when running optimizations");
        module->dump();
        return;
      }
    }

    /// LLVM IR conversion (if enabled)
    if (EmitLLVM) {
      PassManager pm3(&context);
      setupLLVMIREmission(pm3);
      if (mlir::failed(pm3.run(module))) {
        ARTS_ERROR("Error when emitting LLVM IR");
        module->dump();
        return;
      }
    }
    return;
  }

  /// Initial cleanup and simplification
  {
    PassManager pm(&context);
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    setupInitialCleanup(optPM, Opt);

    if (mlir::failed(pm.run(module))) {
      ARTS_ERROR("Error simplyfing the IR");
      module->dump();
      return;
    }
  }

  if (stopAt == PipelineStage::Simplify)
    return;

  /// ARTS dialect conversion and optimization
  {
    PassManager pm(&context);
    setupArtsInliner(pm);
    if (mlir::failed(pm.run(module))) {
      ARTS_ERROR("Error when running ARTS inliner");
      module->dump();
      return;
    }
  }

  if (stopAt == PipelineStage::ArtsInliner)
    return;

  /// OpenMP to ARTS conversion
  {
    PassManager pm(&context);
    setupOpenMPToArts(pm);
    if (mlir::failed(pm.run(module))) {
      ARTS_ERROR("Error when converting OpenMP to ARTS");
      module->dump();
      return;
    }
  }

  if (stopAt == PipelineStage::OpenMPToArts)
    return;

  /// EDT transformations
  {
    PassManager pm(&context);
    setupEdtTransforms(pm, AM.get());
    if (mlir::failed(pm.run(module))) {
      ARTS_ERROR("Error when running EDT transformations");
      module->dump();
      return;
    }
  }

  if (stopAt == PipelineStage::EdtTransforms)
    return;

  /// Db creation and optimization
  {
    PassManager pm(&context);
    setupDb(pm, AM.get(), true, ExportJson);
    if (mlir::failed(pm.run(module))) {
      ARTS_ERROR("Error when running db passes");
      module->dump();
      return;
    }
  }

  if (stopAt == PipelineStage::Db)
    return;

  /// EDT optimizations
  {
    PassManager pm(&context);
    setupEdtOptimizations(pm, AM.get());
    if (mlir::failed(pm.run(module))) {
      ARTS_ERROR("Error when running EDT optimizations");
      module->dump();
      return;
    }
  }
  if (stopAt == PipelineStage::EdtOptimizations)
    return;

  /// Create epochs
  {
    PassManager pm(&context);
    setupEpochs(pm, AM.get());
    if (mlir::failed(pm.run(module))) {
      ARTS_ERROR("Error when creating epochs");
      module->dump();
      return;
    }
  }

  if (stopAt == PipelineStage::Epochs)
    return;

  /// Pre-lowering
  {
    PassManager pm(&context);
    setupPreLowering(pm, AM.get());
    if (mlir::failed(pm.run(module))) {
      ARTS_ERROR("Error when running pre-lowering");
      module->dump();
      return;
    }
  }

  if (stopAt == PipelineStage::PreLowering)
    return;

  /// Convert ARTS to LLVM
  {
    PassManager pm(&context);
    setupArtsToLLVM(pm, Debug);
    if (mlir::failed(pm.run(module))) {
      ARTS_ERROR("Error when converting ARTS to LLVM");
      module->dump();
      return;
    }
  }

  if (stopAt == PipelineStage::ArtsToLLVM)
    return;
}

//===----------------------------------------------------------------------===//
// Main Function
//===----------------------------------------------------------------------===//
int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "MLIR Optimization Driver\n");

  /// Set up the dialect registry and MLIR context.
  DialectRegistry registry;
  registerDialects(registry);
  MLIRContext context(registry);
  initializeContext(context);

  /// Open the input file.
  auto file = openInputFile(InputFilename);
  if (!file) {
    ARTS_ERROR("Could not open input file: " << InputFilename);
    return 1;
  }

  /// Parse the input module.
  auto module = parseSourceString<ModuleOp>(file->getBuffer(), &context);
  if (!module) {
    ARTS_ERROR("Could not parse input file");
    return 1;
  }

  /// Run the pass pipeline once.
  setupPassManager(module.get(), context, StopAt);

  /// Translate the optimized module to LLVM IR and write output.
  if (EmitLLVM) {
    llvm::LLVMContext llvmContext;
    auto llvmModule = translateModuleToLLVMIR(module.get(), llvmContext);
    if (!llvmModule) {
      module->dump();
      ARTS_ERROR("Failed to emit LLVM IR");
      return -1;
    }
    std::string llvmIR;
    llvm::raw_string_ostream llvmStream(llvmIR);
    llvmModule->print(llvmStream, nullptr);

    auto output = openOutputFile(OutputFilename);
    if (!output) {
      ARTS_ERROR("Could not open output file: " << OutputFilename);
      return 1;
    }
    output->os() << llvmStream.str();
    output->keep();
  } else {
    /// Otherwise, print the final MLIR module.
    auto output = openOutputFile(OutputFilename);
    if (!output) {
      ARTS_ERROR("Could not open output file: " << OutputFilename);
      return 1;
    }
    module->print(output->os());
    output->keep();
  }

  return 0;
}
