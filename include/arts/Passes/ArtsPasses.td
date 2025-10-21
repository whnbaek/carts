#ifndef CARTS_PASSES
#define CARTS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def ArtsInliner : Pass<"arts-inliner", "mlir::ModuleOp"> {
  let summary = "Inline functions in the program";
  let dependentDialects = [
    "memref::MemRefDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
    "arts::ArtsDialect",
    "polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createArtsInlinerPass()";
}

def CanonicalizeMemrefs : Pass<"canonicalize-memrefs", "mlir::ModuleOp"> {
  let summary = "Canonicalize memref allocations and accesses";
  let description = [{
    This pass canonicalizes memref allocations and accesses for downstream passes.
    It converts nested array-of-pointer allocations to multi-dimensional memrefs and
    rewrites pointer-based accesses to direct memref indexed accesses.
  }];
  let dependentDialects = [
    "memref::MemRefDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
    "arts::ArtsDialect",
    "arith::ArithDialect",
    "polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createCanonicalizeMemrefsPass()";
}


def ConvertOpenMPToArts : Pass<"convert-openmp-to-arts", "mlir::ModuleOp"> {
  let summary = "Convert OpenMP functions to ARTS";
  let dependentDialects = [
    "memref::MemRefDialect",
    "omp::OpenMPDialect",
    "LLVM::LLVMDialect",
    "arts::ArtsDialect",
    "polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createConvertOpenMPtoArtsPass()";
}

def Db : Pass<"db", "mlir::ModuleOp"> {
  let summary = "Analyze and optimize the datablocks in the program";
  let dependentDialects = [
    "memref::MemRefDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
    "arts::ArtsDialect",
    "arith::ArithDialect",
    "polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createDbPass(nullptr, false)";
  let options = [
    Option<"exportJson", "export-json", "bool", /*default=*/"false",
           "Export DB analysis to JSON files in output/ directory.">,
  ];
}

def CreateDbs : Pass<"create-dbs", "mlir::ModuleOp"> {
  let summary = "Identify Dbs in the program";
  let dependentDialects = [
    "memref::MemRefDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
    "arts::ArtsDialect",
    "arith::ArithDialect",
    "polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createCreateDbsPass(true)";
  let options = [
    Option<"identifyDbs", "identify-dbs", "bool", /*default=*/"true",
           "Automatically identify the datablocks.">,
  ];
}

def CanonicalizeDbs : Pass<"canonicalize-dbs", "mlir::ModuleOp"> {
  let summary = "Canonicalize ARTS DataBlock handles and uses to typed memrefs";
  let description = [{
    This pass rewrites datablock allocations and accesses into a canonical form
    that makes size payloads explicit while keeping typed memrefs so that DB
    analyses and optimizations can reason about the layout.
  }];
  let dependentDialects = [
    "memref::MemRefDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
    "arts::ArtsDialect",
    "arith::ArithDialect",
    "polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createCanonicalizeDbsPass()";
}

def DbLowering : Pass<"db-lowering", "mlir::ModuleOp"> {
  let summary = "Lower ARTS DataBlock handles to opaque pointer form";
  let description = [{
    This pass converts ARTS DataBlocks from typed memrefs to the opaque pointer
    representation required by the runtime. It assumes the datablocks are already
    in canonical form.
  }];
  let dependentDialects = [
    "memref::MemRefDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
    "arts::ArtsDialect",
    "arith::ArithDialect",
    "polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createDbLoweringPass()";
}

def Edt : Pass<"edt", "mlir::ModuleOp"> {
  let summary = "Optimize EDTs";
  let dependentDialects = [
    "memref::MemRefDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
    "arts::ArtsDialect",
    "arith::ArithDialect",
    "polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createEdtPass(nullptr, true)";
  let options = [
    Option<"runAnalysis", "run-analysis", "bool", /*default=*/"true",
           "Enable EDT analysis.">
  ];
}

def Concurrency : Pass<"concurrency", "mlir::ModuleOp"> {
  let summary = "Build EDT concurrency graph and apply concurrency-driven cleanups";
  let dependentDialects = [
    "func::FuncDialect",
    "arts::ArtsDialect"
  ];
  let constructor = "mlir::arts::createConcurrencyPass(nullptr)";
}

def CreateEpochs : Pass<"create-epochs", "mlir::ModuleOp"> {
  let summary = "Analyzes the program and creates ARTS epochs";
  let dependentDialects = [
    "memref::MemRefDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
    "arts::ArtsDialect",
    "arith::ArithDialect",
    "polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createCreateEpochsPass()";
}

def ConvertArtsToLLVM : Pass<"convert-arts-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert ARTS Dialect to LLVM Dialect";
  let dependentDialects = [
    "memref::MemRefDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
    "arts::ArtsDialect",
    "arith::ArithDialect",
    "polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createConvertArtsToLLVMPass()";
  let options = [
    Option<"debug", "debug", "bool", /*default=*/"false",
           "Enable debug mode.">,
  ];
}

def EdtInvariantCodeMotion : Pass<"edt-invariant-code-motion", "mlir::ModuleOp"> {
  let summary = "Hoist invariant operations from arts edt regions";
  let dependentDialects = [
    "arts::ArtsDialect"
  ];
  let constructor = "mlir::arts::createEdtInvariantCodeMotionPass()";
}

def EdtPtrRematerialization : Pass<"edt-ptr-rematerialization", "mlir::ModuleOp"> {
  let summary = "Rematerialize pointer dependencies in EDTs.";
  let description = [{
    This pass rematerializes pointer dependencies in event-driven tasks (EDTs) 
    to optimize memory usage and data movement.
  }];
  let constructor = "mlir::arts::createEdtPtrRematerializationPass()";
}

def EpochLowering : Pass<"epoch-lowering", "mlir::ModuleOp"> {
  let summary = "Lower ARTS epochs and propagate epoch GUIDs to EDTs";
  let dependentDialects = [
    "memref::MemRefDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
    "arts::ArtsDialect",
    "arith::ArithDialect",
    "polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createEpochLoweringPass()";
}

def ParallelEdtLowering : Pass<"parallel-edt-lowering", "mlir::ModuleOp"> {
  let summary = "Lower ARTS parallel EDTs to task graphs";
  let description = [{
    This pass transforms parallel EDT operations that contain loop constructs into
    epoch-wrapped task graphs. It must execute before datablock lowering to ensure
    proper dependency management.
  }];
  let dependentDialects = [
    "memref::MemRefDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
    "arts::ArtsDialect",
    "arith::ArithDialect",
    "scf::SCFDialect",
    "polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createParallelEdtLoweringPass()";
}

def EdtLowering : Pass<"edt-lowering", "mlir::ModuleOp"> {
  let summary = "Lower ARTS EDTs to LLVM";
  let dependentDialects = [
    "memref::MemRefDialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
    "arts::ArtsDialect",
    "arith::ArithDialect",
    "polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createEdtLoweringPass()";
}

#endif // CARTS_PASSES
