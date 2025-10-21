///==========================================================================
/// File: ArtsAnalysisManager.cpp
///
/// This file implements the ArtsAnalysisManager for centralized management
/// of all ARTS analysis objects with clean separation from graph management.
///==========================================================================

#include "arts/Analysis/ArtsAnalysisManager.h"
#include "arts/Analysis/Concurrency/ConcurrencyAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;
using namespace mlir::arts;

ArtsAnalysisManager::ArtsAnalysisManager(ModuleOp module,
                                         const std::string &configFile)
    : module(module), configFile(configFile), abstractMachine(configFile) {}

ArtsAnalysisManager::~ArtsAnalysisManager() {}

void ArtsAnalysisManager::invalidate() {
  if (dbAnalysis)
    dbAnalysis->invalidate();
  if (edtAnalysis)
    edtAnalysis->invalidate();
  if (concurrencyAnalysis)
    concurrencyAnalysis->invalidate();
}

DbAnalysis &ArtsAnalysisManager::getDbAnalysis() {
  if (!dbAnalysis)
    dbAnalysis = std::make_unique<DbAnalysis>(*this);
  return *dbAnalysis.get();
}

EdtAnalysis &ArtsAnalysisManager::getEdtAnalysis() {
  if (!edtAnalysis)
    edtAnalysis = std::make_unique<EdtAnalysis>(*this);
  return *edtAnalysis.get();
}

ConcurrencyAnalysis &ArtsAnalysisManager::getConcurrencyAnalysis() {
  if (!concurrencyAnalysis)
    concurrencyAnalysis = std::make_unique<ConcurrencyAnalysis>(*this);
  return *concurrencyAnalysis.get();
}

DbGraph &ArtsAnalysisManager::getDbGraph(func::FuncOp func) {
  return getDbAnalysis().getOrCreateGraph(func);
}

EdtGraph &ArtsAnalysisManager::getEdtGraph(func::FuncOp func) {
  return getEdtAnalysis().getOrCreateEdtGraph(func);
}

ConcurrencyGraph &ArtsAnalysisManager::getConcurrencyGraph(func::FuncOp func) {
  return getConcurrencyAnalysis().getOrCreateConcurrencyGraph(func);
}

bool ArtsAnalysisManager::invalidateFunction(func::FuncOp func) {
  bool invalidated = false;
  if (dbAnalysis)
    invalidated |= dbAnalysis->invalidateGraph(func);
  if (edtAnalysis)
    invalidated |= edtAnalysis->invalidateGraph(func);
  if (concurrencyAnalysis)
    invalidated |= concurrencyAnalysis->invalidateConcurrencyGraph(func);
  return invalidated;
}

void ArtsAnalysisManager::print(llvm::raw_ostream &os) {
  os << "ArtsAnalysisManager for module: "
     << (module.getName() ? module.getName()->str() : "unnamed") << "\n";

  for (auto func : module.getOps<func::FuncOp>()) {
    os << "  Function: " << func.getName() << "\n";
    os << "    DB Graph: Available\n";
    os << "    EDT Graph: Available\n";
    os << "    Concurrency Graph: Available\n";
  }
}

void ArtsAnalysisManager::exportToJson(llvm::raw_ostream &os,
                                       bool includeAnalysis) {
  os << "{\n";
  os << "  \"module\": \""
     << (module.getName() ? module.getName()->str() : "unnamed") << "\",\n";
  os << "  \"built\": " << (built ? "true" : "false") << ",\n";
  os << "  \"valid\": true";

  /// Export graphs via analysis objects
  os << ",\n  \"functions\": [\n";
  bool first = true;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (!first)
      os << ",\n";
    first = false;
    os << "    {\n";
    os << "      \"function\": \"" << func.getName() << "\",\n";
    os << "      \"graphs\": {\n";
    os << "        \"db\": \"available\",\n";
    os << "        \"edt\": \"available\",\n";
    os << "        \"concurrency\": \"available\"\n";
    os << "      }\n";
    os << "    }";
  }
  os << "\n  ]";

  os << "\n}\n";
}