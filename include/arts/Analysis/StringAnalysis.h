///==========================================================================
/// File: StringAnalysis.h
///
/// This class analyzes the module and itentifies the set of string memref.
///==========================================================================

#ifndef CARTS_ANALYSIS_STRINGANALYSIS_H
#define CARTS_ANALYSIS_STRINGANALYSIS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace arts {

//===----------------------------------------------------------------------===//
// StringAnalysis
//===----------------------------------------------------------------------===//
struct StringAnalysis {
  explicit StringAnalysis(Operation *module) {
    this->module = cast<ModuleOp>(module);
    run();
  }
  ~StringAnalysis() {}

public:
  bool isStringMemRef(Value value) const { return stringMemRefs.count(value); }
  const DenseSet<Value> &getStringMemRefs() const { return stringMemRefs; }
  const DenseMap<Value, LLVM::GlobalOp> &getGlobalSources() const {
    return globalSources;
  }

  /// Main function that performs analysis.
  void run();

private:
  /// Cached results from the analysis.
  DenseSet<Value> stringMemRefs;
  DenseMap<Value, LLVM::GlobalOp> globalSources;
  ModuleOp module;

  /// Check whether the LLVM global contains a null-terminated string.
  bool isStringGlobal(LLVM::GlobalOp globalOp);

  /// Recursively track users of the given global value.
  void trackGlobalUsers();

  /// Track memrefs that are initialized from string globals.
  void trackMemrefInitializations();

  /// Analyze all call operations for string functions and extract underlying
  /// memrefs.
  void trackCallOps();

  /// Helper to recognize string-related functions.
  bool isStringFunction(StringRef funcName);
};

} // namespace arts
} // namespace mlir

#endif // CARTS_ANALYSIS_STRINGANALYSIS_H