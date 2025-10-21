///==========================================================================
/// File: EdtAnalysis.h
///
/// This pass performs a comprehensive analysis on arts.edt operations.
/// It also provides helpers to analyze regions and find dependencies,
/// parameters, and constants.
///==========================================================================

#ifndef CARTS_ANALYSIS_EDTANALYSIS_H
#define CARTS_ANALYSIS_EDTANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"

#include "arts/Analysis/Edt/EdtInfo.h"
#include "arts/Analysis/Graphs/Edt/EdtGraph.h"

// Forward declarations
namespace mlir {
namespace arts {
class ArtsAnalysisManager;
class DbGraph;

//==========================================================================
// EdtAnalysis: per-EDT summaries and pairwise affinity metrics
//==========================================================================

class EdtAnalysis {
public:
  EdtAnalysis(ArtsAnalysisManager &AM);

  /// Build summaries for all EDTs in the module.
  void analyze();

  /// Debug printing
  void print(func::FuncOp func, llvm::raw_ostream &os);
  void toJson(func::FuncOp func, llvm::raw_ostream &os);

  /// Graph accessor
  EdtGraph &getOrCreateEdtGraph(func::FuncOp func);

  /// Invalidate EDT graph for a specific function
  bool invalidateGraph(func::FuncOp func);

  /// Invalidate internal caches
  void invalidate();

private:
  ArtsAnalysisManager &AM;

  /// Order index per EDT in a function
  DenseMap<EdtOp, unsigned> edtOrderIndex;

  /// Per-function graph caches
  llvm::DenseMap<func::FuncOp, std::unique_ptr<EdtGraph>> edtGraphs;

  void analyzeFunc(func::FuncOp func);
};

} // namespace arts
} // namespace mlir

#endif // CARTS_ANALYSIS_EDTANALYSIS_H