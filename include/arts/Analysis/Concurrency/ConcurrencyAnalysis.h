///==========================================================================
/// File: ConcurrencyAnalysis.h
///
/// This file defines the central analysis for concurrency operations.
///==========================================================================

#ifndef ARTS_CONCURRENCY_ANALYSIS_H
#define ARTS_CONCURRENCY_ANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>

namespace mlir {
namespace arts {

class ArtsAnalysisManager;
class EdtGraph;
class ConcurrencyGraph;

/// Central manager for concurrency-related analyses across a module.
class ConcurrencyAnalysis {
public:
  ConcurrencyAnalysis(ArtsAnalysisManager &AM);
  ~ConcurrencyAnalysis();

  /// Get or create concurrency graph for a function
  ConcurrencyGraph &getOrCreateConcurrencyGraph(func::FuncOp func);

  /// Invalidate concurrency graph for a specific function
  bool invalidateConcurrencyGraph(func::FuncOp func);

  /// Invalidate all concurrency graphs
  void invalidate();

  /// Debug printing
  void print(func::FuncOp func, llvm::raw_ostream &os);

  /// Access the ArtsAnalysisManager
  ArtsAnalysisManager &getAM() { return AM; }

private:
  ArtsAnalysisManager &AM;
  llvm::DenseMap<func::FuncOp, std::unique_ptr<ConcurrencyGraph>>
      concurrencyGraphs;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_CONCURRENCY_ANALYSIS_H
