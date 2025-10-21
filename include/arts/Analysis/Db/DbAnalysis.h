///==========================================================================
/// File: DbAnalysis.h
///
/// This file defines the central analysis for DB operations.
///==========================================================================

#ifndef ARTS_ANALYSIS_DB_DBANALYSIS_H
#define ARTS_ANALYSIS_DB_DBANALYSIS_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"

// Forward declarations
namespace mlir {
namespace arts {
class DbGraph;
class DbAliasAnalysis;
struct LoopAnalysis;
class DbDataFlowAnalysis;
class DbAcquireOp;
class ArtsAnalysisManager;
} // namespace arts
} // namespace mlir

namespace mlir {
namespace arts {

/// Central manager for DB-related analyses across a module.
class ArtsAnalysisManager;

class DbAnalysis {
public:
  DbAnalysis(ArtsAnalysisManager &AM);

  ~DbAnalysis();

  /// Get or create DbGraph for a function, building if needed.
  DbGraph &getOrCreateGraph(func::FuncOp func);

  /// Get or create DbGraph without edges/metrics (nodes only).
  DbGraph &getOrCreateGraphNodesOnly(func::FuncOp func);

  /// Invalidate graph for a function.
  bool invalidateGraph(func::FuncOp func);

  /// Invalidate all graphs
  void invalidate();

  /// Print analysis for a function.
  void print(func::FuncOp func);

  /// Access analyses
  DbAliasAnalysis *getAliasAnalysis() { return dbAliasAnalysis.get(); }
  LoopAnalysis *getLoopAnalysis() { return loopAnalysis.get(); }
  DbDataFlowAnalysis *getDataFlowAnalysis() { return dbDataFlowAnalysis.get(); }

private:
  ArtsAnalysisManager &AM;
  llvm::DenseMap<func::FuncOp, std::unique_ptr<DbGraph>> functionGraphMap;
  std::unique_ptr<DbAliasAnalysis> dbAliasAnalysis;
  std::unique_ptr<LoopAnalysis> loopAnalysis;
  std::unique_ptr<DbDataFlowAnalysis> dbDataFlowAnalysis;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_ANALYSIS_DB_DBANALYSIS_H