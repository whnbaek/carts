///==========================================================================
/// File: DbDataFlowAnalysis.h
/// Environment-based dataflow analysis for building DDG.
///==========================================================================

#ifndef ARTS_ANALYSIS_DB_DBDATAFLOWANALYSIS_H
#define ARTS_ANALYSIS_DB_DBDATAFLOWANALYSIS_H

#include "arts/Analysis/Graphs/Db/DbNode.h"
#include "arts/ArtsDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Region;

namespace func {
class FuncOp;
}

namespace scf {
class ForOp;
class IfOp;
} // namespace scf

namespace arts {

class DbGraph;
class DbAcquireNode;
class DbAllocOp;
class DbAnalysis;
class DbAliasAnalysis;

/// Environment-based dataflow analysis for building DDG.
/// Tracks reaching definitions through control flow to create dependency edges.
class DbDataFlowAnalysis {
public:
  /// Environment maps each allocation node to its latest acquire definitions
  using Environment = DenseMap<DbAllocNode *, llvm::SetVector<DbAcquireNode *>>;

  DbDataFlowAnalysis();
  ~DbDataFlowAnalysis() = default;

  void initialize(DbGraph *graph, DbAnalysis *analysis);
  void runAnalysis(func::FuncOp func);

private:
  /// Process different region types
  std::pair<Environment, bool> processRegion(Region &region, Environment &env);
  std::pair<Environment, bool> processEdt(EdtOp edtOp, Environment &env);
  std::pair<Environment, bool> processFor(scf::ForOp forOp, Environment &env);
  std::pair<Environment, bool> processIf(scf::IfOp ifOp, Environment &env);

  /// Find reaching definitions for a given acquire
  SmallVector<DbAcquireNode *> findDefinitions(DbAcquireNode *acquire,
                                               Environment &env);

  /// Merge environments from different control flow paths
  Environment mergeEnvironments(const Environment &env1,
                                const Environment &env2);

  /// Helper that unions definitions from `source` into `target` and returns
  /// true if `target` changed.
  bool unionInto(Environment &target, const Environment &source);

  /// Check if two acquires may have a dependency
  bool mayDepend(DbAcquireNode *producer, DbAcquireNode *consumer);

  DbGraph *graph = nullptr;
  DbAnalysis *analysis = nullptr;
  DbAliasAnalysis *aliasAA = nullptr;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_ANALYSIS_DB_DBDATAFLOWANALYSIS_H
