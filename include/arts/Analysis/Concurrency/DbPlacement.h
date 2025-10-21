///==========================================================================
/// File: DbPlacement.h
/// Defines heuristics for DbAlloc node placement
///==========================================================================

#ifndef ARTS_ANALYSIS_DB_DBPLACEMENT_H
#define ARTS_ANALYSIS_DB_DBPLACEMENT_H

#include "arts/Analysis/Graphs/Db/DbGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace arts {;

struct DbPlacementNodeScore {
  std::string node;
  double score = 0.0;
  /// number of acquires by EDTs placed on node
  uint64_t readers = 0; 
  uint64_t writers = 0;
};

struct DbPlacementDecision {
  Operation *dbAllocOp = nullptr;
  std::string chosenNode;
  SmallVector<DbPlacementNodeScore, 4> candidates;
};

/// Heuristics: place DbAlloc on the node with most readers (EDTs) using it,
/// when those EDTs have a preferred node set; otherwise default to N0.
class DbPlacementHeuristics {
public:
  explicit DbPlacementHeuristics(DbGraph *graph) : dbGraph(graph) {}

  static SmallVector<std::string, 8> makeNodeNames(unsigned count);

  SmallVector<DbPlacementDecision, 16>
  compute(func::FuncOp func, const SmallVector<std::string, 8> &nodes);

  void exportToJson(func::FuncOp func,
                    const SmallVector<DbPlacementDecision, 16> &decisions,
                    llvm::raw_ostream &os) const;

private:
  DbGraph *dbGraph = nullptr; // not owned
};

} // namespace arts
} // namespace mlir

#endif // ARTS_ANALYSIS_DB_DBPLACEMENT_H
