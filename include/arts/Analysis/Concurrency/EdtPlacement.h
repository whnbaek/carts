///==========================================================================
/// File: EdtPlacement.h
/// Defines heuristics for EDT node placement
///==========================================================================

#ifndef ARTS_ANALYSIS_EDT_EDTPLACEMENT_H
#define ARTS_ANALYSIS_EDT_EDTPLACEMENT_H

#include "arts/Analysis/Edt/EdtAnalysis.h"
#include "arts/Analysis/Graphs/Edt/EdtGraph.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace arts {

struct PlacementNodeScore {
  std::string node;
  double score = 0.0;
  double locality = 0.0;
  double contention = 0.0;
  double load = 0.0;
  double risk = 0.0;
};

struct PlacementDecision {
  Operation *edtOp = nullptr;
  std::string chosenNode;
  SmallVector<PlacementNodeScore, 4> candidates;
};

/// Greedy EDT placement using EdtAnalysis affinity metrics only.
/// Analysis-only: does not mutate IR; emits debug decisions for future
/// integration with external placement orchestrators.
class EdtPlacementHeuristics {
public:
  EdtPlacementHeuristics(EdtGraph *graph, EdtAnalysis *analysis)
      : edtGraph(graph), edtAnalysis(analysis) {}

  /// Build a default node list: N0..N{count-1}
  static SmallVector<std::string, 8> makeNodeNames(unsigned count);

  /// Compute placement decisions for provided node set (names only)
  SmallVector<PlacementDecision, 16>
  compute(const SmallVector<std::string, 8> &nodes);

  /// Export placement JSON for a function
  void exportToJson(func::FuncOp func,
                    const SmallVector<PlacementDecision, 16> &decisions,
                    llvm::raw_ostream &os) const;

private:
  EdtGraph *edtGraph = nullptr;
  EdtAnalysis *edtAnalysis = nullptr;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_ANALYSIS_EDT_EDTPLACEMENT_H
