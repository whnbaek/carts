///==========================================================================
/// File: ConcurrencyGraph.h
///
/// This file defines the concurrency graph for EDT analysis.
/// Identifies pairs of EDTs that may execute concurrently.
///==========================================================================

#ifndef ARTS_ANALYSIS_GRAPHS_CONCURRENCY_CONCURRENCYGRAPH_H
#define ARTS_ANALYSIS_GRAPHS_CONCURRENCY_CONCURRENCYGRAPH_H

#include "arts/Analysis/Graphs/Base/GraphBase.h"
#include "arts/Analysis/Graphs/Base/NodeBase.h"
#include "arts/ArtsDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace arts {

// Forward declarations
class EdtGraph;
class ConcurrencyAnalysis;
class ArtsAnalysisManager;

/// Edge between two EDTs that may execute concurrently.
struct ConcurrencyEdge {
  EdtOp *from = nullptr;
  EdtOp *to = nullptr;
  double dataOverlap = 0.0;
  double hazardScore = 0.0;
  bool mayConflict = false;
  double reuseProximity = 0.0;
  double localityScore = 0.0;
  double concurrencyRisk = 0.0;
};

/// Disjoint Set Union for grouping EDTs based on concurrency characteristics
struct DisjointSetUnion {
  llvm::DenseMap<EdtOp, EdtOp> parent;

  EdtOp find(EdtOp x) {
    auto it = parent.find(x);
    if (it == parent.end() || it->second == x)
      return parent[x] = x;
    return it->second = find(it->second);
  }

  void unite(EdtOp a, EdtOp b) {
    a = find(a);
    b = find(b);
    if (a != b)
      parent[b] = a;
  }
};

/// Concurrency graph for EDT analysis.
/// Identifies pairs of EDTs that are not ordered by dependencies and may
/// execute concurrently.
class ConcurrencyGraph : public GraphBase {
public:
  ConcurrencyGraph(func::FuncOp func, ConcurrencyAnalysis *analysis);

  // GraphBase interface
  void build() override;
  void buildNodesOnly() override {
    build();
  } // Same as build for concurrency graph
  void invalidate() override;
  void print(llvm::raw_ostream &os) override;
  void exportToJson(llvm::raw_ostream &os,
                    bool includeAnalysis = false) const override;
  NodeBase *getEntryNode() const override;
  NodeBase *getOrCreateNode(Operation *op) override;
  NodeBase *getNode(Operation *op) const override;
  void forEachNode(const std::function<void(NodeBase *)> &fn) const override;
  bool addEdge(NodeBase *from, NodeBase *to, EdgeBase *edge) override;

  // GraphTraits interface
  NodesIterator nodesBegin() override;
  NodesIterator nodesEnd() override;
  ChildIterator childBegin(NodeBase *node) override;
  ChildIterator childEnd(NodeBase *node) override;

  /// Coloring/grouping methods for EDT and DB placement
  void createPlacementGroups(ModuleOp module);
  void assignGroupAttributes(const DisjointSetUnion &dsu, ModuleOp module);

  /// DB allocation coloring for reuse optimization
  void computeReuseColoring();

private:
  func::FuncOp func;
  ConcurrencyAnalysis *analysis = nullptr;
  EdtGraph *edtGraph = nullptr;
  SmallVector<EdtOp *, 16> edts;
  SmallVector<ConcurrencyEdge, 32> edges;

  // GraphBase requires nodes vector for iteration
  SmallVector<NodeBase *, 8> nodes;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_ANALYSIS_GRAPHS_CONCURRENCY_CONCURRENCYGRAPH_H
