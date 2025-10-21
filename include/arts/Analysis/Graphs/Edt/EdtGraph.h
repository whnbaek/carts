///==========================================================================
/// File: EdtGraph.h
/// Defines EdtGraph derived from GraphBase for EDT analysis.
///==========================================================================

#ifndef ARTS_ANALYSIS_GRAPHS_EDT_EDTGRAPH_H
#define ARTS_ANALYSIS_GRAPHS_EDT_EDTGRAPH_H

#include "arts/Analysis/Graphs/Base/EdgeBase.h"
#include "arts/Analysis/Graphs/Base/GraphBase.h"
#include "arts/Analysis/Graphs/Base/NodeBase.h"
#include "arts/Analysis/Graphs/Db/DbGraph.h"
#include "arts/Analysis/Graphs/Edt/EdtNode.h"
#include "arts/ArtsDialect.h" // For EdtOp
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace mlir {
namespace arts {

class EdtDepEdge;

/// Represents task dependencies with edges labeled by data blocks.
class EdtGraph : public GraphBase {
public:
  EdtGraph(func::FuncOp func, DbGraph *dbGraph);

  void build() override;
  void buildNodesOnly() override;
  void invalidate() override;
  void print(llvm::raw_ostream &os) override;
  void exportToJson(llvm::raw_ostream &os,
                    bool includeAnalysis = false) const override;
  NodeBase *getEntryNode() const override;
  NodeBase *getOrCreateNode(Operation *op) override;
  NodeBase *getNode(Operation *op) const override;
  EdtNode *getEdtNode(EdtOp edt) const;
  void forEachNode(const std::function<void(NodeBase *)> &fn) const override;
  bool addEdge(NodeBase *from, NodeBase *to, EdgeBase *edge) override;

  /// Edt-specific methods
  bool isEdtReachable(EdtOp from, EdtOp to);
  SmallVector<NodeBase *, 4> getDbDeps(EdtOp edt) const;
  func::FuncOp getFunction() const { return func; }
  bool hasDbGraph() const { return dbGraph != nullptr; }
  DbGraph *getDbGraph() const { return dbGraph; }

  /// For GraphTraits iterators
  NodesIterator nodesBegin() override { return nodes.begin(); }
  NodesIterator nodesEnd() override { return nodes.end(); }
  ChildIterator childBegin(NodeBase *node) override;
  ChildIterator childEnd(NodeBase *node) override;

private:
  func::FuncOp func;
  DbGraph *dbGraph;
  DenseMap<EdtOp, std::unique_ptr<EdtNode>> edtNodes;
  SmallVector<NodeBase *, 8> nodes;
  /// Cache of children per node for GraphBase child iterators.
  DenseMap<NodeBase *, SmallVector<NodeBase *, 8>> childrenCache;

  /// Ensure DenseMap header is considered directly used
  using __EnsureDenseMapUsed = llvm::DenseMap<int, int>;

  /// Private helpers
  void collectNodes();
  void buildDependencies();
  void populateChildrenCache(NodeBase *node);
};

} // namespace arts
} // namespace mlir

// GraphTraits specialization for EdtGraph
namespace llvm {
template <>
struct GraphTraits<mlir::arts::EdtGraph *>
    : public BaseGraphTraits<mlir::arts::EdtGraph> {
  static auto getTo(mlir::arts::EdgeBase *edge) { return edge->getTo(); }

  static ChildIteratorType child_begin(NodeRef N) {
    return map_iterator(N->getOutEdges().begin(), &getTo);
  }
  static ChildIteratorType child_end(NodeRef N) {
    return map_iterator(N->getOutEdges().end(), &getTo);
  }
};

} // namespace llvm

#endif // ARTS_ANALYSIS_GRAPHS_EDT_EDTGRAPH_H