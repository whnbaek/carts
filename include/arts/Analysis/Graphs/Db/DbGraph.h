///==========================================================================
/// File: DbGraph.h
/// Defines DbGraph derived from GraphBase for database operation analysis.
///==========================================================================

#ifndef ARTS_ANALYSIS_GRAPHS_DB_DBGRAPH_H
#define ARTS_ANALYSIS_GRAPHS_DB_DBGRAPH_H

#include "arts/Analysis/Db/DbInfo.h"
#include "arts/Analysis/Graphs/Base/EdgeBase.h"
#include "arts/Analysis/Graphs/Base/GraphBase.h"
#include "arts/Analysis/Graphs/Base/GraphTrait.h"
#include "arts/Analysis/Graphs/Base/NodeBase.h"
#include "arts/ArtsDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include <functional>
#include <memory>
#include <string>

namespace mlir {
namespace arts {

// Forward declarations
class DbAnalysis;

class DbAllocNode;
class DbAcquireNode;
class DbChildEdge;
class DbNode;
class DbEdge;

/// DbGraph: Specialized graph for data blocks (alloc/acquire/release).
class DbGraph : public GraphBase {
public:
  DbGraph(func::FuncOp func, DbAnalysis *analysis);
  ~DbGraph();

  void build() override;
  void buildNodesOnly() override;
  void invalidate() override;
  void print(llvm::raw_ostream &os) override;

  /// Export graph to JSON
  void exportToJson(llvm::raw_ostream &os,
                    bool includeAnalysis = false) const override;
  NodeBase *getEntryNode() const override;
  NodeBase *getOrCreateNode(Operation *op) override;
  NodeBase *getNode(Operation *op) const override;

  /// Retrieve node accessors by concrete op type
  DbAllocNode *getDbAllocNode(DbAllocOp op) const;
  DbAcquireNode *getDbAcquireNode(DbAcquireOp op) const;

  /// Convenience methods for getting or creating specific node types
  DbAllocNode *getOrCreateAllocNode(DbAllocOp op);
  DbAcquireNode *getOrCreateAcquireNode(DbAcquireOp op);

  /// Apply a function to every node (alloc/acquire)
  void forEachNode(const std::function<void(NodeBase *)> &fn) const override;
  void forEachAllocNode(const std::function<void(DbAllocNode *)> &fn) const;
  void forEachAcquireNode(const std::function<void(DbAcquireNode *)> &fn) const;

  /// Edges
  bool addEdge(NodeBase *from, NodeBase *to, EdgeBase *edge) override;
  size_t getEdgeCount() const { return edges.size(); }

  /// Get allocation info for a given allocation operation
  const DbAllocInfo &getAllocInfo(DbAllocOp alloc) const;

  /// For GraphTraits iterators
  bool isEmpty() const override { return nodes.empty(); }
  NodesIterator nodesBegin() override { return nodes.begin(); }
  NodesIterator nodesEnd() override { return nodes.end(); }
  ChildIterator childBegin(NodeBase *node) override;
  ChildIterator childEnd(NodeBase *node) override;

private:
  func::FuncOp func;
  DbAnalysis *analysis;

  /// Node maps
  DenseMap<DbAllocOp, std::unique_ptr<DbAllocNode>> allocNodes;
  DenseMap<DbAcquireOp, DbAcquireNode *> acquireNodeMap;

  /// All nodes
  SmallVector<NodeBase *, 8> nodes;

  unsigned nextAllocId = 1;

  /// Private helpers
  void collectNodes();
  void buildDependencies();
  void computeOpOrder();
  void computeMetrics();
  std::string generateAllocId(unsigned id);

  /// Metrics computation helpers
  void computeAllocMetrics(DbAllocOp alloc, DbAllocNode *allocNode);
  void processAcquireNode(DbAcquireNode *acq, DbAllocInfo &info);
  void computeCriticalSpan(DbAllocInfo &info);
  void computeCriticalPath(DbAllocInfo &info);
  void computeLoopDepth(DbAllocInfo &info);
  void computeLongLivedFlag(DbAllocInfo &info);
  void computeEscapingFlag(DbAllocOp alloc, DbAllocInfo &info);
  void computePeakMetrics();
  void computeReuseCandidates();
  void populateChildrenCache(NodeBase *node);

private:
  DenseMap<Operation *, unsigned> opOrder;
  uint64_t peakLiveDbs = 0;
  unsigned long long peakBytes = 0;

  /// Cache of children per node for GraphBase child iterators.
  DenseMap<NodeBase *, SmallVector<NodeBase *, 8>> childrenCache;

  /// Utilities
  unsigned getOrder(Operation *op) const {
    auto it = opOrder.find(op);
    return it == opOrder.end() ? 0u : it->second;
  }
};

} // namespace arts
} // namespace mlir

namespace llvm {
template <>
struct GraphTraits<mlir::arts::DbGraph *>
    : public BaseGraphTraits<mlir::arts::DbGraph> {};
} // namespace llvm

#endif // ARTS_ANALYSIS_GRAPHS_DB_DBGRAPH_H
