///==========================================================================
/// File: GraphBase.h
/// Abstract base class for graphs in ARTS analysis.
///==========================================================================

#ifndef ARTS_ANALYSIS_GRAPHS_GRAPHBASE_H
#define ARTS_ANALYSIS_GRAPHS_GRAPHBASE_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace mlir {
namespace arts {

class NodeBase;
class EdgeBase;

/// Abstract base class for all graphs (e.g., DbGraph, EdtGraph).
/// Provides common interface for building, invalidating, printing, and
/// exporting.
class GraphBase {
public:
  virtual ~GraphBase() = default;

  /// Build the graph from the IR (e.g., walk func to collect nodes/edges and
  /// build dependencies).
  virtual void build() = 0;

  /// Build the graph from the IR (e.g., walk func to collect nodes/edges).
  virtual void buildNodesOnly() = 0;

  /// Invalidate and clear the graph state.
  virtual void invalidate() = 0;

  /// Print summary of the graph.
  virtual void print(llvm::raw_ostream &os) = 0;

  /// Export a JSON representation of the graph.
  virtual void exportToJson(llvm::raw_ostream &os,
                            bool includeAnalysis = false) const = 0;

  /// Get the entry node (e.g., first alloc or root task).
  virtual NodeBase *getEntryNode() const = 0;

  /// Get or create a node for the given operation
  virtual NodeBase *getOrCreateNode(Operation *op) = 0;

  /// Get a node for the given operation if it exists.
  virtual NodeBase *getNode(Operation *op) const = 0;

  /// Get the number of nodes in the graph.
  virtual size_t size() const { return nodes.size(); }

  /// Check if the graph is empty (has no nodes).
  virtual bool isEmpty() const { return nodes.empty(); }

  /// Iterate over all nodes.
  virtual void forEachNode(const std::function<void(NodeBase *)> &fn) const = 0;

  /// Add an edge between nodes (returns true if inserted).
  virtual bool addEdge(NodeBase *from, NodeBase *to, EdgeBase *edge) = 0;

  /// For GraphTraits: iterators over nodes.
  using NodesIterator = SmallVector<NodeBase *, 4>::iterator;
  virtual NodesIterator nodesBegin() = 0;
  virtual NodesIterator nodesEnd() = 0;

  /// For children (outgoing neighbors).
  using ChildIterator = SmallVector<NodeBase *, 4>::iterator;
  virtual ChildIterator childBegin(NodeBase *node) = 0;
  virtual ChildIterator childEnd(NodeBase *node) = 0;

protected:
  // Protected members for derived classes to manage state.
  SmallVector<std::unique_ptr<NodeBase>> nodes;
  DenseMap<std::pair<NodeBase *, NodeBase *>, std::unique_ptr<EdgeBase>> edges;
  bool isBuilt = false;
  bool needsRebuild = true;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_ANALYSIS_GRAPHS_GRAPHBASE_H
