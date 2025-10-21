///==========================================================================
/// File: GraphTrait.h
/// Shared LLVM GraphTraits specializations for ARTS graphs.
///==========================================================================

#ifndef ARTS_ANALYSIS_GRAPHS_GRAPHTRAITS_H
#define ARTS_ANALYSIS_GRAPHS_GRAPHTRAITS_H

#include "arts/Analysis/Graphs/Base/EdgeBase.h"
#include "arts/Analysis/Graphs/Base/GraphBase.h"
#include "arts/Analysis/Graphs/Base/NodeBase.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"

namespace llvm {

/// Base traits for all ARTS graphs.
template <typename GraphTy> struct BaseGraphTraits {
  using NodeRef = ::mlir::arts::NodeBase *;
  using ChildIteratorType = mapped_iterator<
      ::llvm::DenseSet<::mlir::arts::EdgeBase *>::const_iterator,
      ::mlir::arts::NodeBase *(*)(::mlir::arts::EdgeBase *)>;

  static ::mlir::arts::NodeBase *getTo(::mlir::arts::EdgeBase *edge) {
    return edge->getTo();
  }

  static NodeRef getEntryNode(GraphTy *G) { return G->getEntryNode(); }

  static typename ::mlir::arts::GraphBase::NodesIterator
  nodes_begin(GraphTy *G) {
    return G->nodesBegin();
  }

  static typename ::mlir::arts::GraphBase::NodesIterator nodes_end(GraphTy *G) {
    return G->nodesEnd();
  }

  static ChildIteratorType child_begin(NodeRef N) {
    return map_iterator(N->getOutEdges().begin(), &getTo);
  }

  static ChildIteratorType child_end(NodeRef N) {
    return map_iterator(N->getOutEdges().end(), &getTo);
  }
};

/// Specialization for GraphBase*
template <>
struct GraphTraits<::mlir::arts::GraphBase *>
    : public BaseGraphTraits<::mlir::arts::GraphBase> {};
} // namespace llvm

#endif // ARTS_ANALYSIS_GRAPHS_GRAPHTRAITS_H
