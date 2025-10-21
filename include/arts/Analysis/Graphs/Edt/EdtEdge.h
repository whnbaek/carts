///==========================================================================
/// File: EdtEdge.h
/// Defines EDT-specific edges for graph analysis.
///==========================================================================

#ifndef ARTS_ANALYSIS_GRAPHS_EDT_EDTEDGE_H
#define ARTS_ANALYSIS_GRAPHS_EDT_EDTEDGE_H

#include "arts/Analysis/Graphs/Base/EdgeBase.h"
#include "arts/Analysis/Graphs/Db/DbNode.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace arts {

class EdtDepEdge : public EdgeBase {
public:
  EdtDepEdge(NodeBase *from, NodeBase *to, NodeBase *dbNode);

  NodeBase *getFrom() const override { return from; }
  NodeBase *getTo() const override { return to; }
  EdgeKind getKind() const override { return EdgeKind::Dep; }
  StringRef getType() const override { return dbNode->getHierId(); }
  void print(llvm::raw_ostream &os) const override;

  static bool classof(const EdgeBase *E) {
    return E->getKind() == EdgeKind::Dep;
  }

private:
  NodeBase *from;
  NodeBase *to;
  /// DbGraph node causing dependency
  NodeBase *dbNode;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_ANALYSIS_GRAPHS_EDT_EDTEDGE_H