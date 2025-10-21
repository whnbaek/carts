///==========================================================================
/// File: EdtEdge.cpp
/// Implementation of EDT edges for graph analysis.
///==========================================================================

#include "arts/Analysis/Graphs/Edt/EdtEdge.h"

using namespace mlir::arts;

EdtDepEdge::EdtDepEdge(NodeBase *from, NodeBase *to, NodeBase *dbNode)
    : from(from), to(to), dbNode(dbNode) {
  assert(from && "Source node cannot be null");
  assert(to && "Destination node cannot be null");
  assert(dbNode && "DB node cannot be null");
}

void EdtDepEdge::print(llvm::raw_ostream &os) const {
  os << "EDT DEPENDENCY EDGE:\n"
     << "  - From: " << from->getHierId() << "\n"
     << "  - To: " << to->getHierId() << "\n"
     << "  - DB: " << dbNode->getHierId() << "\n";
}