///==========================================================================
/// File: EdtNode.cpp
/// Implementation of EDT nodes for graph analysis.
///==========================================================================

#include "arts/Analysis/Graphs/Edt/EdtNode.h"

using namespace mlir::arts;

EdtNode::EdtNode(EdtOp op) : edtOp(op), opPtr(op.getOperation()) {}

void EdtNode::print(llvm::raw_ostream &os) const {
  os << "EdtNode (" << getHierId() << ")\n";
}