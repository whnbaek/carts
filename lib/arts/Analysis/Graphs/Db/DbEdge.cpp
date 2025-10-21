///==========================================================================
/// File: DbEdge.cpp
/// Implementation of Db edges for graph analysis.
///==========================================================================

#include "arts/Analysis/Graphs/Db/DbEdge.h"
#include "arts/Analysis/Graphs/Base/NodeBase.h"

using namespace mlir::arts;

DbChildEdge::DbChildEdge(NodeBase *from, NodeBase *to) : from(from), to(to) {
  assert(from && "Source node cannot be null");
  assert(to && "Destination node cannot be null");
}

void DbChildEdge::print(llvm::raw_ostream &os) const {
  os << "DB CHILD EDGE:\n"
     << "  - From: " << from->getHierId() << "\n"
     << "  - To: " << to->getHierId() << "\n";
}

static llvm::StringRef depTypeToStr(DbDepType t) {
  switch (t) {
  case DbDepType::RAW:
    return "RAW";
  case DbDepType::WAR:
    return "WAR";
  case DbDepType::WAW:
    return "WAW";
  case DbDepType::RAR:
    return "RAR";
  }
  return "DEP";
}

DbDepEdge::DbDepEdge(NodeBase *from, NodeBase *to, DbDepType depType)
    : from(from), to(to), depType(depType), typeStr(depTypeToStr(depType)) {
  assert(from && "Source node cannot be null");
  assert(to && "Destination node cannot be null");
}

void DbDepEdge::print(llvm::raw_ostream &os) const {
  os << "DB DEP EDGE [" << typeStr << "]:\n"
     << "  - From: " << from->getHierId() << "\n"
     << "  - To: " << to->getHierId() << "\n";
}
