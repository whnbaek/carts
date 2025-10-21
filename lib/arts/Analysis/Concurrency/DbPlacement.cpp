///==========================================================================
/// File: DbPlacement.cpp
/// Defines greedy DbAlloc placement scorer for optimal memory allocation.
///==========================================================================

#include "arts/Analysis/Concurrency/DbPlacement.h"
#include "arts/Analysis/Graphs/Db/DbNode.h"

using namespace mlir;
using namespace mlir::arts;

SmallVector<std::string, 8>
DbPlacementHeuristics::makeNodeNames(unsigned count) {
  SmallVector<std::string, 8> out;
  out.reserve(count);
  for (unsigned i = 0; i < count; ++i)
    out.emplace_back("N" + std::to_string(i));
  return out;
}

SmallVector<DbPlacementDecision, 16>
DbPlacementHeuristics::compute(func::FuncOp func,
                               const SmallVector<std::string, 8> &nodes) {
  SmallVector<DbPlacementDecision, 16> decisions;
  if (!dbGraph)
    return decisions;

  /// Count readers per node for each DbAlloc based on EDT node hint
  llvm::DenseMap<Operation *, llvm::StringMap<uint64_t>> allocToNodeReads;

  func.walk([&](EdtOp edt) {
    auto edtNodeAttr = edt->getAttrOfType<IntegerAttr>("node");
    std::string n = (edtNodeAttr ? ("N" + std::to_string(edtNodeAttr.getInt()))
                                 : std::string("N0"));
    edt.walk([&](DbAcquireOp acq) {
      auto *acqNode = dbGraph->getDbAcquireNode(acq);
      DbAllocNode *parent = acqNode->getParent();
      assert(parent && "Parent allocation not found");
      allocToNodeReads[parent->getOp()][n] += 1;
    });
  });

  /// Build decisions
  dbGraph->forEachNode([&](NodeBase *node) {
    if (auto *an = dyn_cast<DbAllocNode>(node)) {
      Operation *allocOp = an->getOp();
      DbPlacementDecision d;
      d.dbAllocOp = allocOp;
      auto &counts = allocToNodeReads[allocOp];
      uint64_t best = 0;
      std::string bestNode = "N0";
      for (auto &nn : nodes) {
        uint64_t r = counts.lookup(nn);
        d.candidates.push_back({nn, (double)r, r});
        if (r > best) {
          best = r;
          bestNode = nn;
        }
      }
      d.chosenNode = bestNode;
      decisions.push_back(std::move(d));
    }
  });

  return decisions;
}

void DbPlacementHeuristics::exportToJson(
    func::FuncOp func, const SmallVector<DbPlacementDecision, 16> &decisions,
    llvm::raw_ostream &os) const {
  os << "{\n  \"function\": \"" << func.getName()
     << "\",\n  \"db_placement\": [\n";
  bool first = true;
  for (auto &d : decisions) {
    if (!first)
      os << ",\n";
    first = false;
    os << "    {\"alloc\": \"" << d.dbAllocOp << "\", \"chosen\": \""
       << d.chosenNode << "\", \"candidates\": [";
    bool fc = true;
    for (auto &c : d.candidates) {
      if (!fc)
        os << ",";
      fc = false;
      os << "{\"node\": \"" << c.node << "\", \"reads\": " << c.readers << "}";
    }
    os << "]}";
  }
  os << "\n  ]\n}\n";
}
