///==========================================================================
/// File: EdtAnalysis.cpp
///==========================================================================

#include "arts/Analysis/Edt/EdtAnalysis.h"
#include "arts/Analysis/ArtsAnalysisManager.h"
#include "arts/Analysis/Graphs/Edt/EdtGraph.h"
#include "arts/ArtsDialect.h"
/// Dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"

/// Debug
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

// #include "arts/Utils/ArtsDebug.h"
// ARTS_DEBUG_SETUP(edt_analysis)

using namespace mlir;
using namespace arts;

///==========================================================================
/// EdtAnalysis
///==========================================================================

EdtAnalysis::EdtAnalysis(ArtsAnalysisManager &AM) : AM(AM) {
  assert(AM.getModule() && "Module is required");
}

void EdtAnalysis::analyze() {
  for (auto func : AM.getModule().getOps<func::FuncOp>())
    analyzeFunc(func);
}

void EdtAnalysis::analyzeFunc(func::FuncOp func) {
  unsigned edtIndex = 0;
  EdtGraph &edtGraph = getOrCreateEdtGraph(func);
  func.walk([&](EdtOp edt) {
    auto edtNode = edtGraph.getEdtNode(edt);
    auto &info = edtNode->getInfo();
    info.orderIndex = edtIndex;
    edtOrderIndex[edt] = edtIndex++;
  });
}

void EdtAnalysis::print(func::FuncOp func, llvm::raw_ostream &os) {
  os << "EdtAnalysis for function: " << func.getName() << "\n";

  EdtGraph &edtGraph = getOrCreateEdtGraph(func);
  func.walk([&](EdtOp edt) {
    auto edtNode = edtGraph.getEdtNode(edt);
    const EdtInfo *edtInfo = &edtNode->getInfo();
    if (edtInfo) {
      unsigned edtIndex = edtInfo->orderIndex;
      os << "  EDT #" << edtIndex << ":\n";
      os << "    Total ops: " << edtInfo->totalOps << "\n";
      os << "    Loads: " << edtInfo->numLoads
         << ", Stores: " << edtInfo->numStores << "\n";
      os << "    Calls: " << edtInfo->numCalls << "\n";
      os << "    Max loop depth: " << edtInfo->maxLoopDepth << "\n";
      os << "    Compute/Memory ratio: " << edtInfo->computeToMemRatio << "\n";
      os << "    Bases read: " << edtInfo->basesRead.size()
         << ", written: " << edtInfo->basesWritten.size() << "\n";
      os << "    Bytes read: " << edtInfo->bytesRead
         << ", written: " << edtInfo->bytesWritten << "\n";
    }
  });
}

void EdtAnalysis::toJson(func::FuncOp func, llvm::raw_ostream &os) {
  os << "{\n";
  os << "  \"function\": \"" << func.getName() << "\",\n";
  os << "  \"edts\": [\n";

  bool first = true;
  EdtGraph &edtGraph = getOrCreateEdtGraph(func);
  func.walk([&](Operation *op) {
    if (auto edt = dyn_cast<EdtOp>(op)) {
      auto edtNode = static_cast<EdtNode *>(edtGraph.getNode(op));
      const EdtInfo *edtInfo = &edtNode->getInfo();

      if (edtInfo) {
        if (!first)
          os << ",\n";
        first = false;

        unsigned edtIndex = edtInfo->orderIndex;
        os << "    {\n";
        os << "      \"edtId\": " << edtIndex << ",\n";
        os << "      \"totalOps\": " << edtInfo->totalOps << ",\n";
        os << "      \"numLoads\": " << edtInfo->numLoads << ",\n";
        os << "      \"numStores\": " << edtInfo->numStores << ",\n";
        os << "      \"bytesRead\": " << edtInfo->bytesRead << ",\n";
        os << "      \"bytesWritten\": " << edtInfo->bytesWritten << ",\n";
        os << "      \"maxLoopDepth\": " << edtInfo->maxLoopDepth << ",\n";
        os << "      \"computeToMemRatio\": " << edtInfo->computeToMemRatio
           << "\n";
        os << "    }";
      }
    }
  });

  os << "\n  ]\n";
  os << "}\n";
}

EdtGraph &EdtAnalysis::getOrCreateEdtGraph(func::FuncOp func) {
  auto it = edtGraphs.find(func);
  if (it != edtGraphs.end())
    return *it->second.get();
  // Build using DbGraph from DbAnalysis for consistency
  DbGraph &db = AM.getDbAnalysis().getOrCreateGraph(func);
  auto eg = std::make_unique<EdtGraph>(func, &db);
  eg->build();
  auto *ptr = eg.get();
  const_cast<EdtAnalysis *>(this)->edtGraphs[func] = std::move(eg);
  return *ptr;
}

bool EdtAnalysis::invalidateGraph(func::FuncOp func) {
  auto it = edtGraphs.find(func);
  if (it != edtGraphs.end()) {
    if (it->second)
      it->second->invalidate();
    edtGraphs.erase(it);
    return true;
  }
  return false;
}

void EdtAnalysis::invalidate() {
  for (auto &kv : edtGraphs)
    if (kv.second)
      kv.second->invalidate();
  edtGraphs.clear();
}
