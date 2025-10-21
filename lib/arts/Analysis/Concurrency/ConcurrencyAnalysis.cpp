///==========================================================================
/// File: ConcurrencyAnalysis.cpp
///
/// This file implements the central analysis for concurrency operations.
///==========================================================================

#include "arts/Analysis/Concurrency/ConcurrencyAnalysis.h"
#include "arts/Analysis/ArtsAnalysisManager.h"
#include "arts/Analysis/Edt/EdtAnalysis.h"
#include "arts/Analysis/Graphs/Concurrency/ConcurrencyGraph.h"
#include "arts/Analysis/Graphs/Edt/EdtGraph.h"

using namespace mlir;
using namespace mlir::arts;

ConcurrencyAnalysis::ConcurrencyAnalysis(ArtsAnalysisManager &AM) : AM(AM) {
  // ARTS_DEBUG("Initializing ConcurrencyAnalysis");
}

ConcurrencyAnalysis::~ConcurrencyAnalysis() = default;

ConcurrencyGraph &
ConcurrencyAnalysis::getOrCreateConcurrencyGraph(func::FuncOp func) {
  auto it = concurrencyGraphs.find(func);
  if (it != concurrencyGraphs.end())
    return *it->second.get();
  /// Build ConcurrencyGraph
  auto cg = std::make_unique<ConcurrencyGraph>(func, this);
  cg->build();
  auto *ptr = cg.get();
  concurrencyGraphs[func] = std::move(cg);
  return *ptr;
}

bool ConcurrencyAnalysis::invalidateConcurrencyGraph(func::FuncOp func) {
  auto it = concurrencyGraphs.find(func);
  if (it != concurrencyGraphs.end()) {
    if (it->second)
      it->second->invalidate();
    concurrencyGraphs.erase(it);
    return true;
  }
  return false;
}

void ConcurrencyAnalysis::invalidate() {
  for (auto &kv : concurrencyGraphs)
    if (kv.second)
      kv.second->invalidate();
  concurrencyGraphs.clear();
}

void ConcurrencyAnalysis::print(func::FuncOp func, llvm::raw_ostream &os) {
  ConcurrencyGraph &cg = getOrCreateConcurrencyGraph(func);
  cg.print(os);
}
