///==========================================================================
/// File: ConcurrencyGraph.cpp
///
/// This file implements the concurrency graph for EDT analysis.
///==========================================================================

#include "arts/Analysis/Graphs/Concurrency/ConcurrencyGraph.h"
#include "arts/Analysis/ArtsAnalysisManager.h"
#include "arts/Analysis/Concurrency/ConcurrencyAnalysis.h"
#include "arts/Analysis/Edt/EdtAnalysis.h"
#include "arts/Analysis/Graphs/Db/DbGraph.h"
#include "arts/Analysis/Graphs/Db/DbNode.h"
#include "arts/Analysis/Graphs/Edt/EdtGraph.h"
#include "arts/Analysis/Graphs/Edt/EdtNode.h"
#include "arts/Utils/ArtsUtils.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace mlir::arts;

ConcurrencyGraph::ConcurrencyGraph(func::FuncOp func,
                                   ConcurrencyAnalysis *analysis)
    : func(func), analysis(analysis) {
  // EdtGraph will be obtained through the analysis manager during build
}

void ConcurrencyGraph::invalidate() {
  edts.clear();
  edges.clear();
  nodes.clear();
}

void ConcurrencyGraph::build() {
  invalidate();

  // Get EdtGraph through the analysis manager
  if (!analysis) {
    return;
  }

  // Get the ArtsAnalysisManager from the ConcurrencyAnalysis
  ArtsAnalysisManager &AM = analysis->getAM();
  EdtGraph &edtGraphRef = AM.getEdtGraph(func);
  edtGraph = &edtGraphRef;

  if (!edtGraph || !edtGraph->hasDbGraph())
    return;

  /// Collect all edts in a stable order
  SmallVector<EdtOp, 16> edtOps;
  edtGraph->forEachNode([&](NodeBase *node) {
    auto edtNode = static_cast<EdtNode *>(node);
    edtOps.push_back(edtNode->getEdtOp());
  });

  // Store pointers to the EdtOps
  for (auto &edtOp : edtOps) {
    edts.push_back(&edtOp);
  }

  /// Build undirected edges A-B where neither is reachable from the other
  for (size_t i = 0; i < edts.size(); ++i) {
    for (size_t j = i + 1; j < edts.size(); ++j) {
      EdtOp from = *edts[i];
      EdtOp to = *edts[j];
      bool depAB = edtGraph->isEdtReachable(from, to);
      bool depBA = edtGraph->isEdtReachable(to, from);
      if (!depAB && !depBA) {
        ConcurrencyEdge e{};
        e.from = &from;
        e.to = &to;
        if (analysis) {
          // Get EdtAnalysis through the ArtsAnalysisManager
          // ArtsAnalysisManager &AM = analysis->getAM();
          // EdtAnalysis &edtAnalysis = AM.getEdtAnalysis();
          // auto aff = edtAnalysis.affinity(from, to);
          // e.dataOverlap = aff.dataOverlap;
          // e.hazardScore = aff.hazardScore;
          // e.mayConflict = aff.mayConflict;
          // e.reuseProximity = aff.reuseProximity;
          // e.localityScore = aff.localityScore;
          // e.concurrencyRisk = aff.concurrencyRisk;
        }
        edges.push_back(e);
      }
    }
  }
}

// GraphBase interface implementations
NodeBase *ConcurrencyGraph::getEntryNode() const {
  if (!edts.empty() && edtGraph) {
    return edtGraph->getNode(*edts[0]);
  }
  return nullptr;
}

NodeBase *ConcurrencyGraph::getOrCreateNode(Operation *op) {
  if (edtGraph) {
    return edtGraph->getOrCreateNode(op);
  }
  return nullptr;
}

NodeBase *ConcurrencyGraph::getNode(Operation *op) const {
  if (edtGraph) {
    return edtGraph->getNode(op);
  }
  return nullptr;
}

void ConcurrencyGraph::forEachNode(
    const std::function<void(NodeBase *)> &fn) const {
  if (edtGraph) {
    edtGraph->forEachNode(fn);
  }
}

bool ConcurrencyGraph::addEdge(NodeBase *from, NodeBase *to, EdgeBase *edge) {
  // ConcurrencyGraph doesn't support adding edges directly
  // Edges are computed based on concurrency analysis
  return false;
}

// GraphTraits interface implementations
ConcurrencyGraph::NodesIterator ConcurrencyGraph::nodesBegin() {
  return nodes.begin();
}

ConcurrencyGraph::NodesIterator ConcurrencyGraph::nodesEnd() {
  return nodes.end();
}

ConcurrencyGraph::ChildIterator ConcurrencyGraph::childBegin(NodeBase *node) {
  // For concurrency graph, we don't have traditional parent-child relationships
  // Return empty iterator
  return ChildIterator();
}

ConcurrencyGraph::ChildIterator ConcurrencyGraph::childEnd(NodeBase *node) {
  // For concurrency graph, we don't have traditional parent-child relationships
  // Return empty iterator
  return ChildIterator();
}

void ConcurrencyGraph::print(llvm::raw_ostream &os) {
  os << "ConcurrencyGraph: edts=" << edts.size() << ", edges=" << edges.size()
     << "\n";
  for (auto &e : edges) {
    os << "  (" << e.from << ", " << e.to << ")"
       << " overlap=" << e.dataOverlap << " hazard=" << e.hazardScore
       << " reuse=" << e.reuseProximity << " local=" << e.localityScore
       << " risk=" << e.concurrencyRisk
       << " conflict=" << (e.mayConflict ? 1 : 0) << "\n";
  }
}

void ConcurrencyGraph::exportToJson(llvm::raw_ostream &os,
                                    bool includeAnalysis) const {
  using namespace llvm::json;

  Object root;

  Array tasksArr;
  for (auto t : edts) {
    auto *n = edtGraph->getNode(*t);
    tasksArr.push_back(
        Object{{"id", n ? sanitizeString(n->getHierId()) : std::string("")}});
  }
  root["edts"] = std::move(tasksArr);

  Array edgesArr;
  for (auto &e : edges) {
    auto *na = edtGraph->getNode(*e.from);
    auto *nb = edtGraph->getNode(*e.to);
    edgesArr.push_back(
        Object{{"from", na ? sanitizeString(na->getHierId()) : std::string("")},
               {"to", nb ? sanitizeString(nb->getHierId()) : std::string("")},
               {"ovl", e.dataOverlap},
               {"hz", e.hazardScore},
               {"reuse", e.reuseProximity},
               {"local", e.localityScore},
               {"risk", e.concurrencyRisk},
               {"conflict", e.mayConflict},
               {"a", na ? sanitizeString(na->getHierId()) : std::string("")},
               {"b", nb ? sanitizeString(nb->getHierId()) : std::string("")}});
  }
  root["edges"] = std::move(edgesArr);

  /// Include analysis data if requested
  if (includeAnalysis) {
    root["analysis"] = Object{{"numTasks", (double)edts.size()},
                              {"numEdges", (double)edges.size()},
                              {"hasAnalysis", analysis != nullptr}};
  }

  os << llvm::json::Value(std::move(root)) << "\n";
}

void ConcurrencyGraph::createPlacementGroups(ModuleOp module) {
  DisjointSetUnion dsu;

  /// Initialize all EDTs as separate groups
  for (EdtOp *edtPtr : edts) {
    EdtOp edt = *edtPtr;
    dsu.parent[edt] = edt;
  }

  /// Unite EDTs with high locality and no conflicts
  for (const auto &edge : edges) {
    if (!edge.mayConflict && edge.localityScore >= 0.6)
      dsu.unite(*edge.from, *edge.to);
  }

  /// Assign group attributes
  assignGroupAttributes(dsu, module);
}

void ConcurrencyGraph::assignGroupAttributes(const DisjointSetUnion &dsu,
                                             ModuleOp module) {
  llvm::DenseMap<EdtOp, unsigned> repToGroup;
  unsigned nextGroup = 1;

  for (EdtOp *edtPtr : edts) {
    EdtOp edt = *edtPtr;
    EdtOp representative = const_cast<DisjointSetUnion &>(dsu).find(edt);
    unsigned group =
        repToGroup.try_emplace(representative, nextGroup).first->second;

    if (group == nextGroup && representative == edt) {
      nextGroup++;
    }

    edt->setAttr("arts.placement_group",
                 StringAttr::get(module.getContext(),
                                 ("G" + std::to_string(group)).c_str()));
  }
}

void ConcurrencyGraph::computeReuseColoring() {
  if (!analysis) {
    return;
  }

  // Get DbGraph through the ArtsAnalysisManager
  ArtsAnalysisManager &AM = analysis->getAM();
  DbGraph &dbGraphRef = AM.getDbGraph(func);
  DbGraph *dbGraph = &dbGraphRef;
  SmallVector<DbAllocOp, 16> allocs;

  // Collect all allocations from the DB graph
  dbGraph->forEachAllocNode(
      [&](DbAllocNode *node) { allocs.push_back(node->getDbAllocOp()); });

  /// Sort by descending weight: bigger and longer-lived first
  llvm::sort(allocs, [&](DbAllocOp allocA, DbAllocOp allocB) {
    const auto &infoA = dbGraph->getAllocInfo(allocA);
    const auto &infoB = dbGraph->getAllocInfo(allocB);
    unsigned long long weightA =
        std::max<unsigned long long>(
            infoA.staticBytes, (unsigned long long)infoA.totalAccessBytes) *
        (1 + infoA.criticalPath);
    unsigned long long weightB =
        std::max<unsigned long long>(
            infoB.staticBytes, (unsigned long long)infoB.totalAccessBytes) *
        (1 + infoB.criticalPath);
    if (weightA != weightB)
      return weightA > weightB;
    return infoA.allocIndex < infoB.allocIndex;
  });

  auto interferes = [&](DbAllocOp allocX, DbAllocOp allocY) -> bool {
    const auto &infoX = dbGraph->getAllocInfo(allocX);
    const auto &infoY = dbGraph->getAllocInfo(allocY);
    bool lifetimeOverlap = !(infoX.endIndex < infoY.allocIndex ||
                             infoY.endIndex < infoX.allocIndex);
    if (lifetimeOverlap)
      return true;
    // return dbGraph->mayAlias(allocX, allocY);
    return false;
  };

  llvm::SmallDenseMap<DbAllocOp, unsigned> colorMap;
  for (DbAllocOp alloc : allocs) {
    llvm::SmallDenseSet<unsigned> usedColors;
    for (auto &entry : colorMap) {
      DbAllocOp otherAlloc = entry.first;
      if (interferes(alloc, otherAlloc))
        usedColors.insert(entry.second);
    }

    unsigned color = 0;
    while (usedColors.count(color))
      ++color;

    colorMap[alloc] = color;
    // Store the color as an attribute on the allocation operation
    alloc->setAttr(
        "arts.reuse_color",
        IntegerAttr::get(IntegerType::get(alloc.getContext(), 32), color));
  }
}
