///==========================================================================
/// File: DbGraph.cpp
/// Implementation of DbGraph for DB operation analysis.
///==========================================================================
#include "arts/Analysis/Graphs/Db/DbGraph.h"
#include "arts/Analysis/Db/DbAliasAnalysis.h"
#include "arts/Analysis/Db/DbAnalysis.h"
#include "arts/Analysis/Db/DbDataFlowAnalysis.h"
#include "arts/Analysis/Db/DbInfo.h"
#include "arts/Analysis/Graphs/Db/DbEdge.h"
#include "arts/Analysis/Graphs/Db/DbNode.h"
#include "arts/Analysis/LoopAnalysis.h"
#include "arts/Utils/ArtsUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>
using namespace mlir;
using namespace mlir::arts;

#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(db_graph);

std::string DbGraph::generateAllocId(unsigned id) {
  std::string s;
  if (id == 0)
    return "A";
  unsigned n = id;
  while (n > 0) {
    unsigned rem = (n - 1) % 26;
    s = char('A' + rem) + s;
    n = (n - 1) / 26;
  }
  return s;
}

DbGraph::DbGraph(func::FuncOp func, DbAnalysis *analysis)
    : func(func), analysis(analysis) {
  assert(analysis && "Analysis cannot be null");
  ARTS_DEBUG("Creating DB Graph for function: " << func.getName().str());
}

DbGraph::~DbGraph() = default;

void DbGraph::build() {
  if (isBuilt && !needsRebuild)
    return;
  ARTS_DEBUG("Building DB Graph");
  invalidate();
  collectNodes();
  buildDependencies();
  computeOpOrder();
  computeMetrics();
  isBuilt = true;
  needsRebuild = false;
}

void DbGraph::buildNodesOnly() {
  ARTS_DEBUG("Building DB Graph (nodes only)");
  invalidate();
  collectNodes();
  computeOpOrder();
  computeMetrics();
  isBuilt = true;
  needsRebuild = false;
}

void DbGraph::invalidate() {
  allocNodes.clear();
  acquireNodeMap.clear();
  edges.clear();
  nodes.clear();
  childrenCache.clear();
  nextAllocId = 1;
  isBuilt = false;
  needsRebuild = true;
}

NodeBase *DbGraph::getEntryNode() const {
  if (!allocNodes.empty())
    return allocNodes.begin()->second.get();
  return nullptr;
}

NodeBase *DbGraph::getOrCreateNode(Operation *op) {
  if (auto allocOp = dyn_cast<DbAllocOp>(op))
    return getOrCreateAllocNode(allocOp);
  if (auto acquireOp = dyn_cast<DbAcquireOp>(op))
    return getOrCreateAcquireNode(acquireOp);
  return nullptr;
}

NodeBase *DbGraph::getNode(Operation *op) const {
  if (auto allocOp = dyn_cast<DbAllocOp>(op))
    return getDbAllocNode(allocOp);
  if (auto acquireOp = dyn_cast<DbAcquireOp>(op))
    return getDbAcquireNode(acquireOp);
  return nullptr;
}

DbAllocNode *DbGraph::getDbAllocNode(DbAllocOp op) const {
  auto it = allocNodes.find(op);
  return it != allocNodes.end() ? it->second.get() : nullptr;
}

DbAcquireNode *DbGraph::getDbAcquireNode(DbAcquireOp op) const {
  auto it = acquireNodeMap.find(op);
  return it != acquireNodeMap.end() ? it->second : nullptr;
}

DbAllocNode *DbGraph::getOrCreateAllocNode(DbAllocOp op) {
  if (auto node = getDbAllocNode(op))
    return node;

  auto newNode = std::make_unique<DbAllocNode>(op, analysis);
  newNode->setHierId(generateAllocId(nextAllocId++));
  DbAllocNode *ptr = newNode.get();
  allocNodes[op] = std::move(newNode);
  nodes.push_back(ptr);
  return ptr;
}

DbAcquireNode *DbGraph::getOrCreateAcquireNode(DbAcquireOp op) {
  if (auto node = getDbAcquireNode(op))
    return node;

  Operation *underlying = getUnderlyingOperation(op.getPtr());
  if (auto allocOp = dyn_cast_or_null<DbAllocOp>(underlying)) {
    DbAllocNode *allocNode = getOrCreateAllocNode(allocOp);
    DbAcquireNode *acquireNode = allocNode->getOrCreateAcquireNode(op);
    acquireNodeMap[op] = acquireNode;
    /// Create hierarchy edge: alloc -> acquire
    addEdge(allocNode, acquireNode, new DbChildEdge(allocNode, acquireNode));
    nodes.push_back(acquireNode);
    return acquireNode;
  }

  if (auto parentAcquireOp = dyn_cast_or_null<DbAcquireOp>(underlying)) {
    DbAcquireNode *parentAcquire = getOrCreateAcquireNode(parentAcquireOp);
    /// Parent acquire must exist now; its root alloc is tracked inside the node
    DbAcquireNode *acquireNode = parentAcquire->getOrCreateAcquireNode(op);
    acquireNodeMap[op] = acquireNode;
    /// Create hierarchy edge: parent acquire -> child acquire
    addEdge(parentAcquire, acquireNode,
            new DbChildEdge(parentAcquire, acquireNode));
    nodes.push_back(acquireNode);
    return acquireNode;
  }

  assert(false &&
         "Expected DbAllocOp or DbAcquireOp as underlying for acquire");
  return nullptr;
}

void DbGraph::forEachNode(const std::function<void(NodeBase *)> &fn) const {
  for (auto *node : nodes)
    fn(node);
}

void DbGraph::forEachAllocNode(
    const std::function<void(DbAllocNode *)> &fn) const {
  for (const auto &pair : allocNodes)
    fn(pair.second.get());
}

void DbGraph::forEachAcquireNode(
    const std::function<void(DbAcquireNode *)> &fn) const {
  for (const auto &pair : acquireNodeMap)
    fn(pair.second);
}

bool DbGraph::addEdge(NodeBase *from, NodeBase *to, EdgeBase *edge) {
  auto key = std::make_pair(from, to);
  if (edges.count(key))
    return false;
  edges[key] = std::unique_ptr<EdgeBase>(edge);
  from->addOutEdge(edge);
  to->addInEdge(edge);
  return true;
}

const DbAllocInfo &DbGraph::getAllocInfo(DbAllocOp alloc) const {
  auto it = allocNodes.find(alloc);
  assert(it != allocNodes.end() && "Allocation not found in graph");
  return it->second->getInfo();
}

void DbGraph::print(llvm::raw_ostream &os) {
  if (allocNodes.empty())
    return;

  os << "\n";
  os << "DbGraph Analysis: " << this->func.getName().str() << "\n";
  os << "======================================\n";

  os << "Summary: " << allocNodes.size() << " allocs, " << acquireNodeMap.size()
     << " acquires\n";
  os << "Peak: " << peakLiveDbs << " live DBs, " << peakBytes << " bytes\n";
  os << "\n";

  for (const auto &pair : allocNodes) {
    DbAllocNode *alloc = pair.second.get();
    const DbAllocInfo &info = alloc->getInfo();

    os << "DB [" << alloc->getHierId() << "]\n";
    os << "  Op: " << alloc->getDbAllocOp() << "\n";
    os << "  Lifetime: Op " << info.allocIndex << " -> " << info.endIndex
       << " (span=" << info.criticalSpan << ", path=" << info.criticalPath
       << ")\n";
    os << "  Rank: " << info.getRank() << "\n";
    os << "  Size: " << info.staticBytes << " bytes";
    if (info.staticBytes == 0)
      os << " (dynamic)";
    os << "\n";
    os << "  Usage: " << info.numAcquires << " acquires, " << info.numReleases
       << " releases, depth=" << info.maxLoopDepth << "\n";

    /// Show acquires
    if (!info.acquireNodes.empty()) {
      os << "  Acquires (" << info.acquireNodes.size() << "):\n";
      for (auto *acqNode : info.acquireNodes) {
        auto &acqInfo = acqNode->getInfo();
        os << "    [" << acqNode->getHierId() << "] Op " << acqInfo.beginIndex;
        if (acqInfo.endIndex != acqInfo.beginIndex)
          os << "->" << acqInfo.endIndex;

        /// Show mode
        os << ", mode=" << acqNode->getDbAcquireOp().getModeAttr();

        /// Show sizes if available
        if (!acqInfo.constSizes.empty()) {
          os << ", size=[";
          for (size_t i = 0; i < acqInfo.constSizes.size(); ++i) {
            if (i > 0)
              os << ",";
            if (acqInfo.constSizes[i] == INT64_MIN)
              os << "?";
            else
              os << acqInfo.constSizes[i];
          }
          os << "]";
        }
        os << "\n";

        /// Show offset ranges if available
        if (!acqInfo.constOffsets.empty()) {
          os << "      Offsets: [";
          for (size_t i = 0; i < acqInfo.constOffsets.size(); ++i) {
            if (i > 0)
              os << ",";
            if (acqInfo.constOffsets[i] == INT64_MIN)
              os << "?";
            else
              os << acqInfo.constOffsets[i];
          }
          os << "]\n";
        }

        /// Show number of reads and writes
        auto loads = acqNode->getLoads().size();
        auto stores = acqNode->getStores().size();
        os << "      EDT Uses - Reads: " << loads << ", Writes: " << stores
           << "\n";

        /// Show outgoing edges for this acquire
        const auto &outEdges = acqNode->getOutEdges();
        if (!outEdges.empty()) {
          os << "      Outgoing edges:\n";
          for (auto *edge : outEdges)
            os << "        -> " << edge->getTo()->getHierId() << " ["
               << edge->getType() << "]\n";
        }
      }
    }

    /// Show releases tracked in acquire nodes
    unsigned releaseCount = 0;
    alloc->forEachChildNode([&](NodeBase *child) {
      if (auto *acq = dyn_cast<DbAcquireNode>(child)) {
        if (acq->getDbReleaseOp())
          releaseCount++;
      }
    });

    if (releaseCount > 0)
      os << "  Releases: " << releaseCount << " (tracked in acquires)\n";

    /// Show all edges for this allocation
    unsigned totalEdges = 0;
    alloc->forEachChildNode(
        [&](NodeBase *child) { totalEdges += child->getOutEdges().size(); });
    if (totalEdges > 0)
      os << "  Total edges: " << totalEdges;

    os << "\n";
  }

  // /// Show edge summary
  // os << "Edge Summary:\n";
  // os << "======================================\n";
  // unsigned childEdges = 0;

  // forEachNode([&](NodeBase *node) {
  //   for (auto *edge : node->getOutEdges()) {
  //     if (edge->getKind() == EdgeBase::EdgeKind::Child)
  //       childEdges++;
  //   }
  // });

  // os << "  Child edges: " << childEdges << "\n";
  os << "\n";
}

/// Walk through the function and collect all DB operations
/// (alloc/acquire/release) into graph nodes
void DbGraph::collectNodes() {
  ARTS_DEBUG("Phase 1 - Collecting nodes");

  func.walk([&](Operation *op) {
    if (isa<DbAllocOp, DbAcquireOp>(op))
      getOrCreateNode(op);
  });

  ARTS_DEBUG("Collected " << allocNodes.size() << " allocations, and "
                          << acquireNodeMap.size() << " acquires");
}

/// Build dependency edges using environment-based dataflow analysis.
/// Delegates to DbDataFlowAnalysis to track reaching definitions through
/// control flow and create DDG edges.
void DbGraph::buildDependencies() {
  ARTS_DEBUG("Phase 2 - Building dependencies (DDG)");
  auto *dataflowAnalysis = analysis->getDataFlowAnalysis();
  assert(dataflowAnalysis && "Dataflow analysis required for dependencies");

  dataflowAnalysis->initialize(this, analysis);
  dataflowAnalysis->runAnalysis(func);

  ARTS_DEBUG("Phase 2 - Dependency analysis finished");
}

/// Compute operation order based on the order of operations in the function.
void DbGraph::computeOpOrder() {
  unsigned idx = 0;
  func.walk([&](Operation *op) { opOrder[op] = idx++; });
}

/// Compute per-allocation metrics and global summary metrics.
/// This includes timing intervals, critical spans/paths, loop depths,
/// and peak resource usage across all allocations.
void DbGraph::computeMetrics() {
  peakLiveDbs = 0;
  peakBytes = 0;

  /// Process each allocation to compute per-node metrics
  for (const auto &pair : allocNodes)
    computeAllocMetrics(pair.first, pair.second.get());

  /// Compute global peak metrics using sweep-line algorithm
  computePeakMetrics();

  /// Identify allocations that could reuse memory
  computeReuseCandidates();
}

/// Compute metrics for a single allocation.
/// This processes all acquire/release child nodes and derives timing intervals,
/// critical spans, paths, loop depths, and lifetime classification.
void DbGraph::computeAllocMetrics(DbAllocOp alloc, DbAllocNode *allocNode) {
  DbAllocInfo &info = allocNode->getInfo();

  /// Initialize allocation-level counters
  info.isAlloc = true;
  info.allocIndex = getOrder(alloc.getOperation());
  info.numAcquires = 0;
  info.numReleases = 0;
  info.inCount = 0;
  info.outCount = 0;
  info.totalAccessBytes = 0;
  info.acquireNodes.clear();

  /// Process acquire and release nodes to compute timing and access info
  std::function<void(DbAcquireNode *)> processAcqRec;
  processAcqRec = [&](DbAcquireNode *acq) {
    processAcquireNode(acq, info);
    acq->forEachChildNode([&](NodeBase *child) {
      if (auto *nested = dyn_cast<DbAcquireNode>(child))
        processAcqRec(nested);
    });
  };
  allocNode->forEachChildNode([&](NodeBase *child) {
    if (auto *acq = dyn_cast<DbAcquireNode>(child))
      processAcqRec(acq);
  });

  /// Sort acquire nodes by program order for interval processing
  llvm::sort(info.acquireNodes,
             [](const DbAcquireNode *a, const DbAcquireNode *b) {
               return a->getInfo().beginIndex < b->getInfo().beginIndex;
             });

  /// Compute the allocation's end index as the latest release
  unsigned lastEnd = info.allocIndex;
  for (auto *node : info.acquireNodes) {
    const auto &nodeInfo = node->getInfo();
    lastEnd = std::max(lastEnd, nodeInfo.endIndex);
  }
  info.endIndex = lastEnd;

  /// Compute derived lifetime and locality metrics
  computeCriticalSpan(info);
  computeCriticalPath(info);
  computeLoopDepth(info);
  computeLongLivedFlag(info);
}

/// Process an acquire node: compute timing interval, link to release,
/// and aggregate access statistics into the parent allocation.
void DbGraph::processAcquireNode(DbAcquireNode *acq, DbAllocInfo &info) {
  /// Compute the lifetime interval [beginIndex, endIndex] in program order
  auto acquireOp = cast<DbAcquireOp>(acq->getOp());
  unsigned beginIndex = getOrder(acquireOp.getOperation());
  unsigned endIndex = beginIndex;

  /// Use the precomputed release from the acquire node
  DbReleaseOp releaseOp = acq->getDbReleaseOp();
  if (releaseOp) {
    endIndex = getOrder(releaseOp.getOperation());
    /// Store acquire/release pair for downstream passes
    acq->getInfo().acquire = acquireOp;
    acq->getInfo().release = releaseOp;
    ++info.numReleases;
  }

  /// Update timing info in the acquire node
  auto &nodeInfo = acq->getInfo();
  nodeInfo.beginIndex = beginIndex;
  nodeInfo.endIndex = endIndex;

  /// Add to allocation's list of acquires
  info.acquireNodes.push_back(acq);

  /// Aggregate access statistics
  ++info.numAcquires;
  info.inCount += nodeInfo.inCount;
  info.outCount += nodeInfo.outCount;
  info.totalAccessBytes += (unsigned long long)nodeInfo.estimatedBytes;
}

/// Compute critical span for an allocation.
/// It measures the bounding lifetime window for the
/// allocation across all of its acquires. This is the inclusive distance from
/// the earliest acquire beginIndex to the latest acquire endIndex.
void DbGraph::computeCriticalSpan(DbAllocInfo &info) {
  info.criticalSpan = 0;
  if (info.acquireNodes.empty())
    return;

  unsigned minBegin = info.acquireNodes.front()->getInfo().beginIndex;
  unsigned maxEnd = info.acquireNodes.front()->getInfo().endIndex;

  for (auto *node : info.acquireNodes) {
    const auto &nodeInfo = node->getInfo();
    minBegin = std::min(minBegin, nodeInfo.beginIndex);
    maxEnd = std::max(maxEnd, nodeInfo.endIndex);
  }

  info.criticalSpan = maxEnd >= minBegin ? (maxEnd - minBegin + 1) : 0;
}

/// Compute critical path length as the union of acquire intervals.
/// It measures the total active-use time by merging overlapping inclusive
/// intervals [beginIndex, endIndex] and summing their lengths.
void DbGraph::computeCriticalPath(DbAllocInfo &info) {
  SmallVector<std::pair<unsigned, unsigned>, 8> segments;
  for (auto *node : info.acquireNodes) {
    const auto &nodeInfo = node->getInfo();
    segments.emplace_back(nodeInfo.beginIndex, nodeInfo.endIndex);
  }

  llvm::sort(segments, [](auto a, auto b) {
    return a.first < b.first || (a.first == b.first && a.second < b.second);
  });

  unsigned currentStart = 0, currentEnd = 0;
  bool initialized = false;
  info.criticalPath = 0;

  for (auto [start, end] : segments) {
    if (!initialized) {
      currentStart = start;
      currentEnd = end;
      initialized = true;
      continue;
    }

    if (start <= currentEnd) {
      currentEnd = std::max(currentEnd, end);
    } else {
      info.criticalPath += (currentEnd - currentStart + 1);
      currentStart = start;
      currentEnd = end;
    }
  }

  if (initialized)
    info.criticalPath += (currentEnd - currentStart + 1);
}

/// Compute maximum loop depth for an allocation.
/// It captures the deepest loop nesting among all acquires of this
/// allocation as a proxy for locality/pressure heuristics.
void DbGraph::computeLoopDepth(DbAllocInfo &info) {
  auto *loopAnalysis = analysis->getLoopAnalysis();
  assert(loopAnalysis && "Loop analysis required for loop depth");
  unsigned maxDepth = 0;
  for (auto *node : info.acquireNodes) {
    SmallVector<LoopInfo *, 4> enclosingLoops;
    loopAnalysis->collectEnclosingLoops(node->getOp(), enclosingLoops);
    maxDepth = std::max<unsigned>(maxDepth, enclosingLoops.size());
  }
  info.maxLoopDepth = maxDepth;
}

/// Classify whether the allocation is long-lived.
/// It marks allocations that span most of the function's op-order so they
/// can be handled differently by later heuristics (e.g., reuse/coloring).
void DbGraph::computeLongLivedFlag(DbAllocInfo &info) {
  unsigned firstIdx = info.allocIndex;
  unsigned lastIdx = opOrder.empty() ? 0u : (unsigned)opOrder.size() - 1;
  if (lastIdx > 0) {
    double coverage =
        (double)(info.endIndex - firstIdx + 1) / (double)(lastIdx + 1);
    /// If coverage > 0.7, set `isLongLived`.
    info.isLongLived = coverage > 0.7;
  }
}

/// Determine if allocation may be escaping
void DbGraph::computeEscapingFlag(DbAllocOp alloc, DbAllocInfo &info) {
  info.maybeEscaping = false;
  for (Operation *user : alloc.getPtr().getUsers()) {
    if (!isa<DbAcquireOp>(user) && !isa<DbReleaseOp>(user)) {
      info.maybeEscaping = true;
      break;
    }
  }
}

/// Compute peak live databases and bytes using sweep-line algorithm
void DbGraph::computePeakMetrics() {
  struct Event {
    unsigned idx;
    int delta;
    unsigned long long bytes;
  };

  SmallVector<Event, 32> events;
  for (auto &kv : allocNodes) {
    const DbAllocInfo &info = kv.second->getInfo();
    for (auto *node : info.acquireNodes) {
      const auto &nodeInfo = node->getInfo();
      events.push_back({nodeInfo.beginIndex, +1, info.staticBytes});
      events.push_back({nodeInfo.endIndex + 1, -1, info.staticBytes});
    }
  }

  llvm::sort(events,
             [](const Event &a, const Event &b) { return a.idx < b.idx; });

  int liveCount = 0;
  unsigned long long liveBytes = 0;
  for (auto &event : events) {
    liveCount += event.delta;
    if (event.delta > 0)
      liveBytes += event.bytes;
    else
      liveBytes -= event.bytes;
    peakLiveDbs = std::max<uint64_t>(peakLiveDbs, (uint64_t)liveCount);
    peakBytes = std::max<unsigned long long>(peakBytes, liveBytes);
  }
}

/// Compute reuse candidates for allocations
void DbGraph::computeReuseCandidates() {
  SmallVector<DbAllocOp, 16> allocs;
  for (auto &kv : allocNodes) {
    allocs.push_back(kv.first);
  }

  llvm::sort(allocs, [&](DbAllocOp a, DbAllocOp b) {
    return getOrder(a.getOperation()) < getOrder(b.getOperation());
  });

  for (size_t i = 0; i < allocs.size(); ++i) {
    for (size_t j = i + 1; j < allocs.size(); ++j) {
      const auto &infoA = allocNodes[allocs[i]]->getInfo();
      const auto &infoB = allocNodes[allocs[j]]->getInfo();

      /// Check if lifetimes don't overlap and sizes match
      bool noOverlap = (infoA.endIndex <= infoB.allocIndex ||
                        infoB.endIndex <= infoA.allocIndex);
      bool sameSize =
          (infoA.staticBytes != 0 && infoA.staticBytes == infoB.staticBytes);

      if (noOverlap && sameSize) {
        allocNodes[allocs[i]]->getInfo().reuseMatches.push_back(allocs[j]);
        allocNodes[allocs[j]]->getInfo().reuseMatches.push_back(allocs[i]);
      }
    }
  }
}

void DbGraph::exportToJson(llvm::raw_ostream &os, bool includeAnalysis) const {
  using namespace llvm::json;

  auto edgeKindStr = [](EdgeBase::EdgeKind k) -> const char * {
    switch (k) {
    case EdgeBase::EdgeKind::Child:
      return "Child";
    case EdgeBase::EdgeKind::Dep:
      return "Dep";
    }
    return "Unknown";
  };

  Object root;

  Array nodesArr;
  forEachNode([&](NodeBase *node) {
    const char *nkind = "DbNode";
    if (isa<DbAllocNode>(node))
      nkind = "DbAlloc";
    else if (isa<DbAcquireNode>(node))
      nkind = "DbAcquire";
    nodesArr.push_back(Object{{"id", sanitizeString(node->getHierId())},
                              {"group", "db"},
                              {"nodeKind", nkind}});
  });

  Array edgesArr;
  forEachNode([&](NodeBase *from) {
    for (auto *edge : from->getOutEdges()) {
      edgesArr.push_back(
          Object{{"from", sanitizeString(edge->getFrom()->getHierId())},
                 {"to", sanitizeString(edge->getTo()->getHierId())},
                 {"edgeKind", edgeKindStr(edge->getKind())},
                 {"label", edge->getType().str()}});
    }
  });

  root["nodes"] = std::move(nodesArr);
  root["edges"] = std::move(edgesArr);

  /// Include analysis data if requested
  if (includeAnalysis) {
    auto modeToString = [](ArtsMode m) -> std::string {
      switch (m) {
      case ArtsMode::in:
        return "in";
      case ArtsMode::out:
        return "out";
      default:
        return "inout";
      }
    };

    /// Summary info
    root["info"] = Object{
        {"peakLiveDbs", (uint64_t)peakLiveDbs},
        {"peakBytes", std::to_string(peakBytes)},
    };

    /// Allocs
    Array allocs;
    for (const auto &kv : allocNodes) {
      DbAllocOp alloc = kv.first;
      const DbAllocInfo &M = kv.second->getInfo();
      Object A;
      A["id"] = sanitizeString(getNode(alloc.getOperation())->getHierId());
      A["allocIndex"] = (uint64_t)M.allocIndex;
      A["endIndex"] = (uint64_t)M.endIndex;
      A["numAcquires"] = (uint64_t)M.numAcquires;
      A["numReleases"] = (uint64_t)M.numReleases;
      A["staticBytes"] = std::to_string(M.staticBytes);
      A["criticalSpan"] = (uint64_t)M.criticalSpan;
      A["criticalPath"] = (uint64_t)M.criticalPath;
      A["totalAccessBytes"] = std::to_string(M.totalAccessBytes);
      A["isLongLived"] = (bool)M.isLongLived;
      A["maybeEscaping"] = (bool)M.maybeEscaping;

      Array intervals;
      for (const auto *node : M.acquireNodes) {
        const auto &nodeInfo = node->getInfo();
        Object I;
        I["begin"] = (uint64_t)nodeInfo.beginIndex;
        I["end"] = (uint64_t)nodeInfo.endIndex;
        I["mode"] = modeToString(cast<DbAcquireOp>(node->getOp()).getMode());
        Array offs;
        for (auto v : nodeInfo.constOffsets)
          offs.push_back((int64_t)v);
        I["constOffsets"] = std::move(offs);
        Array sizes;
        for (auto v : nodeInfo.constSizes)
          sizes.push_back((int64_t)v);
        I["constSizes"] = std::move(sizes);
        I["estimatedBytes"] = std::to_string(nodeInfo.estimatedBytes);
        intervals.push_back(std::move(I));
      }
      A["intervals"] = std::move(intervals);
      allocs.push_back(std::move(A));
    }
    root["allocs"] = std::move(allocs);

    /// Reuse candidates
    // Array reuse;
    // for (const auto &kv : allocNodes) {
    //   DbAllocOp Aop = kv.first;
    //   const auto &AI = kv.second->getInfo();
    //   for (DbAllocOp Bop : AI.reuseMatches) {
    //     if (getOrder(Aop.getOperation()) < getOrder(Bop.getOperation())) {
    //       Object P;
    //       P["a"] = sanitizeString(getNode(Aop.getOperation())->getHierId());
    //       P["b"] = sanitizeString(getNode(Bop.getOperation())->getHierId());
    //       reuse.push_back(std::move(P));
    //     }
    //   }
    // }
    // root["reuseCandidates"] = std::move(reuse);
  }

  os << llvm::json::Value(std::move(root)) << "\n";
}

/// Helper to populate and sort children cache for a node
void DbGraph::populateChildrenCache(NodeBase *node) {
  auto &vec = childrenCache[node];
  vec.clear();

  /// Collect children from the node's outgoing edges
  for (auto *edge : node->getOutEdges())
    vec.push_back(edge->getTo());

  /// Sort by operation order
  llvm::sort(vec, [&](NodeBase *a, NodeBase *b) {
    Operation *opA = a ? a->getOp() : nullptr;
    Operation *opB = b ? b->getOp() : nullptr;
    return getOrder(opA) < getOrder(opB);
  });
}

GraphBase::ChildIterator DbGraph::childBegin(NodeBase *node) {
  populateChildrenCache(node);
  return childrenCache[node].begin();
}

GraphBase::ChildIterator DbGraph::childEnd(NodeBase *node) {
  auto it = childrenCache.find(node);
  if (it == childrenCache.end()) {
    populateChildrenCache(node);
    return childrenCache[node].end();
  }
  return it->second.end();
}
