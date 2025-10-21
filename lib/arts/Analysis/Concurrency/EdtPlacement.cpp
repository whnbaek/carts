///==========================================================================
/// File: EdtPlacement.cpp
/// Defines greedy EDT placement scorer for optimal task scheduling.
///==========================================================================

#include "arts/Analysis/Concurrency/EdtPlacement.h"

using namespace mlir;
using namespace mlir::arts;

SmallVector<std::string, 8>
EdtPlacementHeuristics::makeNodeNames(unsigned count) {
  SmallVector<std::string, 8> out;
  out.reserve(count);
  for (unsigned i = 0; i < count; ++i)
    out.emplace_back("N" + std::to_string(i));
  return out;
}

SmallVector<PlacementDecision, 16>
EdtPlacementHeuristics::compute(const SmallVector<std::string, 8> &nodes) {
  SmallVector<PlacementDecision, 16> decisions;
  if (!edtGraph || !edtAnalysis)
    return decisions;

  /// Naive load tracking per node
  llvm::StringMap<double> nodeLoad;
  for (auto &n : nodes)
    nodeLoad[n] = 0.0;

  /// Task list
  SmallVector<EdtOp, 16> edts;
  edtGraph->forEachNode([&](NodeBase *node) {
    if (auto *t = dyn_cast<EdtNode>(node))
      edts.push_back(t->getEdtOp());
  });

  /// Track edts already placed per node to evaluate concurrency-aware
  /// affinity and risk when scoring edts.
  llvm::StringMap<SmallVector<EdtOp, 8>> nodeedtsPlaced;

  /// Greedy scoring
  for (EdtOp t : edts) {
    auto edtNode = edtGraph->getEdtNode(t);
    const EdtInfo *sum = edtNode ? &edtNode->getInfo() : nullptr;
    double cmr = sum ? sum->computeToMemRatio : 1.0;
    double baseCost = sum ? (double)sum->totalOps : 1.0;

    PlacementDecision d;
    d.edtOp = t;
    for (auto &n : nodes) {
      /// Without a full fabric model, synthesize components from available
      /// analysis
      double locality = 0.0;
      double affinityBoost = 0.0;
      double riskSum = 0.0;
      /// Weighted locality: use per-alloc access bytes to prioritize the node
      /// where the task's most-used DBs (by bytes) are pseudo-resident.
      if (sum && !sum->dbAllocAccessBytes.empty()) {
        double totalBytes = 0.0;
        double localBytes = 0.0;
        for (auto &kv : sum->dbAllocAccessBytes) {
          Operation *allocOp = kv.first;
          uint64_t bytes = kv.second;
          totalBytes += (double)bytes;
          std::string pseudo =
              "N" + std::to_string((reinterpret_cast<uintptr_t>(allocOp) >> 4) %
                                   std::max<size_t>(nodes.size(), 1));
          if (pseudo == n)
            localBytes += (double)bytes;
        }
        locality = totalBytes > 0.0 ? (localBytes / totalBytes) : 0.0;
      } else if (sum && (!sum->dbAllocsRead.empty() ||
                         !sum->dbAllocsWritten.empty())) {
        size_t hits = 0,
               total = sum->dbAllocsRead.size() + sum->dbAllocsWritten.size();
        for (Operation *allocOp : sum->dbAllocsRead) {
          std::string pseudo =
              "N" + std::to_string((reinterpret_cast<uintptr_t>(allocOp) >> 4) %
                                   std::max<size_t>(nodes.size(), 1));
          hits += (pseudo == n);
        }
        for (Operation *allocOp : sum->dbAllocsWritten) {
          std::string pseudo =
              "N" + std::to_string((reinterpret_cast<uintptr_t>(allocOp) >> 4) %
                                   std::max<size_t>(nodes.size(), 1));
          hits += (pseudo == n);
        }
        locality = total ? (double)hits / (double)total : 0.0;
      }

      /// Concurrency graph inspired adjustment: if this task shares data with
      /// edts already chosen for node n and can run concurrently with them,
      /// prefer co-location when locality is high and hazards are low; penalize
      /// otherwise.
      // if (edtGraph) {
      //   for (Operation *other : nodeedtsPlaced[n]) {
      //     auto a = static_cast<EdtOp>(t);
      //     auto b = static_cast<EdtOp>(other);
      //     bool depAB = edtGraph->isEdtReachable(a, b);
      //     bool depBA = edtGraph->isEdtReachable(b, a);
      //     if (!depAB && !depBA) {
      //       auto aff = edtAnalysis->affinity(a, b);
      //       affinityBoost += aff.localityScore; /// reward shared read
      //       locality riskSum += aff.concurrencyRisk;     /// penalize write
      //       hazards
      //     } else {
      //       /// Sequential reuse: encourage co-location slightly
      //       auto aff = edtAnalysis->affinity(a, b);
      //       affinityBoost += 0.5 * aff.localityScore;
      //     }
      //   }
      // }
      double risk = 0.0; /// aggregate with neighbors if needed
      double load = nodeLoad[n];
      double contention = load; /// proxy

      double score = 0.5 * locality + 0.2 * (1.0 / (1.0 + contention)) +
                     0.2 * (1.0 / (1.0 + load));
      /// Concurrency-aware adjustments
      score += 0.3 * affinityBoost;
      score -= 0.25 * riskSum;
      /// Lightly bias compute-heavy edts to spread out
      score -= 0.1 * cmr;

      d.candidates.push_back({n, score, locality, contention, load, risk});
    }
    std::sort(d.candidates.begin(), d.candidates.end(),
              [](const PlacementNodeScore &a, const PlacementNodeScore &b) {
                return a.score > b.score;
              });
    d.chosenNode = d.candidates.front().node;
    nodeLoad[d.chosenNode] += baseCost;
    nodeedtsPlaced[d.chosenNode].push_back(t);
    decisions.push_back(std::move(d));
  }
  return decisions;
}

void EdtPlacementHeuristics::exportToJson(
    func::FuncOp func, const SmallVector<PlacementDecision, 16> &decisions,
    llvm::raw_ostream &os) const {
  os << "{\n  \"function\": \"" << func.getName()
     << "\",\n  \"placement\": [\n";
  bool first = true;
  for (auto &d : decisions) {
    if (!first)
      os << ",\n";
    first = false;
    os << "    {\"op\": \"" << d.edtOp << "\", \"chosen\": \"" << d.chosenNode
       << "\", \"candidates\": [";
    bool fc = true;
    for (auto &c : d.candidates) {
      if (!fc)
        os << ",";
      fc = false;
      os << "{\"node\": \"" << c.node << "\", \"score\": " << c.score
         << ", \"locality\": " << c.locality
         << ", \"contention\": " << c.contention << ", \"load\": " << c.load
         << ", \"risk\": " << c.risk << "}";
    }
    os << "]}";
  }
  os << "\n  ]\n}\n";
}
