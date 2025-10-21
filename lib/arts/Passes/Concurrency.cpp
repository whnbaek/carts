///==========================================================================
/// File: Concurrency.cpp
/// Defines concurrency-driven optimizations using ARTS analysis managers.
/// The Concurrency pass make high-level decisions about task (EDT) placement
/// and data (DB) affinity - it advices on where to execute EDTs or
/// nodes—essentially, deciding locality (e.g., which worker thread, node, or
/// memory tier) based on access patterns, dependencies, and the abstract
/// machine model.
///==========================================================================

#include "ArtsPassDetails.h"
#include "arts/Analysis/ArtsAnalysisManager.h"
#include "arts/Analysis/Concurrency/DbPlacement.h"
#include "arts/Analysis/Concurrency/EdtPlacement.h"
#include "arts/Analysis/Graphs/Concurrency/ConcurrencyGraph.h"
#include "arts/ArtsDialect.h"
#include "arts/Codegen/ArtsCodegen.h"
#include "arts/Passes/ArtsPasses.h"
#include "arts/Utils/ArtsAbstractMachine.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(concurrency)

using namespace mlir;
using namespace mlir::arts;

///==========================================================================
/// ConcurrencyPass
///==========================================================================
struct ConcurrencyPass : public ConcurrencyBase<ConcurrencyPass> {
  explicit ConcurrencyPass(ArtsAnalysisManager *AM) : AM(AM) {
    assert(AM && "ArtsAnalysisManager must be provided externally");
  }
  void runOnOperation() override;

private:
  /// Apply EDT and DB placement decisions
  void applyPlacementDecisions(func::FuncOp func);

  /// Apply EDT placement decisions to operations
  void applyEdtPlacement(const SmallVector<PlacementDecision, 16> &decisions,
                         const SmallVector<std::string, 8> &nodes);

  /// Apply DB placement decisions to operations
  void applyDbPlacement(const SmallVector<DbPlacementDecision, 16> &decisions,
                        const SmallVector<std::string, 8> &nodes);

  /// Helper to apply placement decisions for any operation type
  template <typename DecisionsT, typename GetOpFunc, typename SetAttrFunc>
  void applyPlacementDecisions(const SmallVector<std::string, 8> &nodes,
                               const DecisionsT &decisions, GetOpFunc getOpFunc,
                               SetAttrFunc setAttrFunc) {
    llvm::StringMap<unsigned> nodeIndex;
    for (unsigned i = 0; i < nodes.size(); ++i) {
      nodeIndex[nodes[i]] = i;
    }

    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    for (const auto &decision : decisions) {
      auto it = nodeIndex.find(decision.chosenNode);
      unsigned idx = (it != nodeIndex.end()) ? it->second : 0u;
      if (Operation *op = getOpFunc(decision)) {
        setAttrFunc(op, idx, builder);
      }
    }
  }

  /// Helper to emit debug output with string stream
  template <typename OutputFunc>
  void emitDebugString(const std::string &sectionName,
                       OutputFunc &&outputFunc) {
    ARTS_DEBUG_SECTION(sectionName, {
      std::string s;
      llvm::raw_string_ostream os(s);
      outputFunc(os);
      ARTS_DBGS() << s;
    });
  }

  /// Emit debug output for placement decisions
  void emitDebugOutput(func::FuncOp func,
                       const EdtPlacementHeuristics &edtPlacer,
                       const SmallVector<PlacementDecision, 16> &edtDecisions,
                       const DbPlacementHeuristics &dbPlacer,
                       const SmallVector<DbPlacementDecision, 16> &dbDecisions);

  /// Apply parallelism strategy to EDT operations
  void applyEdtParallelismStrategy(EdtOp edtOp);

  /// Adjust datablock modes based on EDT concurrency
  void adjustDbModes();

  /// Update runtime query operations based on EDT concurrency
  void updateRuntimeQueryOperations();

  ArtsAbstractMachine *abstractMachine = nullptr;
  ArtsAnalysisManager *AM = nullptr;
  ModuleOp module;
};

void ConcurrencyPass::runOnOperation() {
  module = getOperation();

  /// Get abstractMachine from ArtsAnalysisManager
  abstractMachine = &AM->getAbstractMachine();

  ARTS_INFO_HEADER(Concurrency);
  ARTS_DEBUG_REGION(module.dump(););
  ARTS_INFO("Machine valid=" << (abstractMachine->isValid() ? 1 : 0)
                             << ", nodes=" << abstractMachine->getNodeCount()
                             << ", threads=" << abstractMachine->getThreads());

  /// Apply EDT parallelism strategy to all EDT operations
  module.walk([&](EdtOp edtOp) { applyEdtParallelismStrategy(edtOp); });

  /// Adjust datablock modes based on EDT concurrency
  adjustDbModes();

  /// Update runtime query operations based on EDT concurrency
  updateRuntimeQueryOperations();

  /// Apply EDT and DB placement decisions to all functions
  // module.walk([&](func::FuncOp func) {
  //   applyPlacementDecisions(func);
  // });

  ARTS_INFO_FOOTER(Concurrency);
  ARTS_DEBUG_REGION(module.dump(););
}

void ConcurrencyPass::applyPlacementDecisions(func::FuncOp func) {
  auto nodes = EdtPlacementHeuristics::makeNodeNames(4);
  auto &dbGraph = AM->getDbGraph(func);
  auto &edtGraph = AM->getEdtGraph(func);
  auto &edtAnalysis = AM->getEdtAnalysis();

  /// EDT placement
  EdtPlacementHeuristics edtPlacer(&edtGraph, &edtAnalysis);
  auto edtDecisions = edtPlacer.compute(nodes);
  applyEdtPlacement(edtDecisions, nodes);

  /// DB placement
  DbPlacementHeuristics dbPlacer(&dbGraph);
  auto dbDecisions = dbPlacer.compute(func, nodes);
  applyDbPlacement(dbDecisions, nodes);

  /// Debug output
  emitDebugOutput(func, edtPlacer, edtDecisions, dbPlacer, dbDecisions);
}

void ConcurrencyPass::applyEdtPlacement(
    const SmallVector<PlacementDecision, 16> &decisions,
    const SmallVector<std::string, 8> &nodes) {
  applyPlacementDecisions(
      nodes, decisions, [](const PlacementDecision &d) { return d.edtOp; },
      [](Operation *op, unsigned idx, OpBuilder &builder) {
        op->setAttr("node",
                    IntegerAttr::get(builder.getI32Type(), (int32_t)idx));
      });
}

void ConcurrencyPass::applyDbPlacement(
    const SmallVector<DbPlacementDecision, 16> &decisions,
    const SmallVector<std::string, 8> &nodes) {
  applyPlacementDecisions(
      nodes, decisions,
      [](const DbPlacementDecision &d) -> Operation * { return d.dbAllocOp; },
      [](Operation *op, unsigned idx, OpBuilder &builder) {
        if (op) {
          op->setAttr("node",
                      IntegerAttr::get(builder.getI32Type(), (int32_t)idx));
        }
      });
}

void ConcurrencyPass::updateRuntimeQueryOperations() {
  /// For each EDT that has parallelism strategy set, update runtime query
  /// operations
  module.walk([&](EdtOp edtOp) {
    if (!edtOp.getWorkers().has_value())
      return; /// No parallelism strategy set

    bool isInterNode = (edtOp.getConcurrency() == EdtConcurrency::internode);
    EdtType edtType = edtOp.getType();

    /// Only modify operations within parallel EDTs
    if (edtType != EdtType::parallel)
      return;

    OpBuilder builder(edtOp.getContext());

    /// Walk through operations in the EDT body and update runtime queries
    edtOp.walk([&](Operation *op) {
      if (auto getWorkerOp = dyn_cast<GetCurrentWorkerOp>(op)) {
        if (isInterNode) {
          /// For internode parallelism, replace get_current_worker with
          /// get_current_node
          ARTS_INFO("Replacing arts.get_current_worker with "
                    "arts.get_current_node in internode EDT");
          builder.setInsertionPoint(getWorkerOp);
          auto getNodeOp =
              builder.create<GetCurrentNodeOp>(getWorkerOp.getLoc());
          getWorkerOp.replaceAllUsesWith(getNodeOp.getResult());
          getWorkerOp.erase();
        }
      } else if (auto getTotalWorkersOp = dyn_cast<GetTotalWorkersOp>(op)) {
        if (isInterNode) {
          /// For internode parallelism, replace get_total_workers with
          /// get_total_nodes
          ARTS_INFO("Replacing arts.get_total_workers with "
                    "arts.get_total_nodes in internode EDT");
          builder.setInsertionPoint(getTotalWorkersOp);
          auto getTotalNodesOp =
              builder.create<GetTotalNodesOp>(getTotalWorkersOp.getLoc());
          getTotalWorkersOp.replaceAllUsesWith(getTotalNodesOp.getResult());
          getTotalWorkersOp.erase();
        }
      } else if (auto getNodeOp = dyn_cast<GetCurrentNodeOp>(op)) {
        if (!isInterNode) {
          /// For intranode parallelism, replace get_current_node with
          /// get_current_worker
          ARTS_INFO("Replacing arts.get_current_node with "
                    "arts.get_current_worker in intranode EDT");
          builder.setInsertionPoint(getNodeOp);
          auto getWorkerOp =
              builder.create<GetCurrentWorkerOp>(getNodeOp.getLoc());
          getNodeOp.replaceAllUsesWith(getWorkerOp.getResult());
          getNodeOp.erase();
        }
      } else if (auto getTotalNodesOp = dyn_cast<GetTotalNodesOp>(op)) {
        if (!isInterNode) {
          /// For intranode parallelism, replace get_total_nodes with
          /// get_total_workers
          ARTS_INFO("Replacing arts.get_total_nodes with "
                    "arts.get_total_workers in intranode EDT");
          builder.setInsertionPoint(getTotalNodesOp);
          auto getTotalWorkersOp =
              builder.create<GetTotalWorkersOp>(getTotalNodesOp.getLoc());
          getTotalNodesOp.replaceAllUsesWith(getTotalWorkersOp.getResult());
          getTotalNodesOp.erase();
        }
      }
    });
  });
}

void ConcurrencyPass::adjustDbModes() {
  /// For each EDT that has parallelism strategy set, adjust datablock modes
  module.walk([&](EdtOp edtOp) {
    if (!edtOp.getWorkers().has_value())
      return; /// No parallelism strategy set

    bool isInterNode = (edtOp.getConcurrency() == EdtConcurrency::internode);

    /// If intranode, ensure datablocks use pinned mode (default)
    /// If internode, adjust based on access mode
    std::function<void(Operation *)> adjustDbMode = [&](Operation *op) {
      if (auto dbAlloc = dyn_cast<DbAllocOp>(op)) {
        if (isInterNode) {
          /// For internode: adjust based on ArtsMode
          ArtsMode accessMode = dbAlloc.getMode();
          DbMode newDbMode;
          if (accessMode == ArtsMode::in) {
            newDbMode = DbMode::read;
          } else {
            /// out or inout -> write mode
            newDbMode = DbMode::write;
          }

          /// Update the dbMode attribute
          OpBuilder builder(dbAlloc.getContext());
          dbAlloc.setDbModeAttr(
              DbModeAttr::get(builder.getContext(), newDbMode));
          ARTS_INFO("Adjusted datablock mode to "
                    << (newDbMode == DbMode::read ? "read" : "write")
                    << " for internode EDT");
        } else {
          /// For intranode: ensure pinned mode
          OpBuilder builder(dbAlloc.getContext());
          dbAlloc.setDbModeAttr(
              DbModeAttr::get(builder.getContext(), DbMode::pinned));
          ARTS_INFO("Ensured datablock mode is pinned for intranode EDT");
        }
      }

      /// Recursively check operands
      for (Value operand : op->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp())
          adjustDbMode(defOp);
      }
    };

    /// Start traversal from EDT dependencies
    for (Value dep : edtOp.getDependencies()) {
      if (Operation *defOp = dep.getDefiningOp())
        adjustDbMode(defOp);
    }
  });
}

void ConcurrencyPass::applyEdtParallelismStrategy(EdtOp edtOp) {
  /// Only apply to parallel EDTs that don't already have workers attribute set
  if (edtOp.getType() != EdtType::parallel)
    return;

  /// Already has parallelism strategy set
  if (edtOp.getWorkers().has_value())
    return;

  /// Determine parallelism strategy based on abstract machine
  int numWorkers = 0;
  EdtConcurrency concurrencyType = EdtConcurrency::intranode;

  if (!abstractMachine->isValid()) {
    ARTS_WARN("Invalid abstract machine configuration, using runtime fallback");
    /// Don't set attributes - let runtime handle it
    return;
  }

  if (abstractMachine->hasValidNodeCount()) {
    int nodeCount = abstractMachine->getNodeCount();
    if (nodeCount > 1) {
      numWorkers = nodeCount;
      concurrencyType = EdtConcurrency::internode;
      ARTS_INFO("Setting EDT parallelism: inter-node across " << nodeCount
                                                              << " nodes");
    } else if (abstractMachine->hasValidThreads()) {
      int threads = abstractMachine->getThreads();
      numWorkers = threads;
      concurrencyType = EdtConcurrency::intranode;
      ARTS_INFO("Setting EDT parallelism: intra-node across " << threads
                                                              << " threads");
    }
  } else if (abstractMachine->hasValidThreads()) {
    int threads = abstractMachine->getThreads();
    numWorkers = threads;
    concurrencyType = EdtConcurrency::intranode;
    ARTS_INFO("Setting EDT parallelism: intra-node across " << threads
                                                            << " threads");
  } else {
    /// Unknown configuration - let runtime handle it
    ARTS_INFO(
        "Unknown thread count: letting runtime determine EDT parallelism");
    return;
  }

  /// Set the attributes on the EDT operation
  OpBuilder builder(edtOp.getContext());
  edtOp.setConcurrency(concurrencyType);
  edtOp.setWorkersAttr(workersAttr::get(builder.getContext(), numWorkers));
  ARTS_INFO("Set EDT concurrency="
            << (concurrencyType == EdtConcurrency::internode ? "internode"
                                                             : "intranode")
            << ", workers=" << numWorkers);
}

void ConcurrencyPass::emitDebugOutput(
    func::FuncOp func, const EdtPlacementHeuristics &edtPlacer,
    const SmallVector<PlacementDecision, 16> &edtDecisions,
    const DbPlacementHeuristics &dbPlacer,
    const SmallVector<DbPlacementDecision, 16> &dbDecisions) {
  auto &concurrencyGraph = AM->getConcurrencyGraph(func);

  /// Print concurrency graph
  emitDebugString("edt-concurrency-pass",
                  [&](llvm::raw_string_ostream &os) -> void {
                    concurrencyGraph.print(os);
                  });

  /// Export EDT placement to JSON
  emitDebugString("edt-placement-json",
                  [&](llvm::raw_string_ostream &os) -> void {
                    edtPlacer.exportToJson(func, edtDecisions, os);
                  });

  /// Export DB placement to JSON
  emitDebugString("db-placement-json",
                  [&](llvm::raw_string_ostream &os) -> void {
                    dbPlacer.exportToJson(func, dbDecisions, os);
                  });
}

namespace mlir {
namespace arts {

std::unique_ptr<Pass> createConcurrencyPass(ArtsAnalysisManager *AM) {
  return std::make_unique<ConcurrencyPass>(AM);
}

} // namespace arts
} // namespace mlir
