///==========================================================================
/// File: DbDataFlowAnalysis.cpp
/// Environment-based DDG construction with proper dependency tracking.
///==========================================================================

#include "arts/Analysis/Db/DbDataFlowAnalysis.h"

#include "arts/Analysis/Db/DbAliasAnalysis.h"
#include "arts/Analysis/Db/DbAnalysis.h"
#include "arts/Analysis/Graphs/Db/DbEdge.h"
#include "arts/Analysis/Graphs/Db/DbGraph.h"
#include "arts/Analysis/Graphs/Db/DbNode.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(db_dataflow);

using namespace mlir;
using namespace mlir::arts;

static int analysisDepth = 0;
struct IndentScope {
  IndentScope() { ++analysisDepth; }
  ~IndentScope() { --analysisDepth; }
};

#define INDENT() std::string(analysisDepth * 2, ' ')

DbDataFlowAnalysis::DbDataFlowAnalysis() {
  ARTS_DEBUG("DbDataFlowAnalysis constructed");
}

void DbDataFlowAnalysis::initialize(DbGraph *graphIn, DbAnalysis *analysisIn) {
  ARTS_DEBUG("Initializing DbDataFlowAnalysis");
  graph = graphIn;
  analysis = analysisIn;
  aliasAA = analysis->getAliasAnalysis();
  assert(graph && "Graph cannot be null");
  assert(analysis && "Analysis cannot be null");
  assert(aliasAA && "Alias analysis cannot be null");
  ARTS_INFO("Initialized DDG dataflow analysis");
}

void DbDataFlowAnalysis::runAnalysis(func::FuncOp func) {
  ARTS_INFO("Running DDG dataflow analysis on function: " << func.getName());
  DbDataFlowAnalysis::Environment env;
  processRegion(func.getBody(), env);
  ARTS_INFO("Finished DDG dataflow analysis");
}

std::pair<DbDataFlowAnalysis::Environment, bool>
DbDataFlowAnalysis::processRegion(Region &region,
                                  DbDataFlowAnalysis::Environment &env) {
  DbDataFlowAnalysis::Environment newEnv = env;
  bool changed = false;

  for (Block &block : region) {
    for (Operation &op : block) {
      IndentScope scope;
      std::pair<DbDataFlowAnalysis::Environment, bool> result;

      if (auto edtOp = dyn_cast<EdtOp>(&op))
        result = processEdt(edtOp, newEnv);
      else if (auto epochOp = dyn_cast<EpochOp>(&op))
        result = processRegion(epochOp.getBody(), newEnv);
      else if (auto ifOp = dyn_cast<scf::IfOp>(&op))
        result = processIf(ifOp, newEnv);
      else if (auto forOp = dyn_cast<scf::ForOp>(&op))
        result = processFor(forOp, newEnv);
      else
        continue;

      newEnv = std::move(result.first);
      changed |= result.second;
    }
  }

  ARTS_DEBUG(INDENT() << "Finished processing region. Changed: "
                      << (changed ? "true" : "false"));
  return {std::move(newEnv), changed};
}

std::pair<DbDataFlowAnalysis::Environment, bool>
DbDataFlowAnalysis::processEdt(EdtOp edtOp,
                               DbDataFlowAnalysis::Environment &env) {
  ARTS_DEBUG(INDENT() << "Processing EDT");

  auto edtDeps = edtOp.getDependenciesAsVector();
  bool changed = false;
  DbDataFlowAnalysis::Environment newEnv = env;

  SmallVector<DbAcquireNode *, 4> inputAcquires;
  SmallVector<DbAcquireNode *, 4> outputAcquires;

  /// Categorize acquires as inputs (readers) or outputs (writers)
  for (auto dep : edtDeps) {
    auto acquireOp = dyn_cast<DbAcquireOp>(dep.getDefiningOp());
    if (!acquireOp)
      continue;

    auto *acquireNode = graph->getDbAcquireNode(acquireOp);
    if (!acquireNode)
      continue;

    auto mode = acquireOp.getMode();
    if (mode == ArtsMode::in || mode == ArtsMode::inout)
      inputAcquires.push_back(acquireNode);
    if (mode == ArtsMode::out || mode == ArtsMode::inout)
      outputAcquires.push_back(acquireNode);
  }

  /// Process input acquires: find RAW dependencies from previous writers
  ARTS_DEBUG(INDENT() << "  Processing " << inputAcquires.size()
                      << " input acquires (RAW)");
  for (auto *consumer : inputAcquires) {
    /// Find all potential producer (writer) definitions in environment
    auto prodDefs = findDefinitions(consumer, newEnv);

    if (prodDefs.empty()) {
      ARTS_DEBUG(INDENT() << "    - No definition for " << consumer->getHierId()
                          << " (entry dependency)");
      /// TODO: Create edge from entry node if we add one
      continue;
    }

    ARTS_DEBUG(INDENT() << "    - Found " << prodDefs.size()
                        << " potential producers for "
                        << consumer->getHierId());

    /// Create RAW edges from producers to this consumer
    for (auto *producer : prodDefs) {
      if (!mayDepend(producer, consumer))
        continue;

      ARTS_DEBUG(INDENT() << "      Creating RAW edge: "
                          << producer->getHierId() << " -> "
                          << consumer->getHierId());

      auto *edge = new DbDepEdge(producer, consumer, DbDepType::RAW);
      if (graph->addEdge(producer, consumer, edge)) {
        ARTS_DEBUG(INDENT() << "        Edge created successfully");
      } else {
        ARTS_DEBUG(INDENT() << "        Edge already exists");
        delete edge;
      }
    }
  }

  /// Process output acquires: find WAW dependencies and update environment
  ARTS_DEBUG(INDENT() << "  Processing " << outputAcquires.size()
                      << " output acquires (WAW)");
  for (auto *writer : outputAcquires) {
    DbAllocNode *parentAlloc = writer->getParent();

    /// Find previous writer definitions to create WAW dependencies
    auto prodDefs = findDefinitions(writer, newEnv);

    for (auto *producer : prodDefs) {
      if (!mayDepend(producer, writer))
        continue;

      ARTS_DEBUG(INDENT() << "    - Creating WAW edge: "
                          << producer->getHierId() << " -> "
                          << writer->getHierId());

      auto *edge = new DbDepEdge(producer, writer, DbDepType::WAW);
      if (graph->addEdge(producer, writer, edge)) {
        ARTS_DEBUG(INDENT() << "      WAW edge created successfully");
      } else {
        ARTS_DEBUG(INDENT() << "      WAW edge already exists");
        delete edge;
      }
    }

    /// Update environment: this writer now defines the allocation regardless of
    /// whether a new edge was emitted.
    auto &defs = newEnv[parentAlloc];
    bool needsUpdate = defs.size() != 1 || !defs.contains(writer);
    if (needsUpdate) {
      ARTS_DEBUG(INDENT() << "    - Updating environment: "
                          << writer->getHierId() << " is new definition for "
                          << parentAlloc->getHierId());
      defs.clear();
      defs.insert(writer);
      changed = true;
    }
  }

  return {newEnv, changed};
}

std::pair<DbDataFlowAnalysis::Environment, bool>
DbDataFlowAnalysis::processFor(scf::ForOp forOp,
                               DbDataFlowAnalysis::Environment &env) {
  ARTS_DEBUG(INDENT() << "Processing scf.for loop (fixed-point iteration)");

  DbDataFlowAnalysis::Environment loopEnv = env;
  bool overallChanged = false;
  unsigned iteration = 0;
  constexpr unsigned maxIterations = 5;

  size_t prevEdgeCount = graph->getEdgeCount();

  /// Fixed-point iteration for loop-carried dependencies
  while (iteration < maxIterations) {
    size_t edgeCountBefore = graph->getEdgeCount();

    auto bodyResult = processRegion(forOp.getRegion(), loopEnv);
    overallChanged |= bodyResult.second;

    bool envChanged = unionInto(loopEnv, bodyResult.first);
    overallChanged |= envChanged;

    size_t edgeCountAfter = graph->getEdgeCount();

    ARTS_DEBUG(INDENT() << "  Iteration " << iteration + 1
                        << " - envChanged: " << (envChanged ? "true" : "false")
                        << ", edges(before/after): " << edgeCountBefore
                        << " -> " << edgeCountAfter);

    /// Check for convergence: no new info and no new edges
    if (!bodyResult.second && !envChanged && edgeCountAfter == prevEdgeCount) {
      ARTS_DEBUG(INDENT() << "  Converged: no changes in iteration "
                          << iteration + 1);
      break;
    }

    prevEdgeCount = edgeCountAfter;
    ++iteration;
  }

  if (iteration >= maxIterations) {
    ARTS_DEBUG(INDENT() << "  Max iterations reached without full convergence");
  } else {
    ARTS_DEBUG(INDENT() << "  Loop converged after " << iteration + 1
                        << " iterations");
  }

  return {loopEnv, overallChanged};
}

std::pair<DbDataFlowAnalysis::Environment, bool>
DbDataFlowAnalysis::processIf(scf::IfOp ifOp,
                              DbDataFlowAnalysis::Environment &env) {
  ARTS_DEBUG(INDENT() << "Processing scf.if (then/else merge)");

  auto thenResult = processRegion(ifOp.getThenRegion(), env);

  DbDataFlowAnalysis::Environment elseEnv = env;
  bool elseChanged = false;
  if (!ifOp.getElseRegion().empty()) {
    auto elseResult = processRegion(ifOp.getElseRegion(), env);
    elseEnv = std::move(elseResult.first);
    elseChanged = elseResult.second;
  }

  /// Merge both branches: conservative union of definitions
  DbDataFlowAnalysis::Environment merged = thenResult.first;
  bool mergeChanged = unionInto(merged, elseEnv);
  bool changed = thenResult.second || elseChanged || mergeChanged;

  ARTS_DEBUG(INDENT() << "  Merged then/else - changed: "
                      << (changed ? "true" : "false"));
  return {std::move(merged), changed};
}

SmallVector<DbAcquireNode *>
DbDataFlowAnalysis::findDefinitions(DbAcquireNode *acquire,
                                    DbDataFlowAnalysis::Environment &env) {
  SmallVector<DbAcquireNode *> defs;
  DbAllocNode *parentAlloc = acquire->getParent();

  auto it = env.find(parentAlloc);
  if (it != env.end()) {
    for (auto *def : it->second) {
      /// Check alias via roots, then refine with overlap classification
      auto aliasRes = aliasAA->classifyAlias(*def, *acquire);
      if (aliasRes == DbAliasAnalysis::AliasResult::NoAlias)
        continue;
      if (aliasRes == DbAliasAnalysis::AliasResult::MayAlias) {
        auto k = aliasAA->estimateOverlap(def, acquire);
        if (k == DbAliasAnalysis::OverlapKind::Disjoint)
          continue;
      }
      defs.push_back(def);
    }
  }

  return defs;
}

DbDataFlowAnalysis::Environment DbDataFlowAnalysis::mergeEnvironments(
    const DbDataFlowAnalysis::Environment &env1,
    const DbDataFlowAnalysis::Environment &env2) {
  DbDataFlowAnalysis::Environment merged = env1;
  unionInto(merged, env2);
  return merged;
}

bool DbDataFlowAnalysis::unionInto(
    DbDataFlowAnalysis::Environment &target,
    const DbDataFlowAnalysis::Environment &source) {
  bool changed = false;
  for (const auto &pair : source) {
    auto &defs = target[pair.first];
    for (auto *acquire : pair.second)
      if (!defs.contains(acquire)) {
        defs.insert(acquire);
        changed = true;
      }
  }
  return changed;
}

bool DbDataFlowAnalysis::mayDepend(DbAcquireNode *producer,
                                   DbAcquireNode *consumer) {
  /// Must belong to same allocation (already ensured by findDefinitions)
  assert(producer->getParent() == consumer->getParent() &&
         "Producer and consumer must share same allocation");

  /// Check EDT context: must have same parent EDT to have direct dependency
  /// If different EDT parents, we need more sophisticated analysis
  auto prodEdt = producer->getEdtUser();
  auto consEdt = consumer->getEdtUser();

  if (prodEdt != consEdt) {
    ARTS_DEBUG(
        INDENT() << "      Different EDT contexts - need inter-EDT analysis");
    /// For now, be conservative: assume may depend
  }

  /// Producer must write (out or inout)
  auto prodMode = producer->getDbAcquireOp().getMode();
  if (prodMode != ArtsMode::out && prodMode != ArtsMode::inout) {
    return false;
  }

  /// Consumer can read or write
  auto consMode = consumer->getDbAcquireOp().getMode();

  /// WAW: Write-After-Write
  if (prodMode == ArtsMode::out && consMode == ArtsMode::out) {
    return true;
  }

  /// RAW: Read-After-Write (most common)
  if ((prodMode == ArtsMode::out || prodMode == ArtsMode::inout) &&
      (consMode == ArtsMode::in || consMode == ArtsMode::inout)) {
    return true;
  }

  /// WAR: Write-After-Read (anti-dependency) - less critical for correctness
  /// but may be needed for optimization
  if ((prodMode == ArtsMode::in || prodMode == ArtsMode::inout) &&
      (consMode == ArtsMode::out || consMode == ArtsMode::inout)) {
    /// For now, we don't track WAR dependencies
    return false;
  }

  return false;
}
