///==========================================================================
/// File: Edt.cpp
///==========================================================================

/// Dialects
#include "arts/Utils/ArtsUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
/// Arts
#include "ArtsPassDetails.h"
#include "arts/Analysis/ArtsAnalysisManager.h"
#include "arts/Analysis/Graphs/Edt/EdtGraph.h"
#include "arts/ArtsDialect.h"
#include "arts/Passes/ArtsPasses.h"
/// Other
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
/// Debug
#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(edt);

using namespace mlir;
using namespace mlir::func;
using namespace mlir::arts;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//
namespace {
struct EdtPass : public arts::EdtBase<EdtPass> {
  EdtPass(ArtsAnalysisManager *AM, bool runAnalysis) : AM(AM) {
    assert(AM && "ArtsAnalysisManager must be provided externally");
    this->runAnalysis = runAnalysis;
  }
  void runOnOperation() override;
  bool convertParallelIntoSingle(EdtOp &op);

  /// Utility for creating EDT with merged dependencies and region transfer
  EdtOp createEdtWithMergedDepsAndRegion(OpBuilder &builder, Location loc,
                                         arts::EdtType newType, EdtOp sourceOp,
                                         ArrayRef<Value> additionalDeps = {});

  bool processParallelEdts();
  bool processSyncTaskEdts();
  bool removeBarriers();

  /// Graph-driven cleanups
  bool removeRedundantBarriersWithGraphs(func::FuncOp func,
                                         arts::EdtGraph &graph);

private:
  ModuleOp module;
  ArtsAnalysisManager *AM = nullptr;
  SetVector<Operation *> opsToRemove;
};
} // namespace

void EdtPass::runOnOperation() {
  module = getOperation();
  ARTS_INFO_HEADER(EdtPass);
  ARTS_DEBUG_REGION(module.dump(););

  if (runAnalysis) {
    ARTS_INFO("Running EDT pass with analysis");
    // removeBarriers();
  } else {
    ARTS_INFO("Running EDT pass without analysis");
    processParallelEdts();
    processSyncTaskEdts();
  }

  /// Remove ops marked for removal
  removeOps(module, opsToRemove);

  ARTS_INFO_FOOTER(EdtPass);
  ARTS_DEBUG_REGION(module.dump(););
}

/// Remove unconditional barriers that provide no additional ordering beyond
/// computed EDT dependencies (graph-informed pruning).
bool EdtPass::removeBarriers() {
  bool changed = false;
  module.walk([&](func::FuncOp func) {
    auto &edtGraph = AM->getEdtGraph(func);
    if (edtGraph.size() == 0)
      return;

    changed |= removeRedundantBarriersWithGraphs(func, edtGraph);
  });
  return changed;
}

bool EdtPass::processParallelEdts() {
  /// Gather all parallel-EDT ops in the module.
  SmallVector<EdtOp, 8> parallelOps;
  module.walk([&](EdtOp edt) {
    if (edt.getType() == arts::EdtType::parallel)
      parallelOps.push_back(edt);
  });

  /// Try to convert parallel EDTs into single EDTs if they contain exactly one
  /// child.
  for (EdtOp op : parallelOps)
    convertParallelIntoSingle(op);
  return true;
}

/// Converts a parallel EDT region into a sync-task EDT when it contains a
/// single inner `arts.edt` and no other ops beyond terminators/barriers.
bool EdtPass::convertParallelIntoSingle(EdtOp &op) {
  uint32_t numOps = 0;
  EdtOp singleOp = nullptr;

  for (auto &block : op.getRegion()) {
    for (auto &inst : block) {
      ++numOps;
      if (auto edt = dyn_cast<arts::EdtOp>(&inst)) {
        if (edt.getType() == arts::EdtType::single) {
          if (singleOp)
            llvm_unreachable(
                "Multiple single ops in parallel op not supported");
          singleOp = edt;
        } else
          return false;

      } else if (isa<arts::YieldOp>(&inst) || isa<arts::BarrierOp>(&inst))
        continue;
      else {
        return false;
      }
    }
  }

  /// Account for at least edt, yield, and possibly barrier
  if (!singleOp || numOps < 3)
    return false;

  /// Create a new EDT operation using the utility function
  OpBuilder builder(op);
  SmallVector<Value> parallelDeps(op.getDependencies().begin(),
                                  op.getDependencies().end());
  auto newEdt = createEdtWithMergedDepsAndRegion(
      builder, op.getLoc(), arts::EdtType::sync, singleOp, parallelDeps);

  /// Replace the parallel EDT with the new EDT and erase the single EDT
  op->replaceAllUsesWith(newEdt);
  singleOp->erase();

  /// Mark the parallel EDT for removal
  opsToRemove.insert(op);

  ARTS_INFO("Converted parallel EDT into single EDT");
  return true;
}

/// Utility function to create a new EDT operation with merged dependencies
/// and transfer the region from the source operation.
EdtOp EdtPass::createEdtWithMergedDepsAndRegion(
    OpBuilder &builder, Location loc, arts::EdtType newType, EdtOp sourceOp,
    ArrayRef<Value> additionalDeps) {

  /// Collect all dependencies from source operation
  SetVector<Value> allDeps;
  for (Value dep : sourceOp.getDependencies())
    allDeps.insert(dep);

  /// Add any additional dependencies
  for (Value dep : additionalDeps)
    allDeps.insert(dep);

  /// Create new EDT operation with merged dependencies and intranode
  /// concurrency
  auto newEdt = builder.create<arts::EdtOp>(
      loc, newType, EdtConcurrency::intranode, allDeps.getArrayRef());

  /// Transfer region from source to new operation
  Region &sourceRegion = sourceOp.getRegion();
  Region &newRegion = newEdt.getRegion();

  /// Clear new region and transfer blocks
  if (newRegion.empty())
    newRegion.push_back(new Block());
  Block &newBody = newRegion.front();
  newBody.clear();

  /// Move operations from source region to new region
  SmallVector<Operation *, 16> opsToMove;
  for (auto &block : sourceRegion) {
    for (auto &op : block)
      opsToMove.push_back(&op);
  }

  /// Move operations to new region
  for (auto *opToMove : opsToMove)
    opToMove->moveBefore(&newBody, newBody.end());

  return newEdt;
}

bool EdtPass::processSyncTaskEdts() {
  /// If the given single EdtOp is not nested within another EdtOp (i.e., is
  /// top-level), and is marked as sync, embed its region's contents in an
  /// arts::EpochOp. This effectively assigns the work to the master thread,
  /// avoiding unnecessary signal/sync overhead. The EdtOp itself is erased
  /// after its body is moved.
  auto convertToEpoch = [](EdtOp &op) -> bool {
    OpBuilder builder(op);
    /// If the op is not top-level, return false.
    if (op->getParentOfType<EdtOp>())
      return false;

    /// Create an arts::EpochOp and its block
    auto loc = op.getLoc();
    auto epochOp = builder.create<arts::EpochOp>(loc);
    auto &epochBlock = epochOp.getRegion().emplaceBlock();
    builder.setInsertionPointToEnd(&epochBlock);
    builder.create<arts::YieldOp>(loc);

    /// Move all operations except the terminator from the EdtOp's region to the
    /// epoch block
    Block *edtBody = &op.getRegion().front();

    /// Collect operations to move before moving them
    SmallVector<Operation *, 8> opsToMove;
    for (Operation &childOp : edtBody->without_terminator())
      opsToMove.push_back(&childOp);

    /// Move all operations to the epoch block
    for (Operation *childOp : opsToMove)
      childOp->moveBefore(epochBlock.getTerminator());

    /// Erase the now-empty EdtOp
    op.erase();

    builder.setInsertionPointAfter(epochOp);
    return true;
  };

  /// Collect all sync task EDTs
  SmallVector<EdtOp, 8> syncTaskOps;
  module.walk([&](EdtOp edt) {
    if (edt.getType() == arts::EdtType::sync)
      syncTaskOps.push_back(edt);
  });

  if (syncTaskOps.empty())
    return false;

  /// Try to convert each sync task-EDT to an EpochOp.
  for (EdtOp op : syncTaskOps) {
    op.setType(arts::EdtType::task);
    convertToEpoch(op);
  }
  return true;
}

bool EdtPass::removeRedundantBarriersWithGraphs(func::FuncOp func,
                                                arts::EdtGraph &graph) {
  bool changed = false;

  /// Collect barriers within this function and check redundancy
  SmallVector<arts::BarrierOp, 8> toErase;
  func.walk([&](arts::BarrierOp barrier) {
    Block *block = barrier->getBlock();
    /// Partition EDTs in the same block into before/after
    SmallVector<arts::EdtOp, 8> beforeTasks;
    SmallVector<arts::EdtOp, 8> afterTasks;
    bool pastBarrier = false;
    for (Operation &op : *block) {
      if (&op == barrier.getOperation()) {
        pastBarrier = true;
        continue;
      }
      if (auto edt = dyn_cast<arts::EdtOp>(&op)) {
        (pastBarrier ? afterTasks : beforeTasks).push_back(edt);
      }
    }

    if (beforeTasks.empty() || afterTasks.empty())
      return;

    bool redundant = true;
    for (arts::EdtOp a : beforeTasks) {
      for (arts::EdtOp b : afterTasks) {
        if (!graph.isEdtReachable(a, b)) {
          redundant = false;
          break;
        }
      }
      if (!redundant)
        break;
    }

    if (redundant)
      toErase.push_back(barrier);
  });

  for (auto b : toErase) {
    ARTS_INFO("Removing redundant barrier");
    b.erase();
    changed = true;
  }
  return changed;
}

///===----------------------------------------------------------------------===///
/// Pass creation
///===----------------------------------------------------------------------===///
namespace mlir {
namespace arts {
std::unique_ptr<Pass> createEdtPass(ArtsAnalysisManager *AM, bool runAnalysis) {
  return std::make_unique<EdtPass>(AM, runAnalysis);
}
} // namespace arts
} // namespace mlir