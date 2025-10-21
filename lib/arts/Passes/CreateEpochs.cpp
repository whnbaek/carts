///==========================================================================
/// File: CreateEpochs.cpp
///==========================================================================

/// Dialects
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
/// Arts
#include "ArtsPassDetails.h"
#include "arts/ArtsDialect.h"
#include "arts/Passes/ArtsPasses.h"
#include "arts/Utils/ArtsUtils.h"
/// Other
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(create_epochs);

using namespace mlir;
using namespace mlir::arts;

static void clearIsSyncAttr(EdtOp op) {
  auto newTypeAttr = EdtTypeAttr::get(op.getContext(), EdtType::single);
  op.setTypeAttr(newTypeAttr);
}

static void setIsTaskAttr(EdtOp op) {
  auto newTypeAttr = EdtTypeAttr::get(op.getContext(), EdtType::task);
  op.setTypeAttr(newTypeAttr);
}

/// Helper function to process synchronous EDT ops.
static void processSyncEdtOp(EdtOp op) {
  /// Only process EDT ops with sync attribute.
  if (op.getTypeAttr().getValue() != EdtType::sync)
    return;
  ARTS_DEBUG("Processing Sync EDT Op: " << op);
  auto loc = op.getLoc();
  OpBuilder builder(op);
  auto epochOp = builder.create<EpochOp>(loc);
  auto &epochBlock = epochOp.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(&epochBlock);
  builder.create<YieldOp>(loc);

  /// Move the EDT op to the end of the new block.
  op->moveBefore(&epochBlock, --epochBlock.end());

  /// Remove the sync attribute and mark the op as a task.
  clearIsSyncAttr(op);
  setIsTaskAttr(op);
}

/// Helper function to process barrier ops
static void processBarrierOp(BarrierOp barrier) {
  ARTS_DEBUG("Processing BarrierOp");
  auto loc = barrier.getLoc();

  bool hasParentEdt = true;
  Operation *parentOp = barrier->getParentOfType<EdtOp>();
  if (!parentOp) {
    parentOp = barrier->getParentOfType<func::FuncOp>();
    hasParentEdt = false;
  }

  /// Determine the appropriate parent insertion point.
  Region *parentIP = nullptr;
  parentOp->walk([&](EdtOp childEdt) {
    /// Check if the childEDT is a direct child of the parentOp when necessary.
    if (hasParentEdt && (childEdt->getParentOfType<EdtOp>() != parentOp))
      return;

    /// Skip the childEDT if it cannot reach the barrier.
    if (!isReachable(childEdt, barrier))
      return;

    /// Initialize or update the parent insertion point.
    if (!parentIP)
      parentIP = childEdt->getParentRegion();
    else {
      while (parentIP && !parentIP->isAncestor(childEdt->getParentRegion()))
        parentIP = parentIP->getParentRegion();
    }
  });
  assert(parentIP && "Parent insertion point cannot be null");

  /// Collect operations preceding the barrier to avoid iterator invalidation.
  bool finishCollection = false;
  SmallVector<Operation *, 8> opsToMove;
  for (Block &block : *parentIP) {
    for (Operation &op : block) {
      if (&op == barrier) {
        finishCollection = true;
        break;
      }
      opsToMove.push_back(&op);
    }
    if (finishCollection)
      break;
  }

  /// Create a new epoch op and prepare its region.
  OpBuilder builder(parentIP);
  auto epochOp = builder.create<EpochOp>(loc);
  auto &epochRegion = epochOp.getRegion();
  if (epochRegion.empty())
    epochRegion.push_back(new Block());
  Block *newBlock = &epochRegion.front();

  /// Move collected operations into the new epoch block.
  for (Operation *op : opsToMove) {
    ARTS_INFO("Moving operation: " << *op);
    op->moveBefore(newBlock, newBlock->end());
  }

  /// Finalize epoch block by inserting a yield op.
  builder.setInsertionPointToEnd(newBlock);
  builder.create<YieldOp>(loc);

  /// Remove the barrier op.
  barrier.erase();
}

///==========================================================================
// Pass Implementation
///==========================================================================
namespace {
struct CreateEpochsPass : public CreateEpochsBase<CreateEpochsPass> {
  void runOnOperation() override;
};
} // end namespace

void CreateEpochsPass::runOnOperation() {
  ModuleOp module = getOperation();
  ARTS_INFO_HEADER(CreateEpochsPass);
  ARTS_DEBUG_REGION(module.dump(););

  /// Process Sync EDT Ops: for each EDT op that is sync, create an epoch op
  /// and move the EDT op inside the epoch op.
  ARTS_DEBUG_HEADER(ProcessSyncEdtOp);
  module.walk([](EdtOp op) { processSyncEdtOp(op); });
  ARTS_DEBUG_FOOTER(ProcessSyncEdtOp);

  /// Process Barrier Ops: for each barrier, collect all EDTs that are affected
  /// by the barrier and embed them in a new epoch op.
  ARTS_DEBUG_HEADER(ProcessBarrierOp);
  module.walk([&](BarrierOp barrier) { processBarrierOp(barrier); });
  ARTS_DEBUG_FOOTER(ProcessBarrierOp);

  ARTS_INFO_FOOTER(CreateEpochsPass);
  ARTS_DEBUG_REGION(module.dump(););
}

//==========================================================================
/// Pass creation
//==========================================================================
namespace mlir {
namespace arts {
std::unique_ptr<Pass> createCreateEpochsPass() {
  return std::make_unique<CreateEpochsPass>();
}
} // namespace arts
} // namespace mlir
