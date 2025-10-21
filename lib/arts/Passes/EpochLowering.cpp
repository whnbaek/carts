///==========================================================================
/// Epoch Lowering Pass - Complete Implementation
/// Transforms arts.epoch operations into runtime-compatible function calls
/// and propagates epoch GUIDs to contained EdtCreateOps
///
/// This pass runs after EdtLowering and implements the epoch lowering process:
/// 1. Create epoch GUID for each epoch operation
/// 2. Propagate the epoch GUID to all EdtCreateOps within the epoch
/// 3. Lower epoch operations to runtime calls
/// 4. Return the epoch GUID for synchronization
///==========================================================================

#include "ArtsPassDetails.h"
#include "arts/ArtsDialect.h"
#include "arts/Codegen/ArtsCodegen.h"
#include "arts/Passes/ArtsPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(epoch_lowering);

using namespace mlir;
using namespace mlir::func;
using namespace mlir::arts;

//===----------------------------------------------------------------------===//
// Epoch Lowering Pass Implementation
//===----------------------------------------------------------------------===//
struct EpochLoweringPass : public arts::EpochLoweringBase<EpochLoweringPass> {
  explicit EpochLoweringPass(bool debug = false) : debugMode(debug) {}
  ~EpochLoweringPass() { delete AC; }

  void runOnOperation() override;

private:
  /// State
  ModuleOp module;
  ArtsCodegen *AC = nullptr;
  bool debugMode = false;
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

void EpochLoweringPass::runOnOperation() {
  module = getOperation();
  AC = new ArtsCodegen(module, debugMode);

  ARTS_INFO_HEADER(EpochLoweringPass);
  ARTS_DEBUG_REGION(module.dump(););

  /// Collect all epoch operations to process
  SmallVector<EpochOp> epochOps;
  module.walk([&](EpochOp epochOp) { epochOps.push_back(epochOp); });

  ARTS_INFO("Found " << epochOps.size() << " epoch operations to lower");

  /// Process each epoch operation
  for (EpochOp epochOp : epochOps) {
    ARTS_INFO("Lowering Epoch Op " << epochOp);

    /// If region is empty or contains only a terminator, elide the epoch
    auto &epochRegion = epochOp.getRegion();
    if (epochRegion.empty() ||
        epochRegion.front().without_terminator().empty()) {
      epochOp.erase();
      continue;
    }

    /// Create epoch GUID for the epoch (using NULL_GUID = 0)
    AC->setInsertionPoint(epochOp);
    auto guid = AC->createIntConstant(0, AC->Int64, epochOp.getLoc());
    auto edtSlot =
        AC->createIntConstant(DEFAULT_EDT_SLOT, AC->Int32, epochOp.getLoc());
    auto currentEpoch = AC->createEpoch(guid, edtSlot, epochOp.getLoc());

    /// Find all EdtCreateOp operations within this epoch region and update them
    /// with the epoch GUID
    SmallVector<EdtCreateOp, 8> edtCreatesToUpdate;
    epochOp.walk([&](EdtCreateOp edtCreateOp) {
      /// Only update EdtCreateOp operations that don't already have an epoch
      /// GUID
      if (!edtCreateOp.getEpochGuid())
        edtCreatesToUpdate.push_back(edtCreateOp);
    });

    ARTS_INFO("Updating " << edtCreatesToUpdate.size()
                          << " EdtCreateOps with epoch GUID");

    /// Update all EdtCreateOps in this epoch with the epoch GUID
    for (EdtCreateOp edtCreateOp : edtCreatesToUpdate) {
      AC->setInsertionPoint(edtCreateOp);

      /// Build new EdtCreateOp with epoch GUID
      auto newEdtCreateOp = AC->create<EdtCreateOp>(
          edtCreateOp.getLoc(), edtCreateOp.getParamMemref(),
          edtCreateOp.getDepCount(), edtCreateOp.getRoute(), currentEpoch);

      /// Copy attributes from the old operation
      for (auto attr : edtCreateOp->getAttrs())
        newEdtCreateOp->setAttr(attr.getName(), attr.getValue());

      /// Replace all uses of the old EdtCreateOp with the new one
      edtCreateOp->replaceAllUsesWith(newEdtCreateOp);

      /// Replace the old EdtCreateOp
      edtCreateOp->erase();
    }

    /// Move operations out of epoch region to before the epoch
    auto &epochRegionForMove = epochOp.getRegion();
    if (!epochRegionForMove.empty()) {
      auto &epochBlock = epochRegionForMove.front();
      SmallVector<Operation *> opsToMove;
      for (auto &innerOp : epochBlock.without_terminator()) {
        if (!isa<EpochOp>(innerOp))
          opsToMove.push_back(&innerOp);
      }
      for (auto *opToMove : opsToMove)
        opToMove->moveBefore(epochOp);
    }

    /// Wait on the epoch
    AC->setInsertionPointAfter(epochOp);
    AC->waitOnHandle(currentEpoch, epochOp.getLoc());

    /// Replace the epoch operation with its result (the epoch GUID)
    epochOp.replaceAllUsesWith(currentEpoch);
    epochOp.erase();
  }

  ARTS_INFO_FOOTER(EpochLoweringPass);
  ARTS_DEBUG_REGION(module.dump(););
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace arts {

std::unique_ptr<Pass> createEpochLoweringPass() {
  return std::make_unique<EpochLoweringPass>();
}

} // namespace arts
} // namespace mlir
