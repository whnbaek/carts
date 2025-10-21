///==========================================================================
/// File: EdtInvariantCodeMotion.cpp
///==========================================================================

/// Dialects
#include "ArtsPassDetails.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

/// Arts
#include "arts/ArtsDialect.h"

/// Others
#include "arts/Transforms/EdtInvariantCodeMotion.h"

/// LLVM support
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(edt - invariant - code - motion);

using namespace mlir;
using namespace mlir::arts;

namespace {
struct EdtInvariantCodeMotionPass
    : public arts::EdtInvariantCodeMotionBase<EdtInvariantCodeMotionPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void EdtInvariantCodeMotionPass::runOnOperation() {
  ModuleOp module = getOperation();
  ARTS_INFO_HEADER(EdtInvariantCodeMotionPass);
  ARTS_DEBUG_REGION(module.dump(););

  bool changed = false;
  /// Walk through all EdtOp instances in the module.
  module.walk([&](arts::EdtOp edtOp) {
    ARTS_DEBUG_TYPE("Processing EDT:\n" << edtOp);

    /// Use the new function to move invariant code out of this EDT.
    auto movedCount = moveEdtInvariantCode(edtOp);
    if (movedCount > 0)
      changed = true;

    ARTS_INFO("Moved " << movedCount << " operations out of EDT.");
  });

  // if (!changed)
  //  markAllAnalysesPreserved();

  ARTS_INFO_FOOTER(EdtInvariantCodeMotionPass);
  ARTS_DEBUG_REGION(module.dump(););
}

///===----------------------------------------------------------------------===///
/// Pass creation
///===----------------------------------------------------------------------===///
namespace mlir {
namespace arts {
std::unique_ptr<Pass> createEdtInvariantCodeMotionPass() {
  return std::make_unique<EdtInvariantCodeMotionPass>();
}

} // namespace arts
} // namespace mlir
