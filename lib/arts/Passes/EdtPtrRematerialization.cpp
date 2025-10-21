///==========================================================================
/// File: EdtPtrRematerialization.cpp
///==========================================================================

/// Dialects
#include "ArtsPassDetails.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

/// Arts
#include "arts/ArtsDialect.h"
#include "arts/Transforms/EdtPtrRematerialization.h"

#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(edt_ptr_rematerialization);

using namespace mlir;
using namespace mlir::arts;

namespace {
struct EdtPtrRematerializationPass
    : public arts::EdtPtrRematerializationBase<EdtPtrRematerializationPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void EdtPtrRematerializationPass::runOnOperation() {
  ModuleOp module = getOperation();
  ARTS_INFO_HEADER(EdtPtrRematerializationPass);
  ARTS_DEBUG_REGION(module.dump(););

  /// Walk through all EdtOp instances in the module.
  module.walk([&](arts::EdtOp edtOp) { rematerializePointersInEdt(edtOp); });

  ARTS_INFO_FOOTER(EdtPtrRematerializationPass);
  ARTS_DEBUG_REGION(module.dump(););
}

///===----------------------------------------------------------------------===///
/// Pass creation
///===----------------------------------------------------------------------===///
namespace mlir {
namespace arts {
std::unique_ptr<Pass> createEdtPtrRematerializationPass() {
  return std::make_unique<EdtPtrRematerializationPass>();
}

} // namespace arts
} // namespace mlir
