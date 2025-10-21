///==========================================================================
/// File: LoopAnalysis.h
///
/// This class analyzes the module and itentifies the set of string memref.
///==========================================================================

#ifndef CARTS_ANALYSIS_LOOPANALYSIS_H
#define CARTS_ANALYSIS_LOOPANALYSIS_H

#include "mlir/IR/BuiltinAttributes.h"
/// Dialects
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace arts {

//===----------------------------------------------------------------------===//
// LoopAnalysis
//===----------------------------------------------------------------------===//
struct LoopInfo {
  bool isAffine;
  affine::AffineForOp affine;
  scf::ParallelOp scfParallel;
  scf::ForOp scfFor;
  Value inductionVar;

  LoopInfo(bool isAffine, affine::AffineForOp affine,
           scf::ParallelOp scfParallel, Value inductionVar)
      : isAffine(isAffine), affine(affine), scfParallel(scfParallel),
        inductionVar(inductionVar) {}
};

struct LoopAnalysis {
  explicit LoopAnalysis(Operation *module) {
    this->module = cast<ModuleOp>(module);
    run();
  }
  ~LoopAnalysis() {
    for (auto &loop : loopInfoMap) {
      delete loop.second;
    }
  }

public:
  /// Main function that performs analysis.
  void run();

  /// Get loop information for a given operation.
  LoopInfo *getLoopInfo(Operation *op);

  /// Check if a value is dependent on a loop and return the loops it depends
  /// on.
  bool isDependentOnLoop(Value val, SmallVectorImpl<Operation *> &loops);

  /// For a given operation, collect enclosing loops
  void collectEnclosingLoops(Operation *op,
    SmallVectorImpl<LoopInfo *> &enclosingLoops);

  /// Collect loops whose induction variables affect the given operation's
  /// operands. The returned vector contains unique loop operations.
  void collectAffectingLoops(Operation *op,
    SmallVectorImpl<Operation *> &affectingLoops);

  /// Collect loops whose induction variables affect the given SSA value.
  /// The returned vector contains loop operations that the value depends on.
  void collectAffectingLoops(Value val,
    SmallVectorImpl<Operation *> &affectingLoops);

private:
  ModuleOp module;
  /// List of loops in the module.
  SmallVector<Operation *, 4> loops;
  /// Maps a loop operation to the information about the loop.
  DenseMap<Operation *, LoopInfo*> loopInfoMap;
  /// Collect loops and variables that depend on them.
  DenseMap<Value, SmallVector<Operation *, 4>> loopValsMap;

  /// Analyze the induction variable of the collected loop.
  void analyzeLoopIV();
};

} // namespace arts
} // namespace mlir

#endif // CARTS_ANALYSIS_LOOPANALYSIS_H