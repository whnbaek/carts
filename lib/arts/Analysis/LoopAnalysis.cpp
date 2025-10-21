///==========================================================================
/// File: LoopAnalysis.cpp
///==========================================================================

#include "arts/Analysis/LoopAnalysis.h"
#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(loop_analysis);

using namespace mlir;
using namespace arts;
using namespace affine;

void LoopAnalysis::run() {
  ARTS_DEBUG_HEADER(LoopAnalysis);

  /// Analyze scf::ForOp loops.
  module.walk([&](Operation *op) {
    if (auto fop = dyn_cast<scf::ForOp>(op)) {
      loops.push_back(fop);
      loopInfoMap[fop] = new LoopInfo(false, {}, {}, fop.getInductionVar());
    } else if (auto aop = dyn_cast<affine::AffineForOp>(op)) {
      loops.push_back(aop);
      loopInfoMap[aop] = new LoopInfo(true, aop, {}, aop.getInductionVar());
    }
  });

  ARTS_DEBUG_REGION({
    if (loops.empty()) {
      ARTS_WARN("No loops found in module");
    } else {
      ARTS_INFO("Discovered " << loops.size() << " loops");
    }
  });

  /// Analyze induction variables.
  analyzeLoopIV();
  ARTS_DEBUG_FOOTER(LoopAnalysis);
}

LoopInfo *LoopAnalysis::getLoopInfo(Operation *op) {
  if (loopInfoMap.count(op))
    return loopInfoMap[op];
  return nullptr;
}

bool LoopAnalysis::isDependentOnLoop(Value val,
                                     SmallVectorImpl<Operation *> &loops) {
  if (loopValsMap.count(val)) {
    loops = loopValsMap[val];
    return true;
  }
  return false;
}

void LoopAnalysis::collectEnclosingLoops(
    Operation *op, SmallVectorImpl<LoopInfo *> &enclosingLoops) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
    if (auto ap = dyn_cast<AffineForOp>(p))
      enclosingLoops.push_back(getLoopInfo(ap));
    else if (auto po = dyn_cast<scf::ParallelOp>(p))
      enclosingLoops.push_back(getLoopInfo(po));
    else if (auto fo = dyn_cast<scf::ForOp>(p))
      enclosingLoops.push_back(getLoopInfo(fo));
  }
  std::reverse(enclosingLoops.begin(), enclosingLoops.end());
}

void LoopAnalysis::collectAffectingLoops(
    Operation *op, SmallVectorImpl<Operation *> &affectingLoops) {
  DenseSet<Operation *> uniq;
  for (Value operand : op->getOperands()) {
    SmallVector<Operation *, 4> loopsForVal;
    if (isDependentOnLoop(operand, loopsForVal)) {
      for (Operation *l : loopsForVal) {
        if (uniq.insert(l).second)
          affectingLoops.push_back(l);
      }
    }
  }
}

void LoopAnalysis::collectAffectingLoops(
    Value val, SmallVectorImpl<Operation *> &affectingLoops) {
  SmallVector<Operation *, 4> loopsForVal;
  if (isDependentOnLoop(val, loopsForVal))
    affectingLoops.append(loopsForVal.begin(), loopsForVal.end());
}

void LoopAnalysis::analyzeLoopIV() {
  auto loopBfs = [&](Operation *loopOp, Value iv) {
    DenseSet<Value> visited;
    SmallVector<Value, 8> queue;
    visited.insert(iv);
    queue.push_back(iv);
    while (!queue.empty()) {
      Value cur = queue.pop_back_val();
      for (auto &use : cur.getUses()) {
        Operation *owner = use.getOwner();
        for (Value res : owner->getResults()) {
          if (visited.insert(res).second)
            queue.push_back(res);
        }
      }
    }
    for (auto &val : visited)
      loopValsMap[val].push_back(loopOp);
  };

  for (auto loopPair : loopInfoMap)
    loopBfs(loopPair.first, loopPair.second->inductionVar);
};
