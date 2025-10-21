///==========================================================================
/// DbAcquireNode.cpp - Implementation of DbAcquire node
///==========================================================================

#include "arts/Analysis/Db/DbAnalysis.h"
#include "arts/Analysis/Graphs/Db/DbNode.h"
#include "arts/Analysis/LoopAnalysis.h"
#include "arts/Utils/ArtsUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::arts;
#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(db_acquire_node);

//===----------------------------------------------------------------------===//
// DbAcquireNode
//===----------------------------------------------------------------------===//
DbAcquireNode::DbAcquireNode(DbAcquireOp op, NodeBase *parent,
                             DbAllocNode *rootAlloc, DbAnalysis *analysis,
                             std::string initialHierId)
    : dbAcquireOp(op), dbReleaseOp(nullptr), op(op.getOperation()),
      parent(parent), rootAlloc(rootAlloc), analysis(analysis) {
  if (!initialHierId.empty())
    hierId = std::move(initialHierId);

  std::string hierTag = hierId.empty() ? std::string("<unassigned>") : hierId;
  ARTS_DEBUG("Creating DbAcquireNode [" << hierTag << "] for operation:\n   "
                                        << op);

  const size_t numSizes = op.getSizes().size();
  const size_t numOffsets = op.getOffsets().size();
  const size_t numIndices = op.getIndices().size();

  info.sizes.reserve(numSizes);
  info.offsets.reserve(numOffsets);
  info.indices.reserve(numIndices);
  info.constSizes.reserve(numSizes);
  info.constOffsets.reserve(numOffsets);

  /// Copy slice parameters from the operation
  info.sizes.assign(op.getSizes().begin(), op.getSizes().end());
  info.offsets.assign(op.getOffsets().begin(), op.getOffsets().end());
  info.indices.assign(op.getIndices().begin(), op.getIndices().end());

  bool hasUnknown = false;

  /// Extract constant offsets where possible
  for (Value v : info.offsets) {
    int64_t cst = INT64_MIN;
    /// INT64_MIN indicates unknown offset
    if (!arts::getConstantIndex(v, cst))
      hasUnknown = true;
    info.constOffsets.push_back(cst);
  }

  /// Extract constant sizes and compute total element count
  unsigned long long totalElems = 1;
  for (Value v : info.sizes) {
    int64_t cst = INT64_MIN;
    /// INT64_MIN indicates unknown size
    if (!arts::getConstantIndex(v, cst))
      hasUnknown = true;
    info.constSizes.push_back(cst);
    if (hasUnknown)
      continue;

    /// Check for overflow when multiplying dimensions
    if (totalElems <= std::numeric_limits<unsigned long long>::max() /
                          std::max<int64_t>(cst, 1)) {
      totalElems *= (unsigned long long)std::max<int64_t>(cst, 1);
    } else {
      totalElems = std::numeric_limits<unsigned long long>::max();
      break;
    }
  }

  /// Compute estimated bytes for this acquire slice
  if (!hasUnknown) {
    unsigned long long elemBytes = arts::getElementTypeByteSize(
        rootAlloc->getDbAllocOp().getElementType());
    if (elemBytes > 0) {
      if (totalElems >
          std::numeric_limits<unsigned long long>::max() / elemBytes) {
        info.estimatedBytes = std::numeric_limits<unsigned long long>::max();
        ARTS_WARN("Estimated bytes overflow, using max value");
      } else {
        info.estimatedBytes = totalElems * elemBytes;
      }
    }
  }

  /// Identify the EDT that consumes this acquire and the block argument used
  Value ptrResult = dbAcquireOp.getPtr();
  assert(ptrResult.hasOneUse() && "Acquire ptr should be used exactly once");

  /// The ptr result should be passed directly to an EDT or another acquire
  Operation *singleUser = *ptrResult.getUsers().begin();
  if (auto nestedAcquire = dyn_cast<DbAcquireOp>(singleUser)) {
    /// Nested acquire; create child node lazily via getOrCreateAcquireNode
    /// The rest of this constructor computes info for this node; nested
    /// children will compute their own.
  } else {
    /// Use utility function to get EDT and block argument
    auto [edt, blockArg] = arts::getEdtBlockArgumentForAcquire(dbAcquireOp);
    edtUser = edt;
    useInEdt = blockArg;
    assert(edtUser && useInEdt &&
           "Acquire ptr should be used by an EDT or another acquire");
  }

  /// Find the corresponding DbReleaseOp for this acquire
  assert(useInEdt && "Acquire ptr should be used by an EDT or another acquire");
  for (Operation *user : useInEdt.getUsers()) {
    auto releaseOp = dyn_cast<DbReleaseOp>(user);
    if (!releaseOp)
      continue;
    if (releaseOp.getSource() == useInEdt) {
      dbReleaseOp = releaseOp;
      break;
    }
  }

  /// Get the in/out mode based on the memory accesses
  if (useInEdt)
    collectAccesses(useInEdt);
  if (loads.size() > 0) {
    info.inCount = 1;
    info.outCount = 0;
  } else if (stores.size() > 0) {
    info.inCount = 0;
    info.outCount = 1;
  } else {
    info.inCount = 1;
    info.outCount = 1;
  }

  /// Compute accessed offset ranges for this acquire
  /// This analyzes the memory access patterns within the EDT
  if (useInEdt)
    computeAccessedOffsetRanges();
}

DbAcquireNode *DbAcquireNode::getOrCreateAcquireNode(DbAcquireOp op) {
  auto it = childMap.find(op);
  if (it != childMap.end())
    return it->second;
  std::string childId = getHierId().str() + "." + std::to_string(nextChildId++);
  auto newNode =
      std::make_unique<DbAcquireNode>(op, this, rootAlloc, analysis, childId);
  DbAcquireNode *ptr = newNode.get();
  childAcquires.push_back(std::move(newNode));
  childMap[op] = ptr;
  return ptr;
}

void DbAcquireNode::forEachChildNode(
    const std::function<void(NodeBase *)> &fn) const {
  for (const auto &n : childAcquires)
    fn(n.get());
}

void DbAcquireNode::print(llvm::raw_ostream &os) const {
  os << "DbAcquireNode (" << getHierId() << ")";
  os << " estimatedBytes=" << info.estimatedBytes;
  os << "\n";
}

/// Compute per-dimension accessed offset ranges for this acquire by scanning
/// memref.load/store users inside the enclosing EDT. Returns, for each
/// dimension, the acquire base offset and the best-known lower/upper index
/// Values that were used (e.g., loop bounds or constant indices).
SmallVector<DbOffsetRange, 4> DbAcquireNode::computeAccessedOffsetRanges() {
  const size_t rank = info.sizes.size();
  SmallVector<DbOffsetRange, 4> result(rank);
  info.offsetRanges.clear();
  info.offsetRanges.resize(rank);

  if (rank == 0)
    return result;

  auto *loopAnalysis = analysis->getLoopAnalysis();
  assert(loopAnalysis && "LoopAnalysis must be available");

  OpBuilder builder(op);
  builder.setInsertionPoint(op);

  SmallVector<Value, 4> minIdx(rank, Value());
  SmallVector<Value, 4> maxIdx(rank, Value());
  SmallVector<std::optional<int64_t>, 4> minConst(rank);
  SmallVector<std::optional<int64_t>, 4> maxConst(rank);
  SmallVector<SmallVector<Value, 4>, 4> observed(rank);

  auto tryConstIndex = [&](Value v) -> std::optional<int64_t> {
    if (!v)
      return std::nullopt;
    auto lb = ValueBoundsConstraintSet::computeConstantBound(
        presburger::BoundType::LB, v, std::nullopt, nullptr,
        /*closedUB=*/false);
    if (failed(lb))
      return std::nullopt;
    auto ub = ValueBoundsConstraintSet::computeConstantBound(
        presburger::BoundType::UB, v, std::nullopt, nullptr,
        /*closedUB=*/true);
    if (failed(ub) || *lb != *ub)
      return std::nullopt;
    return *lb;
  };

  llvm::DenseMap<Value, llvm::DenseMap<Value, Value>> inclusiveUpperCache;

  auto getInclusiveUpper = [&](Value ub, Value step) -> Value {
    if (!ub || !step)
      return Value();
    auto &byStep = inclusiveUpperCache[ub];
    if (auto it = byStep.find(step); it != byStep.end())
      return it->second;

    Value diff = builder.createOrFold<arith::SubIOp>(op->getLoc(), ub, step);
    byStep.try_emplace(step, diff);
    return diff;
  };

  auto updateMinConst = [&](size_t dim, int64_t value, Value src) {
    auto &slot = minConst[dim];
    if (!slot || value < *slot) {
      slot = value;
      if (src)
        minIdx[dim] = src;
    }
  };
  auto updateMaxConst = [&](size_t dim, int64_t value, Value src) {
    auto &slot = maxConst[dim];
    if (!slot || value > *slot) {
      slot = value;
      if (src)
        maxIdx[dim] = src;
    }
  };

  /// Process both loads and stores to get complete accessed index ranges.
  auto processAccesses = [&](const SmallVector<Operation *, 16> &accesses) {
    for (Operation *acc : accesses) {
      if (!edtUser.getBody().isAncestor(acc->getParentRegion()))
        continue;

      Value mem;
      ValueRange idxs;
      if (auto ld = dyn_cast<memref::LoadOp>(acc)) {
        mem = ld.getMemRef();
        idxs = ld.getIndices();
      } else if (auto ref = dyn_cast<arts::DbRefOp>(acc)) {
        mem = ref.getSource();
        idxs = ref.getIndices();
      } else if (auto st = dyn_cast<memref::StoreOp>(acc)) {
        mem = st.getMemRef();
        idxs = st.getIndices();
      } else {
        continue;
      }

      if (!mem) {
        ARTS_WARN("Encountered null memref value in access analysis");
        continue;
      }

      auto memTy = dyn_cast<MemRefType>(mem.getType());
      if (!memTy || (size_t)memTy.getRank() != rank)
        continue;

      if (idxs.size() < rank) {
        ARTS_WARN("Skipping access with insufficient indices: have "
                  << idxs.size() << ", expect " << rank);
        continue;
      }

      for (size_t d = 0; d < rank; ++d) {
        Value iv = idxs[d];
        if (!iv) {
          ARTS_WARN("Encountered null index value at dimension " << d);
          continue;
        }

        observed[d].push_back(iv);

        if (auto maybeConst = tryConstIndex(iv)) {
          updateMinConst(d, *maybeConst, iv);
          updateMaxConst(d, *maybeConst, iv);
          continue;
        }

        Value lbV, ubV;
        std::optional<int64_t> lbC, ubCMinusOne;

        /// Use LoopAnalysis to find loops affecting this index expression.
        SmallVector<Operation *, 4> affecting;
        loopAnalysis->collectAffectingLoops(iv, affecting);
        for (Operation *affectingLoop : affecting) {
          if (!affectingLoop)
            continue;
          if (auto forOp = dyn_cast<scf::ForOp>(affectingLoop)) {
            Value lbTmp = forOp.getLowerBound();
            Value ubTmp = forOp.getUpperBound();
            Value stepV = forOp.getStep();

            if (!lbV)
              lbV = lbTmp;
            if (!ubV) {
              Value inclusive = getInclusiveUpper(ubTmp, stepV);
              ubV = inclusive ? inclusive : ubTmp;
            }

            if (auto lbVal = tryConstIndex(lbTmp))
              if (!lbC || *lbVal < *lbC)
                lbC = *lbVal;

            if (auto ubVal = tryConstIndex(ubTmp)) {
              int64_t stepVal = 1;
              if (auto stepConst = tryConstIndex(stepV))
                stepVal = *stepConst > 0 ? *stepConst : 1;
              int64_t closed = *ubVal - stepVal;
              if (!ubCMinusOne || closed > *ubCMinusOne)
                ubCMinusOne = closed;
            }
          }
        }

        if (lbC)
          updateMinConst(d, *lbC, lbV ? lbV : iv);
        else if (lbV && !minIdx[d])
          minIdx[d] = lbV;

        if (ubCMinusOne)
          updateMaxConst(d, *ubCMinusOne, ubV);
        else if (ubV && !maxIdx[d])
          maxIdx[d] = ubV;
      }
    }
  };

  /// Process both loads and stores to get complete accessed ranges
  processAccesses(loads);
  processAccesses(stores);

  /// Refine bounds with value-bounds analysis when possible.
  auto refineWithValueBounds = [&](size_t dim, presburger::BoundType kind,
                                   bool closed, std::optional<int64_t> &slot,
                                   Value &slotValue, bool isMin) {
    for (Value candidate : observed[dim]) {
      if (!candidate)
        continue;
      auto bound = ValueBoundsConstraintSet::computeConstantBound(
          kind, candidate, std::nullopt, nullptr, closed);
      if (failed(bound))
        continue;
      int64_t boundVal = *bound;
      if (!slot || (isMin ? boundVal < *slot : boundVal > *slot)) {
        slot = boundVal;
        if (!slotValue)
          slotValue = candidate;
      }
    }
  };

  for (size_t d = 0; d < rank; ++d) {
    if (!minConst[d])
      refineWithValueBounds(d, presburger::BoundType::LB, /*closed=*/false,
                            minConst[d], minIdx[d], /*isMin=*/true);
    if (!maxConst[d])
      refineWithValueBounds(d, presburger::BoundType::UB, /*closed=*/true,
                            maxConst[d], maxIdx[d], /*isMin=*/false);
  }

  /// Materialize constant indices if needed and populate result + info
  for (size_t d = 0; d < rank; ++d) {
    DbOffsetRange &range = result[d];

    if (!minIdx[d] && minConst[d])
      minIdx[d] =
          builder.create<arith::ConstantIndexOp>(op->getLoc(), *minConst[d]);
    if (!maxIdx[d] && maxConst[d])
      maxIdx[d] =
          builder.create<arith::ConstantIndexOp>(op->getLoc(), *maxConst[d]);

    if (!minIdx[d] && !observed[d].empty())
      minIdx[d] = observed[d].front();
    if (!maxIdx[d] && !observed[d].empty())
      maxIdx[d] = observed[d].front();

    range.minIndex = minIdx[d];
    range.maxIndex = maxIdx[d];
    range.minConst = minConst[d];
    range.maxConst = maxConst[d];

    info.offsetRanges[d] = range;
  }

  ARTS_DEBUG_REGION({
    auto stringifyValue = [](Value v) -> std::string {
      if (!v)
        return "?";
      std::string buffer;
      llvm::raw_string_ostream os(buffer);
      v.print(os);
      return os.str();
    };

    ARTS_DEBUG("Computed accessed offset ranges for " << getHierId());
    for (size_t d = 0; d < rank; ++d) {
      std::string minStr = minConst[d] ? std::to_string(*minConst[d])
                                       : stringifyValue(minIdx[d]);
      std::string maxStr = maxConst[d] ? std::to_string(*maxConst[d])
                                       : stringifyValue(maxIdx[d]);
      ARTS_DEBUG("  dim " << d << " => [" << minStr << ", " << maxStr << "]");
    }
  });

  return result;
}

/// Compute a maximal prefix of invariant indices (one per leading dimension
/// of the acquired memref) that are uniform across the EDT. These indices
/// can be used to coarsen the acquire by pinning leading dimensions.
/// The returned Values are SSA values already present in the IR; this API
/// does not materialize new constants or perform replacements.
SmallVector<Value, 4> DbAcquireNode::computeInvariantIndices() {
  ARTS_DEBUG("Starting invariant indices computation for " << dbAcquireOp);

  SmallVector<Value, 4> result;
  const size_t rank = info.sizes.size();

  SmallVector<Operation *> memoryAccesses;
  getMemoryAccesses(memoryAccesses, true, true);
  ARTS_DEBUG("Found " << memoryAccesses.size() << " memory accesses");

  SmallVector<Value, 4> candidate(rank, Value());
  SmallVector<bool, 4> invalid(rank, false);

  auto &edtRegion = edtUser.getBody();
  for (Operation *acc : memoryAccesses) {
    ValueRange idxs;
    if (auto ld = dyn_cast<memref::LoadOp>(acc))
      idxs = ld.getIndices();
    else if (auto dbLd = dyn_cast<DbRefOp>(acc))
      idxs = dbLd.getIndices();
    else if (auto st = dyn_cast<memref::StoreOp>(acc))
      idxs = st.getIndices();
    else
      continue;

    if (!edtRegion.isAncestor(acc->getParentRegion()))
      continue;

    auto minRank = std::min<size_t>(rank, idxs.size());
    for (size_t d = 0; d < minRank; ++d) {
      if (invalid[d])
        continue;
      Value idxV = idxs[d];
      if (!candidate[d])
        candidate[d] = idxV;
      else if (candidate[d] != idxV)
        invalid[d] = true;
    }
  }

  for (size_t d = 0; d < rank; ++d) {
    if (!candidate[d] || invalid[d])
      break;
    if (!arts::isInvariantInEdt(edtRegion, candidate[d]))
      break;
    result.push_back(candidate[d]);
  }

  ARTS_DEBUG_REGION({
    ARTS_DEBUG(" - Invariant indices:");
    for (size_t d = 0; d < result.size(); ++d) {
      ARTS_DEBUG("   - Dimension " << d << ": " << result[d]);
    }
  });
  return result;
}

/// Collect all memory accesses (loads/stores) that transitively use a value.
void DbAcquireNode::collectAccesses(Value db) {
  SmallVector<Value, 8> worklist{db};
  DenseSet<Value> visited;
  visited.reserve(16);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;

    for (Operation *user : v.getUsers()) {
      /// Follow through casts
      if (auto castOp = dyn_cast<memref::CastOp>(user)) {
        worklist.push_back(castOp.getResult());
        continue;
      }

      /// Track memref and LLVM stores
      if (isa<memref::StoreOp, LLVM::StoreOp>(user)) {
        stores.push_back(user);
        continue;
      }

      /// Track memref loads
      if (auto loadOp = dyn_cast<memref::LoadOp>(user)) {
        loads.push_back(user);
        Type resultTy = loadOp.getResult().getType();
        if (isa<MemRefType>(resultTy))
          worklist.push_back(loadOp.getResult());
        continue;
      }

      if (auto dbRefOp = dyn_cast<DbRefOp>(user)) {
        references.push_back(user);
        Type resultTy = dbRefOp.getResult().getType();
        if (isa<MemRefType>(resultTy))
          worklist.push_back(dbRefOp.getResult());
        continue;
      }

      /// Track LLVM loads
      if (auto loadOp = dyn_cast<LLVM::LoadOp>(user)) {
        loads.push_back(user);
        continue;
      }
    }
  }
}
