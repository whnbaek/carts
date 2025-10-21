///==========================================================================
/// File: DbAliasAnalysis.cpp
/// Defines alias analysis for DB operations and memory tracking.
///==========================================================================

#include "arts/Analysis/Db/DbAliasAnalysis.h"
#include "arts/Analysis/Db/DbAnalysis.h"
#include "arts/Analysis/Graphs/Db/DbGraph.h"
#include "arts/Analysis/Graphs/Db/DbNode.h"
#include "arts/Utils/ArtsUtils.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(db_alias);

#include <algorithm>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::arts;

namespace {
std::pair<Value, Value> makeOrderedPair(Value a, Value b) {
  /// Use pointer comparison for ordering
  return a.getAsOpaquePointer() < b.getAsOpaquePointer() ? std::make_pair(a, b)
                                                         : std::make_pair(b, a);
}

struct SliceInterval {
  std::optional<int64_t> min;
  std::optional<int64_t> max;
};

std::optional<int64_t> tryGetConst(ArrayRef<int64_t> vec, size_t idx) {
  if (idx >= vec.size())
    return std::nullopt;
  int64_t val = vec[idx];
  if (val == INT64_MIN)
    return std::nullopt;
  return val;
}

std::optional<int64_t> tryComputeConstantBound(presburger::BoundType type,
                                               Value v, bool closedUB = false) {
  if (!v)
    return std::nullopt;
  auto bound = ValueBoundsConstraintSet::computeConstantBound(
      type, v, std::nullopt, nullptr, closedUB);
  if (failed(bound))
    return std::nullopt;
  return *bound;
}

std::optional<int64_t> tryComputeConstantBound(presburger::BoundType type,
                                               AffineMap map,
                                               ValueDimList operands,
                                               bool closedUB = false) {
  auto bound = ValueBoundsConstraintSet::computeConstantBound(
      type, map, std::move(operands), nullptr, closedUB);
  if (failed(bound))
    return std::nullopt;
  return *bound;
}

SliceInterval computeSliceInterval(const DbAcquireNode *node, size_t dim) {
  SliceInterval interval;
  if (!node)
    return interval;

  const DbAcquireInfo &info = node->getInfo();

  /// Minimum offset (start of the slice)
  if (auto cst = tryGetConst(info.constOffsets, dim))
    interval.min = cst;
  else if (dim < info.offsets.size())
    interval.min =
        tryComputeConstantBound(presburger::BoundType::LB, info.offsets[dim]);

  /// Maximum offset + extent
  std::optional<int64_t> offsetMax;
  if (auto cst = tryGetConst(info.constOffsets, dim))
    offsetMax = cst;
  else if (dim < info.offsets.size())
    offsetMax = tryComputeConstantBound(presburger::BoundType::UB,
                                        info.offsets[dim], /*closedUB=*/true);

  std::optional<int64_t> sizeMax;
  if (auto cst = tryGetConst(info.constSizes, dim))
    sizeMax = cst;
  else if (dim < info.sizes.size())
    sizeMax =
        tryComputeConstantBound(presburger::BoundType::UB, info.sizes[dim],
                                /*closedUB=*/true);

  if (!offsetMax && sizeMax && dim < info.offsets.size()) {
    /// Attempt to infer offset upper bound jointly with size if the
    /// individual upper bound was unavailable.
    MLIRContext *ctx = node->getOp()->getContext();
    AffineExpr d0 = getAffineDimExpr(0, ctx);
    AffineExpr d1 = getAffineDimExpr(1, ctx);
    AffineExpr expr = d0 + d1 - getAffineConstantExpr(1, ctx);
    AffineMap map = AffineMap::get(2, 0, expr);
    ValueDimList operands;
    operands.emplace_back(info.offsets[dim], std::nullopt);
    operands.emplace_back(info.sizes[dim], std::nullopt);
    if (auto bound =
            tryComputeConstantBound(presburger::BoundType::UB, map,
                                    std::move(operands), /*closedUB=*/true)) {
      offsetMax =
          *bound - std::max<int64_t>(sizeMax.value_or(1), int64_t(1)) + 1;
    }
  }

  if (offsetMax && sizeMax) {
    int64_t width = std::max<int64_t>(*sizeMax, int64_t(1));
    interval.max = *offsetMax + width - 1;
  } else if (offsetMax) {
    interval.max = offsetMax;
  }

  return interval;
}

std::optional<bool> checkDisjoint(const SliceInterval &lhs,
                                  const SliceInterval &rhs) {
  if (lhs.max && rhs.min && *lhs.max < *rhs.min)
    return true;
  if (rhs.max && lhs.min && *rhs.max < *lhs.min)
    return true;
  if (lhs.min && lhs.max && rhs.min && rhs.max) {
    bool overlap = !(*lhs.max < *rhs.min || *rhs.max < *lhs.min);
    return overlap ? std::optional<bool>(false) : std::optional<bool>(true);
  }
  return std::nullopt;
}

bool slicesProvablyDisjoint(const DbAcquireNode *lhs, const DbAcquireNode *rhs,
                            const std::string &indent) {
  if (!lhs || !rhs)
    return false;

  const auto &lhsInfo = lhs->getInfo();
  const auto &rhsInfo = rhs->getInfo();
  size_t dims = std::min(lhsInfo.sizes.size(), rhsInfo.sizes.size());

  for (size_t d = 0; d < dims; ++d) {
    SliceInterval lhsInterval = computeSliceInterval(lhs, d);
    SliceInterval rhsInterval = computeSliceInterval(rhs, d);
    if (auto disjoint = checkDisjoint(lhsInterval, rhsInterval)) {
      if (*disjoint) {
        ARTS_INFO("      Dim[" << d << "]: symbolic range separation detected"
                               << indent);
        return true;
      }
    }
  }
  return false;
}
} // namespace

DbAliasAnalysis::DbAliasAnalysis(DbAnalysis *analysis) : analysis(analysis) {
  assert(analysis && "Analysis cannot be null");
}

DbAliasAnalysis::OverlapKind
DbAliasAnalysis::estimateOverlap(const DbAcquireNode *a,
                                 const DbAcquireNode *b) {
  if (!a || !b)
    return OverlapKind::Unknown;

  const auto &ai = a->getInfo();
  const auto &bi = b->getInfo();

  size_t rank = std::min(ai.constSizes.size(), bi.constSizes.size());
  if (rank == 0)
    return OverlapKind::Unknown;

  bool equal =
      (ai.constSizes == bi.constSizes) && (ai.constOffsets == bi.constOffsets);
  bool anyUnknown = false;
  bool disjoint = false;

  for (size_t i = 0; i < rank; ++i) {
    int64_t ao = i < ai.constOffsets.size() ? ai.constOffsets[i] : INT64_MIN;
    int64_t asz = i < ai.constSizes.size() ? ai.constSizes[i] : INT64_MIN;
    int64_t bo = i < bi.constOffsets.size() ? bi.constOffsets[i] : INT64_MIN;
    int64_t bsz = i < bi.constSizes.size() ? bi.constSizes[i] : INT64_MIN;
    if (ao == INT64_MIN || asz == INT64_MIN || bo == INT64_MIN ||
        bsz == INT64_MIN) {
      anyUnknown = true;
      continue;
    }
    int64_t a0 = ao, a1 = ao + asz;
    int64_t b0 = bo, b1 = bo + bsz;
    bool overlap = !(a1 <= b0 || b1 <= a0);
    if (!overlap) {
      disjoint = true;
      break;
    }
  }

  if (disjoint)
    return OverlapKind::Disjoint;
  if (equal)
    return OverlapKind::Full;
  if (anyUnknown)
    return OverlapKind::Unknown;
  return OverlapKind::Partial;
}

static const char *aliasResultToString(DbAliasAnalysis::AliasResult res) {
  switch (res) {
  case DbAliasAnalysis::AliasResult::NoAlias:
    return "NO ALIAS";
  case DbAliasAnalysis::AliasResult::MustAlias:
    return "MUST ALIAS";
  case DbAliasAnalysis::AliasResult::MayAlias:
    return "MAY ALIAS";
  }
  llvm_unreachable("Unhandled alias result");
}

DbAliasAnalysis::AliasResult
DbAliasAnalysis::classifyAlias(const NodeBase &a, const NodeBase &b,
                               const std::string &indent) {
  Value ptrA = getUnderlyingValue(a);
  Value ptrB = getUnderlyingValue(b);

  ARTS_INFO("Analyzing alias between: " << a.getHierId() << " -- "
                                        << b.getHierId());

  /// Check cache
  auto key = makeOrderedPair(ptrA, ptrB);
  auto it = aliasCache.find(key);
  if (it != aliasCache.end()) {
    ARTS_INFO("  Result: " << aliasResultToString(it->second) << " (cache hit)"
                           << indent);
    return it->second;
  }

  /// Same value aliases with itself
  if (ptrA == ptrB) {
    aliasCache[key] = AliasResult::MustAlias;
    ARTS_INFO("  Result: MUST ALIAS (same pointer)" << indent);
    return AliasResult::MustAlias;
  }

  /// Start with conservative assumption (may alias)
  AliasResult result = AliasResult::MayAlias;
  ARTS_INFO("  Step 1: Checking slice information" << indent);

  /// Helper to extract slice information from DbInfo (preferred) or op directly
  auto sliceOf = [&](const NodeBase &n)
      -> std::tuple<Value, SmallVector<int64_t>, SmallVector<int64_t>> {
    SmallVector<int64_t> offs, lens;
    Value root;

    if (auto *acq = dyn_cast<DbAcquireNode>(&n)) {
      auto op = cast<DbAcquireOp>(acq->getOp());
      root = op.getSourcePtr();

      /// Use precomputed DbInfo from acquire node
      const auto &info = acq->getInfo();
      ARTS_INFO("    Using DbInfo for acquire "
                << acq->getHierId() << " (rank=" << info.constOffsets.size()
                << ")" << indent);
      offs.assign(info.constOffsets.begin(), info.constOffsets.end());
      lens.assign(info.constSizes.begin(), info.constSizes.end());
    } else if (auto *alloc = dyn_cast<DbAllocNode>(&n)) {
      root = alloc->getDbAllocOp().getPtr();

      /// Use precomputed DbInfo from alloc node
      const auto &info = alloc->getInfo();
      ARTS_INFO("    Using DbInfo for alloc "
                << alloc->getHierId() << " (rank=" << info.constOffsets.size()
                << ")" << indent);
      offs.assign(info.constOffsets.begin(), info.constOffsets.end());
      lens.assign(info.constSizes.begin(), info.constSizes.end());
    }
    return {root, offs, lens};
  };

  auto [ra, oa, la] = sliceOf(a);
  auto [rb, ob, lb] = sliceOf(b);

  /// Different roots definitely don't alias
  if (ra && rb && ra != rb) {
    ARTS_INFO("  Step 1 Result: NO ALIAS (different roots)" << indent);
    result = AliasResult::NoAlias;
  }
  /// Same root: check for slice disjointness
  else if (ra && rb && ra == rb) {
    ARTS_INFO("  Step 1: Same root, checking slice overlap" << indent);
    ARTS_INFO("    A: rank=" << oa.size() << indent);
    ARTS_INFO("    B: rank=" << ob.size() << indent);

    /// If both slices are constant and disjoint in any dim -> no alias
    bool anyUnknown = false;
    bool equalSlices = true;
    if (oa.size() == ob.size() && la.size() == lb.size() &&
        oa.size() == la.size()) {
      bool disjoint = false;
      for (size_t i = 0; i < oa.size(); ++i) {
        if (oa[i] == INT64_MIN || ob[i] == INT64_MIN || la[i] == INT64_MIN ||
            lb[i] == INT64_MIN) {
          ARTS_INFO("      Dim[" << i << "]: unknown" << indent);
          anyUnknown = true;
          equalSlices = false;
          continue;
        }
        int64_t a0 = oa[i];
        int64_t a1 = oa[i] + la[i];
        int64_t b0 = ob[i];
        int64_t b1 = ob[i] + lb[i];
        bool overlap = !(a1 <= b0 || b1 <= a0);
        ARTS_INFO("      Dim[" << i << "]: A=[" << a0 << "," << a1 << ") B=["
                               << b0 << "," << b1 << ") "
                               << (overlap ? "OVERLAP" : "DISJOINT") << indent);
        if (a0 != b0 || la[i] != lb[i])
          equalSlices = false;
        if (!overlap) {
          disjoint = true;
          break;
        }
      }
      if (!anyUnknown && disjoint) {
        ARTS_INFO("  Step 1 Result: NO ALIAS (disjoint slices)" << indent);
        result = AliasResult::NoAlias;
      } else if (!anyUnknown && equalSlices) {
        ARTS_INFO("  Step 1 Result: MUST ALIAS (identical slices)" << indent);
        result = AliasResult::MustAlias;
      } else if (anyUnknown) {
        ARTS_INFO("  Step 1 Result: UNKNOWN (incomplete slice info)" << indent);
      } else {
        ARTS_INFO("  Step 1 Result: MAY ALIAS (overlapping slices)" << indent);
      }
    } else {
      ARTS_INFO("  Step 1: Cannot compare (mismatched ranks)" << indent);
    }

    if (result == AliasResult::MayAlias) {
      if (auto *acqA = dyn_cast<DbAcquireNode>(&a)) {
        if (auto *acqB = dyn_cast<DbAcquireNode>(&b)) {
          if (slicesProvablyDisjoint(acqA, acqB, indent)) {
            ARTS_INFO("  Step 1 Result: NO ALIAS (symbolic slice separation)"
                      << indent);
            result = AliasResult::NoAlias;
          }
        }
      }
    }
  }

  /// If still unknown, use DbAnalysis overlap estimator for acquires
  if (result == AliasResult::MayAlias) {
    /// Try to refine using per-dimension accessed offset ranges if available
    if (auto *acqA = dyn_cast<DbAcquireNode>(&a)) {
      if (auto *acqB = dyn_cast<DbAcquireNode>(&b)) {
        const auto &ia = acqA->getInfo();
        const auto &ib = acqB->getInfo();
        if (!ia.offsetRanges.empty() &&
            ia.offsetRanges.size() == ib.offsetRanges.size()) {
          bool allConst = true;
          bool disjoint = false;
          bool identical = true;
          for (size_t d = 0; d < ia.offsetRanges.size(); ++d) {
            auto &ar = ia.offsetRanges[d];
            auto &br = ib.offsetRanges[d];

            std::optional<int64_t> amin = ar.minConst;
            std::optional<int64_t> amax = ar.maxConst;
            std::optional<int64_t> bmin = br.minConst;
            std::optional<int64_t> bmax = br.maxConst;

            int64_t tmp;
            if (!amin && ar.minIndex)
              if (arts::getConstantIndex(ar.minIndex, tmp))
                amin = tmp;
            if (!amax && ar.maxIndex)
              if (arts::getConstantIndex(ar.maxIndex, tmp))
                amax = tmp;
            if (!bmin && br.minIndex)
              if (arts::getConstantIndex(br.minIndex, tmp))
                bmin = tmp;
            if (!bmax && br.maxIndex)
              if (arts::getConstantIndex(br.maxIndex, tmp))
                bmax = tmp;

            if (!(amin && amax && bmin && bmax)) {
              allConst = false;
              break;
            }

            if (*amax < *bmin || *bmax < *amin) {
              disjoint = true;
              break;
            }
            if (*amin != *bmin || *amax != *bmax)
              identical = false;
          }
          if (allConst && disjoint) {
            ARTS_INFO("  Step 2 Result: NO ALIAS (disjoint accessed ranges)"
                      << indent);
            result = AliasResult::NoAlias;
          } else if (allConst && identical) {
            ARTS_INFO("  Step 2 Result: MUST ALIAS (identical ranges)"
                      << indent);
            result = AliasResult::MustAlias;
          }
        }
      }
    }
  }

  /// If still unknown, use our overlap estimator for acquires
  if (result == AliasResult::MayAlias) {
    const auto *acqA = dyn_cast<DbAcquireNode>(&a);
    const auto *acqB = dyn_cast<DbAcquireNode>(&b);
    if (acqA && acqB) {
      ARTS_INFO("  Step 2: Using DbAliasAnalysis::estimateOverlap" << indent);
      auto k = estimateOverlap(acqA, acqB);
      const char *kindStr = k == OverlapKind::Disjoint  ? "DISJOINT"
                            : k == OverlapKind::Partial ? "PARTIAL"
                            : k == OverlapKind::Full    ? "FULL"
                                                        : "UNKNOWN";
      ARTS_INFO("  Step 2 Result: " << kindStr << indent);

      if (k == OverlapKind::Disjoint)
        result = AliasResult::NoAlias;
      else if (k == OverlapKind::Full)
        result = AliasResult::MustAlias;
    }
  }

  /// Cache the result
  aliasCache[key] = result;
  ARTS_INFO("  => FINAL RESULT: " << aliasResultToString(result) << " (cached)"
                                  << indent);
  return result;
}

Value DbAliasAnalysis::getUnderlyingValue(const NodeBase &node) {
  Operation *op = node.getOp();
  if (isa<DbAllocOp>(op)) {
    return op->getResult(0);
  } else if (isa<DbAcquireOp>(op)) {
    Value sourcePtr = cast<DbAcquireOp>(op).getSourcePtr();
    return arts::getUnderlyingValue(sourcePtr);
  } else if (isa<DbReleaseOp>(op)) {
    Value source = cast<DbReleaseOp>(op).getSource();
    return arts::getUnderlyingValue(source);
  }
  llvm_unreachable("Invalid DB node type");
}
