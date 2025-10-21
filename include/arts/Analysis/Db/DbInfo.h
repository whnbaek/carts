///==========================================================================
/// File: DbInfo.h
/// Shared DB information object for analysis.
///==========================================================================

#ifndef ARTS_ANALYSIS_DB_DBINFO_H
#define ARTS_ANALYSIS_DB_DBINFO_H

#include "arts/ArtsDialect.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir {
namespace arts {
class DbAcquireNode;

/// DbInfo aggregates facts and hints about a datablock.
struct DbInfo {
  /// Basic facts
  bool isAlloc = false;
  unsigned inCount = 0;
  unsigned outCount = 0;

  /// Slice information
  SmallVector<Value, 4> offsets;
  SmallVector<Value, 4> sizes;
  /// Constant fallbacks: INT64_MIN if unknown
  SmallVector<int64_t, 4> constOffsets;
  SmallVector<int64_t, 4> constSizes;
  /// Estimated byte size for the described slice or allocation.
  /// 0 if unknown.
  unsigned long long estimatedBytes = 0;

  /// Get the rank of the DB.
  uint64_t getRank() const { return sizes.size(); }
};

/// Allocation-level info with computed metrics for optimization analysis
struct DbAllocInfo : public DbInfo {
  /// Program order index where this allocation occurs
  unsigned allocIndex = 0;

  /// Program order index of the last release (end of lifetime)
  /// Use: Defines the allocation's total lifetime [allocIndex, endIndex]
  unsigned endIndex = 0;

  /// Number of acquire operations on this allocation
  /// Use: High count suggests frequent access, candidate for pinning/caching
  unsigned numAcquires = 0;

  /// Number of release operations on this allocation
  /// Use: Mismatch with numAcquires indicates leaks or improper pairing
  unsigned numReleases = 0;

  /// Static size in bytes (0 if dynamic/unknown)
  /// Computed from payload sizes or memref shape at allocation time
  /// Use: Known sizes enable static memory planning and reuse analysis
  unsigned long long staticBytes = 0;

  /// List of all acquire nodes for this allocation
  /// Use: Iterate to analyze per-acquire access patterns and lifetimes
  SmallVector<DbAcquireNode *, 8> acquireNodes;

  /// Whether this allocation lives for most of the function (coverage > 70%)
  /// Meaning: Long-lived allocations span large portions of execution
  /// Use: Prefer heap over stack, or persistent allocation strategies
  ///      Short-lived allocations are better candidates for stack/pooling
  bool isLongLived = false;

  /// Whether this allocation may escape (used outside acquire/release)
  /// Meaning: Has uses beyond the tracked acquire/release pairs
  /// Use: Escaping allocations cannot be optimized as aggressively
  ///      (e.g., cannot assume exclusive ownership or elide allocation)
  bool maybeEscaping = false;

  /// Maximum loop nesting depth among all acquires
  /// Meaning: Deeper nesting = higher temporal reuse, more pressure
  /// Use: Prioritize cache optimization for high-depth allocations
  ///      Guides prefetching and tiling decisions
  unsigned maxLoopDepth = 0;

  /// Critical span: distance from earliest acquire to latest release
  /// Meaning: The bounding lifetime window (may include idle periods)
  /// Use: Bounds the memory footprint duration for capacity planning
  ///      Large span with small path indicates intermittent usage
  unsigned criticalSpan = 0;

  /// Critical path: total active-use time after merging overlapping intervals
  /// Meaning: Actual usage duration (idle periods removed)
  /// Use: path << span indicates fragmented usage, candidate for splitting
  ///      path ≈ span indicates continuous usage, keep unified
  ///      Helps decide between single persistent vs multiple ephemeral
  ///      allocations
  unsigned criticalPath = 0;

  /// Total estimated bytes accessed across all acquires
  /// Meaning: Sum of all acquire slice sizes (may double-count overlaps)
  /// Use: Estimates memory bandwidth requirements
  ///      High traffic suggests bottleneck, consider prefetching/blocking
  unsigned long long totalAccessBytes = 0;

  /// Other allocations with non-overlapping lifetimes and same size
  /// Meaning: Potential memory reuse opportunities via coloring
  /// Use: Assign same physical memory to non-overlapping allocations
  ///      Reduces peak memory footprint without correctness impact
  SmallVector<DbAllocOp, 8> reuseMatches;
};

/// Per-dimension offset range for an acquire
struct DbOffsetRange {
  Value minIndex;                  /// lower index bound when available
  Value maxIndex;                  /// upper index bound when available
  std::optional<int64_t> minConst; /// constant lower bound if known
  std::optional<int64_t> maxConst; /// constant upper bound if known
};

/// Acquire-level info with lifetime and access pattern analysis
struct DbAcquireInfo : public DbInfo {
  /// Index values used in the acquire operation
  SmallVector<Value, 4> indices;

  /// Program order index where this acquire occurs
  unsigned beginIndex = 0;

  /// Program order index where the corresponding release occurs
  /// Meaning: Defines the [beginIndex, endIndex] lifetime interval for this
  /// slice Use: Non-overlapping intervals enable parallelization
  ///      Overlaps indicate potential data races or shared access
  unsigned endIndex = 0;

  /// The acquire operation
  DbAcquireOp acquire;

  /// The corresponding release operation (if found)
  /// Use: Missing release indicates leak or permanent acquisition
  DbReleaseOp release;

  /// Per-dimension offset ranges: base + [minIndex, maxIndex]
  /// Meaning: Actual memory footprint accessed within the EDT
  /// Computed by analyzing memref.load/store patterns in the EDT body
  /// Use: Tighten acquire slices to reduce false sharing
  ///      Identify stride patterns for prefetching
  ///      Detect over-acquisition (acquired > used)
  SmallVector<DbOffsetRange, 4> offsetRanges;
};

/// Release-level info
struct DbReleaseInfo : public DbInfo {
  unsigned releaseIndex = 0;
  DbReleaseOp release;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_ANALYSIS_DB_DBINFO_H
