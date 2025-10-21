///==========================================================================
/// File: DbAliasAnalysis.h
/// Alias analysis helpers for DB nodes.
///==========================================================================

#ifndef ARTS_ANALYSIS_DB_DBALIASANALYSIS_H
#define ARTS_ANALYSIS_DB_DBALIASANALYSIS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace arts {
class DbAnalysis;
class NodeBase;
class DbAcquireNode;
class DbAllocNode;
class DbAcquireOp;

class DbAliasAnalysis {
public:
  explicit DbAliasAnalysis(DbAnalysis *analysis);

  enum class AliasResult { NoAlias, MayAlias, MustAlias };

  AliasResult classifyAlias(const NodeBase &a, const NodeBase &b,
                            const std::string &indent = "");

  bool mayAlias(const NodeBase &a, const NodeBase &b,
                const std::string &indent = "") {
    return classifyAlias(a, b, indent) != AliasResult::NoAlias;
  }
  bool mustAlias(const NodeBase &a, const NodeBase &b,
                 const std::string &indent = "") {
    return classifyAlias(a, b, indent) == AliasResult::MustAlias;
  }

  /// Overlap classification between DB acquire slices
  enum class OverlapKind { Unknown, Disjoint, Partial, Full };
  OverlapKind estimateOverlap(const DbAcquireNode *a, const DbAcquireNode *b);

  Value getUnderlyingValue(const NodeBase &node);

private:
  DbAnalysis *analysis;
  DenseMap<std::pair<Value, Value>, AliasResult> aliasCache;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_ANALYSIS_DB_DBALIASANALYSIS_H
