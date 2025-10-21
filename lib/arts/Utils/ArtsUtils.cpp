///==========================================================================
/// File: ArtsUtils.cpp
/// Defines utility functions for the ARTS dialect.
///==========================================================================
#include "arts/Utils/ArtsUtils.h"
#include "arts/ArtsDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/CSE.h"
#include "polygeist/Ops.h"
#include <algorithm>
#include <cassert>

/// Debug
#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(arts_utils);

namespace mlir {
namespace arts {

//===----------------------------------------------------------------------===//
/// IR Simplification Utilities
//===----------------------------------------------------------------------===//

/// Simplifies the IR by running common subexpression elimination (CSE)
/// across the module using the provided dominance information.
bool simplifyIR(ModuleOp module, DominanceInfo &domInfo) {
  IRRewriter rewriter(module.getContext());
  bool changed = false;
  eliminateCommonSubExpressions(rewriter, domInfo, module, &changed);

  return changed;
}

//===----------------------------------------------------------------------===//
/// Value Analysis Utilities
//===----------------------------------------------------------------------===//

/// Checks if the given value is a constant, including constant-like operations
/// such as constant indices and constant operations.
bool isValueConstant(Value val) {
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return false;

  if (defOp->hasTrait<OpTrait::ConstantLike>())
    return true;

  if (isa<arith::ConstantIndexOp>(defOp) || isa<arith::ConstantOp>(defOp))
    return true;
  return false;
};

/// Attempts to extract a constant index value from the given value, supporting
/// both constant index operations and constant operations with integer
/// attributes.
bool getConstantIndex(Value v, int64_t &out) {
  if (!v)
    return false;
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) {
    out = c.value();
    return true;
  }
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(c.getValue())) {
      out = intAttr.getInt();
      return true;
    }
  }
  return false;
}

/// Determines if the given value represents a non-zero index, returning true
/// for non-zero constants or unknown (non-constant) values.
bool isNonZeroIndex(Value v) {
  if (!v)
    return false;
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return c.value() != 0;
  return true;
}

//===----------------------------------------------------------------------===//
/// Type and Size Utilities
//===----------------------------------------------------------------------===//

/// Computes the byte size of the given element type, supporting integer and
/// floating-point types. When the type is a memref, all static dimensions must
/// be known; otherwise, the size is treated as unknown (returns 0).
uint64_t getElementTypeByteSize(Type elemTy) {
  if (auto memTy = dyn_cast<MemRefType>(elemTy)) {
    uint64_t elementBytes = getElementTypeByteSize(memTy.getElementType());
    if (elementBytes == 0)
      return 0;

    uint64_t total = elementBytes;
    for (int64_t dim : memTy.getShape()) {
      if (dim == ShapedType::kDynamic)
        return 0;
      total *= static_cast<uint64_t>(std::max<int64_t>(dim, 0));
    }
    return total;
  }
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    return intTy.getWidth() / 8u;
  if (auto fTy = dyn_cast<FloatType>(elemTy))
    return fTy.getWidth() / 8u;
  /// Unknown type
  return 0;
}

/// Gets the element memref type for a given element type and sizes.
MemRefType getElementMemRefType(Type elementType,
                                ArrayRef<Value> elementSizes) {
  SmallVector<int64_t> elementShape(elementSizes.size(), ShapedType::kDynamic);
  return MemRefType::get(elementShape, elementType);
}

//===----------------------------------------------------------------------===//
/// String Utilities
//===----------------------------------------------------------------------===//

/// Sanitizes the given string by replacing dots and dashes with underscores,
/// making it suitable for use as an identifier.
std::string sanitizeString(StringRef s) {
  std::string id = s.str();
  std::replace(id.begin(), id.end(), '.', '_');
  std::replace(id.begin(), id.end(), '-', '_');
  return id;
}

//===----------------------------------------------------------------------===//
/// Range and Value Comparison Utilities
//===----------------------------------------------------------------------===//

/// Compares two ValueRanges for equality, checking both size and element-wise
/// equivalence.
bool equalRange(ValueRange a, ValueRange b) {
  if (a.size() != b.size())
    return false;
  for (auto it = a.begin(), jt = b.begin(); it != a.end(); ++it, ++jt) {
    if (*it != *jt)
      return false;
  }
  return true;
}

/// Checks if all values in the given range are identical, returning false for
/// empty ranges.
bool allSameValue(ValueRange values) {
  if (values.empty())
    return false;
  Value first = values[0];
  return llvm::all_of(values, [&](Value v) { return v == first; });
}

/// Check if two values represent equivalent scaling factors.
/// Used to recognize patterns like (N * sizeof(T)) / sizeof(T) -> N.
bool scalesAreEquivalent(Value lhs, Value rhs) {
  Value a = stripNumericCasts(lhs);
  Value b = stripNumericCasts(rhs);
  if (a == b)
    return true;

  auto constantValue = [](Value v) -> std::optional<int64_t> {
    if (auto cIdx = v.getDefiningOp<arith::ConstantIndexOp>())
      return cIdx.value();
    if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>())
      return cInt.value();
    return std::nullopt;
  };

  if (auto lhsConst = constantValue(a))
    if (auto rhsConst = constantValue(b))
      return lhsConst == rhsConst;

  if (auto lhsType = a.getDefiningOp<polygeist::TypeSizeOp>())
    if (auto rhsType = b.getDefiningOp<polygeist::TypeSizeOp>())
      return lhsType.getType() == rhsType.getType();

  return false;
}

//===----------------------------------------------------------------------===//
/// EDT Analysis Utilities
//===----------------------------------------------------------------------===//

/// Determines if a value is invariant within the given EDT region, meaning it
/// is defined outside the region or not modified within it.
bool isInvariantInEdt(Region &edtRegion, Value value) {
  SmallPtrSet<Value, 16> visited;

  std::function<bool(Value)> definedInvariant = [&](Value v) -> bool {
    if (!v)
      return false;
    if (!visited.insert(v).second)
      return true;

    if (isValueConstant(v))
      return true;

    if (auto blockArg = v.dyn_cast<BlockArgument>()) {
      Block *owner = blockArg.getOwner();
      if (!owner)
        return false;
      Region *ownerRegion = owner->getParent();
      if (!ownerRegion)
        return false;

      /// Entry block arguments of the EDT region are considered invariant
      /// because their defining value is outside the region.
      if (ownerRegion == &edtRegion && owner == &edtRegion.front())
        return true;

      /// Block arguments that belong to regions outside of this EDT are also
      /// treated as invariant inputs.
      if (!edtRegion.isAncestor(ownerRegion))
        return true;

      return false;
    }

    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      return false;

    if (!edtRegion.isAncestor(defOp->getParentRegion()))
      return true;

    if (isa<arith::ConstantIndexOp, arith::ConstantIntOp>(defOp))
      return true;

    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
            arith::DivUIOp, arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp,
            arith::TruncIOp>(defOp)) {
      for (Value operand : defOp->getOperands())
        if (!definedInvariant(operand))
          return false;
      return true;
    }

    return false;
  };

  if (!definedInvariant(value))
    return false;

  auto isPointerLike = [](Type type) {
    return type.isa<MemRefType, UnrankedMemRefType>();
  };

  if (!isPointerLike(value.getType()))
    return true;

  for (Operation *user : value.getUsers()) {
    if (!edtRegion.isAncestor(user->getParentRegion()))
      continue;

    if (auto store = dyn_cast<memref::StoreOp>(user))
      if (store.getMemref() == value)
        return false;

    if (auto atomicRmw = dyn_cast<memref::AtomicRMWOp>(user))
      if (atomicRmw.getMemref() == value)
        return false;
  }

  return true;
}

/// Checks if the target operation is reachable from the source operation in
/// the EDT control flow graph.
bool isReachable(Operation *source, Operation *target) {
  /// Early exit if either pointer is null or both are the same.
  if (!source || !target)
    return false;
  if (source == target)
    return true;

  /// If both operations are in the same block, check their order.
  if (source->getBlock() == target->getBlock())
    return source->isBeforeInBlock(target);

  /// Get the EDT parent for both operations. This acts as our upper limit.
  auto srcEdt = source->getParentOfType<EdtOp>();
  auto tgtEdt = target->getParentOfType<EdtOp>();
  if (srcEdt != tgtEdt)
    return false;

  /// Traverse up the parent chain simultaneously but stop once the EDT parent
  /// is reached.
  Operation *src = source;
  Operation *tgt = target;
  while (true) {
    if (src->getBlock() == tgt->getBlock())
      return src->isBeforeInBlock(tgt);
    if (src == srcEdt && tgt == tgtEdt)
      break;
    if (src->getParentOp() != srcEdt)
      src = src->getParentOp();
    if (tgt->getParentOp() != tgtEdt)
      tgt = tgt->getParentOp();
  }
  return false;
}

/// Finds the EdtOp that uses a DbAcquireOp and returns the corresponding block
/// argument.
std::pair<EdtOp, BlockArgument>
getEdtBlockArgumentForAcquire(DbAcquireOp acquireOp) {
  /// Find the EDT that uses this acquire's pointer result
  EdtOp edtUser = nullptr;
  unsigned argIdx = 0;

  for (auto &use : acquireOp->getUses()) {
    Operation *userOp = use.getOwner();
    if (auto edtOp = dyn_cast<EdtOp>(userOp)) {
      edtUser = edtOp;
      argIdx = use.getOperandNumber();
      break;
    }
  }

  if (!edtUser)
    return {nullptr, nullptr};

  /// EDT operands are segmented: route, deps, ...
  /// Index 1 is the deps segment (block arguments)
  auto [depsBegin, depsLen] = edtUser.getODSOperandIndexAndLength(1);

  /// Check if the operand is within deps range
  if (argIdx < depsBegin || argIdx >= depsBegin + depsLen)
    return {nullptr, nullptr};

  /// Convert operand index to block argument index
  argIdx = argIdx - depsBegin;

  /// Get the block argument
  Block &body = edtUser.getRegion().front();
  if (argIdx >= body.getNumArguments())
    return {nullptr, nullptr};

  BlockArgument blockArg = body.getArgument(argIdx);
  return {edtUser, blockArg};
}

/// Internal helper to trace the underlying value through various operations,
/// with cycle detection to avoid infinite recursion.
static Value getUnderlyingValueImpl(Value v, SmallPtrSet<Value, 8> &visited) {
  if (!v)
    return nullptr;

  /// Check for cycles
  if (!visited.insert(v).second)
    return nullptr;

  if (v.isa<BlockArgument>()) {
    Block *block = v.getParentBlock();
    Operation *owner = block->getParentOp();
    if (auto edt = dyn_cast<EdtOp>(owner)) {
      unsigned argIndex = v.cast<BlockArgument>().getArgNumber();
      /// Block arguments correspond directly to EDT operands
      if (argIndex < edt.getNumOperands()) {
        Value operand = edt.getOperand(argIndex);
        return getUnderlyingValueImpl(operand, visited);
      } else {
        return v;
      }
    } else {
      /// Function argument
      return v;
    }
  } else if (auto op = v.getDefiningOp()) {
    /// Handle different operation types
    if (isa<DbAllocOp, memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp>(
            op))
      return v;
    else if (auto dbAcquire = dyn_cast<DbAcquireOp>(op))
      return getUnderlyingValueImpl(dbAcquire.getSourcePtr(), visited);
    else if (auto dbGep = dyn_cast<DbGepOp>(op))
      return getUnderlyingValueImpl(dbGep.getBasePtr(), visited);
    else if (auto subview = dyn_cast<memref::SubViewOp>(op))
      return getUnderlyingValueImpl(subview.getSource(), visited);
    else if (auto castOp = dyn_cast<memref::CastOp>(op))
      return getUnderlyingValueImpl(castOp.getSource(), visited);
    else if (isa<arts::UndefOp>(op))
      return nullptr;
    else
      return nullptr;
  } else {
    /// Value has no defining operation and is not a block argument
    return nullptr;
  }
}

//===----------------------------------------------------------------------===//
/// Underlying Value Tracing Utilities
//===----------------------------------------------------------------------===//

/// Strip numeric cast operations to find the underlying value.
/// Traverses through index casts, sign/zero extensions, and truncations.
Value stripNumericCasts(Value value) {
  while (true) {
    if (auto idxCast = value.getDefiningOp<arith::IndexCastOp>()) {
      value = idxCast.getIn();
      continue;
    }
    if (auto ext = value.getDefiningOp<arith::ExtSIOp>()) {
      value = ext.getIn();
      continue;
    }
    if (auto ext = value.getDefiningOp<arith::ExtUIOp>()) {
      value = ext.getIn();
      continue;
    }
    if (auto trunc = value.getDefiningOp<arith::TruncIOp>()) {
      value = trunc.getIn();
      continue;
    }
    break;
  }
  return value;
}

/// Traces the underlying root allocation value for the given value, unwinding
/// through various MLIR operations.
Value getUnderlyingValue(Value v) {
  SmallPtrSet<Value, 8> visited;
  return getUnderlyingValueImpl(v, visited);
}

/// Retrieves the underlying operation that defines the root value for the
/// given value.
Operation *getUnderlyingOperation(Value v) {
  Value underlyingValue = getUnderlyingValue(v);
  if (!underlyingValue)
    return nullptr;

  /// If the underlying value is a result of an operation, return that operation
  if (auto definingOp = underlyingValue.getDefiningOp())
    return definingOp;

  return nullptr;
}

/// Finds the datablock-related operation (DbAllocOp or DbAcquireOp) associated
/// with the given value.
Operation *getUnderlyingDb(Value v) {
  if (!v)
    return nullptr;

  /// Directly return the DbAcquireOp or DbAllocOp if present
  if (auto acq = v.getDefiningOp<DbAcquireOp>())
    return acq.getOperation();
  if (auto alloc = v.getDefiningOp<DbAllocOp>())
    return alloc.getOperation();
  if (auto dbLoad = v.getDefiningOp<DbRefOp>())
    return getUnderlyingDb(dbLoad.getSource());

  /// If it's a block argument of an EDT, map to the corresponding operand and
  /// recurse. Block arguments correspond to dependencies
  if (auto blockArg = v.dyn_cast<BlockArgument>()) {
    Block *block = blockArg.getOwner();
    Operation *owner = block->getParentOp();
    if (auto edt = dyn_cast<EdtOp>(owner)) {
      unsigned argIndex = blockArg.getArgNumber();
      /// EDT operands layout: [route] [dependencies...]
      /// Block arguments correspond to dependencies, so offset by 1
      auto [depsBegin, depsLen] = edt.getODSOperandIndexAndLength(1);
      if (argIndex < depsLen) {
        Value operand = edt.getOperand(depsBegin + argIndex);
        return getUnderlyingDb(operand);
      }
    }
  }

  /// Peel through common view/cast/gep nodes to reach DB ops
  if (Operation *def = v.getDefiningOp()) {
    if (auto gep = dyn_cast<DbGepOp>(def))
      return getUnderlyingDb(gep.getBasePtr());
    if (auto castOp = dyn_cast<memref::CastOp>(def))
      return getUnderlyingDb(castOp.getSource());
    if (auto subview = dyn_cast<memref::SubViewOp>(def))
      return getUnderlyingDb(subview.getSource());
  }

  /// As a last resort, try underlying root op if it's a DB op
  if (Operation *root = getUnderlyingOperation(v)) {
    if (isa<DbAcquireOp, DbAllocOp>(root))
      return root;
  }

  return nullptr;
}

/// Finds the DbAllocOp associated with the given value.
Operation *getUnderlyingDbAlloc(Value v) {
  Operation *underlyingDb = getUnderlyingDb(v);
  if (!underlyingDb)
    return nullptr;
  if (isa<DbAllocOp>(underlyingDb))
    return underlyingDb;
  auto dbAcquire = dyn_cast<DbAcquireOp>(underlyingDb);
  assert(dbAcquire);
  return getUnderlyingDbAlloc(dbAcquire.getSourcePtr());
}
//===----------------------------------------------------------------------===//
/// Operation Removal and Dead Code Elimination Utilities
//===----------------------------------------------------------------------===//

/// Recursively removes the given operation and all its dependent operations
/// that become dead, tracking visited operations to avoid infinite recursion.
static void removeOpImpl(Operation *op, OpBuilder &builder,
                         SmallPtrSet<Operation *, 32> &seen, bool recursive,
                         SetVector<Operation *> &opsToRemove) {
  if (!op || !op->getBlock())
    return;

  /// Already visited
  if (!seen.insert(op).second)
    return;

  ARTS_DEBUG("   - Removing operation: " << *op);

  /// Get dependents before modifying uses
  SmallVector<Operation *, 8> dependents;
  if (recursive) {
    for (Value result : op->getResults())
      for (Operation *user : result.getUsers())
        dependents.push_back(user);
  }

  /// If this is a terminator, remove its parent
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    if (Operation *parent = op->getParentOp())
      opsToRemove.insert(parent);
  }

  /// Drop all uses of this operation's results to dereference them
  op->dropAllUses();

  /// Erase the operation immediately since all uses have been dropped
  op->erase();

  /// No recursion requested
  if (!recursive)
    return;

  /// Remove dependents recursively
  for (Operation *userOp : dependents)
    removeOpImpl(userOp, builder, seen, recursive, opsToRemove);
}

/// Removes a set of operations from the module by dereferencing their uses
/// and recursively removing dependent operations that become dead.
void removeOps(ModuleOp module, SetVector<Operation *> &opsToRemove,
               bool recursive) {
  if (opsToRemove.empty())
    return;

  OpBuilder builder(module.getContext());
  SmallPtrSet<Operation *, 32> seen;

  /// Remove operations, processing newly discovered dead parents
  /// Use index-based iteration since we may add to opsToRemove during
  /// processing
  ARTS_DEBUG(" - Removing operations");
  size_t idx = 0;
  while (idx < opsToRemove.size()) {
    Operation *op = opsToRemove[idx++];
    removeOpImpl(op, builder, seen, recursive, opsToRemove);
  }
  opsToRemove.clear();
  ARTS_DEBUG(" - Removed " << seen.size() << " operations total");
}

/// Remove undef operations with optional recursion
void removeUndefOps(ModuleOp module) {
  SetVector<Operation *> tempOps;
  module.walk([&](arts::UndefOp op) { tempOps.insert(op); });
  ARTS_DEBUG(" - Removing " << tempOps.size() << " undef operations");
  removeOps(module, tempOps, true);
}

/// Replaces all results of the given operation with undef operations.
void replaceWithUndef(Operation *op, OpBuilder &builder) {
  if (!op || op->getNumResults() == 0)
    return;

  OpBuilder::InsertionGuard IG(builder);
  builder.setInsertionPoint(op);
  for (Value result : op->getResults()) {
    if (!result.use_empty()) {
      auto undef =
          builder.create<arts::UndefOp>(op->getLoc(), result.getType());
      result.replaceAllUsesWith(undef.getResult());
    }
  }
}

/// Replaces uses of one value with another under dominance constraints,
/// skipping the dominating operation.
void replaceUses(Value from, Value to, DominanceInfo &domInfo,
                 Operation *dominatingOp) {
  /// Ensure that 'from' has a defining operation.
  auto *definingOp = from.getDefiningOp();
  if (!definingOp)
    return;

  /// Replace all uses of 'from' with 'to' if the user is dominated by the
  /// defining operation of 'from'.
  from.replaceUsesWithIf(to, [&](OpOperand &operand) {
    if (operand.getOwner() == dominatingOp)
      return false;
    return domInfo.dominates(dominatingOp, operand.getOwner());
  });
}

/// Replaces uses according to a mapping of values.
void replaceUses(DenseMap<Value, Value> &rewireMap) {
  for (auto &rewire : rewireMap)
    rewire.first.replaceAllUsesWith(rewire.second);
  rewireMap.clear();
}

/// Replaces uses of a value within a specific region.
void replaceInRegion(Region &region, Value from, Value to) {
  from.replaceUsesWithIf(to, [&](OpOperand &operand) {
    return region.isAncestor(operand.getOwner()->getParentRegion());
  });
}

/// Replaces uses according to a mapping within a specific region.
void replaceInRegion(Region &region, DenseMap<Value, Value> &rewireMap,
                     bool clear) {
  for (auto &rewire : rewireMap)
    replaceInRegion(region, rewire.first, rewire.second);
  if (clear)
    rewireMap.clear();
}

//===----------------------------------------------------------------------===//
/// Index Splitting Utilities for Datablocks
//===----------------------------------------------------------------------===//

/// Splits datablock indices into outer and inner indices based on the
/// datablock's structure. For a memref<?xmemref<?xi32>> with indices [i, j, k]:
/// - Outer indices [i]: select which element memref
/// - Inner indices [j, k]: index into the element memref
std::pair<SmallVector<Value>, SmallVector<Value>>
splitDbIndices(Operation *dbOp, ValueRange indices, OpBuilder &builder,
               Location loc) {
  SmallVector<Value> outerIndices, innerIndices;

  /// Get datablock properties
  auto hasSingleSize = dbHasSingleSize(dbOp);
  auto outerRank = getSizesFromDb(dbOp).size();

  /// If the Db has single size, all the indices are inner indices
  if (hasSingleSize)
    innerIndices.assign(indices.begin(), indices.end());

  /// If the indices are less than or equal to the outer rank, all the indices
  /// are outer indices
  else if (indices.size() <= outerRank)
    outerIndices.assign(indices.begin(), indices.end());

  /// If the indices are greater than the outer rank, split the indices into
  /// outer and inner indices
  else {
    outerIndices.assign(indices.begin(), indices.begin() + outerRank);
    innerIndices.assign(indices.begin() + outerRank, indices.end());
  }

  /// For all other cases, add a default index of 0
  if (outerIndices.empty())
    outerIndices.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));
  if (innerIndices.empty())
    innerIndices.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));

  return std::make_pair(outerIndices, innerIndices);
}

//===----------------------------------------------------------------------===//
// Type Casting and Conversion Utilities
//===----------------------------------------------------------------------===//

/// Cast a value to index type if needed.
Value castToIndex(Value value, OpBuilder &builder, Location loc) {
  if (!value)
    return value;
  if (value.getType().isIndex())
    return value;
  if (value.getType().isIntOrIndex())
    return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                              value);
  return value;
}

//===----------------------------------------------------------------------===//
// Pattern Recognition and Analysis Utilities
//===----------------------------------------------------------------------===//

/// Extract original size from (N * scale) / scale pattern.
/// Common in malloc size calculations: malloc(N * sizeof(T)) / sizeof(T) -> N.
Value extractOriginalSize(Value numerator, Value denominator,
                          OpBuilder &builder, Location loc) {
  Value stripped = stripNumericCasts(numerator);
  if (auto mul = stripped.getDefiningOp<arith::MulIOp>()) {
    Value lhs = mul.getLhs();
    Value rhs = mul.getRhs();
    if (scalesAreEquivalent(lhs, denominator))
      return castToIndex(stripNumericCasts(rhs), builder, loc);
    if (scalesAreEquivalent(rhs, denominator))
      return castToIndex(stripNumericCasts(lhs), builder, loc);
  }
  return Value();
}

/// Extract array index from byte offset pattern: bytes = (index * elemBytes).
/// Handles common patterns where GEP indices are scaled by element byte size.
/// Returns the logical array index or null Value if pattern doesn't match.
Value extractArrayIndexFromByteOffset(Value byteOffset, Type elemType) {
  /// Strip any index casts to get to the core computation
  Value idxSource = byteOffset;
  while (auto castOp = idxSource.getDefiningOp<arith::IndexCastOp>())
    idxSource = castOp.getIn();

  /// Look for multiplication: index * elemBytes
  auto mulOp = idxSource.getDefiningOp<arith::MulIOp>();
  if (!mulOp)
    return Value();

  int64_t elemSize = getElementTypeByteSize(elemType);
  if (elemSize == 0)
    return Value();

  /// Identify which operand is the constant element size
  auto isElementSizeConstant = [elemSize](Value v) -> bool {
    if (auto constIdx = v.getDefiningOp<arith::ConstantIndexOp>())
      return constIdx.value() == elemSize;
    if (auto constInt = v.getDefiningOp<arith::ConstantIntOp>())
      return constInt.value() == elemSize && constInt.getType().isInteger(64);
    return false;
  };

  Value lhs = mulOp.getLhs();
  Value rhs = mulOp.getRhs();

  if (isElementSizeConstant(lhs))
    return rhs;
  if (isElementSizeConstant(rhs))
    return lhs;
  /// Pattern doesn't match expected form
  return Value();
}

} // namespace arts
} // namespace mlir
