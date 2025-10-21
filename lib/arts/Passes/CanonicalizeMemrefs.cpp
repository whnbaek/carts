///==========================================================================
/// File: CanonicalizeMemrefs.cpp
///
/// This pass preprocesses nested allocation patterns by converting them to
/// single N-dimensional allocations. This is essential for proper DB creation
/// and LLVM code generation.
///
/// Pass Overview:
/// 1. Detect nested allocation patterns (arrays of pointers)
/// 2. Convert them to single N-dimensional allocations
/// 3. Rewrite memory accesses to use direct N-dimensional indexing
/// 4. Remove the initialization loops and legacy allocations
///
///
/// Instead of creating separate DBs for each row, this pass:
/// 1. Detects the nested allocation pattern
/// 2. Creates a single N-dimensional datablock (e.g., memref<?x?xf64>)
/// 3. Transforms cascaded memory accesses (load(load(A[i])[j])) into
///    direct N-dimensional access (load(A_ND[i,j]))
/// 4. Removes the initialization loop that allocated individual rows
//==========================================================================

#include "ArtsPassDetails.h"
#include "arts/ArtsDialect.h"
#include "arts/Utils/ArtsUtils.h"

#include "arts/Passes/ArtsPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

/// Debug
#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(canonicalize_memrefs);

using namespace mlir;
using namespace mlir::arts;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct CanonicalizeMemrefsPass
    : public arts::CanonicalizeMemrefsBase<CanonicalizeMemrefsPass> {

  void runOnOperation() override;

private:
  ModuleOp module;
  SetVector<Operation *> opsToRemove;

  /// Helper: mark an operation for deferred removal
  void markForRemoval(Operation *op) {
    if (!op)
      return;
    opsToRemove.insert(op);
  }

  /// Pattern recognition for nested allocations
  struct NestedAllocPattern {
    Value outerAlloc;      /// The outermost array allocation for this level
    Value dimSize;         /// Size for this dimension
    Value innerDimSize;    /// Inner size only for base case
    Value innerAllocValue; /// Inner alloc value only for base case
    Type elementType;      /// Final scalar element type (e.g., f64)
    scf::ForOp initLoop;   /// The initialization loop for this level
    std::unique_ptr<NestedAllocPattern> inner; /// Next nested level

    NestedAllocPattern() = default;
    NestedAllocPattern(NestedAllocPattern &&) = default;
    NestedAllocPattern &operator=(NestedAllocPattern &&) = default;
    NestedAllocPattern(const NestedAllocPattern &) = delete;
    NestedAllocPattern &operator=(const NestedAllocPattern &) = delete;

    int getDepth() const { return inner ? 1 + inner->getDepth() : 2; }

    SmallVector<Value> getAllSizes() const {
      SmallVector<Value> s = {dimSize};
      if (inner)
        s.append(inner->getAllSizes());
      else
        s.push_back(innerDimSize);

      return s;
    }

    SmallVector<scf::ForOp> getAllInitLoops() const {
      SmallVector<scf::ForOp> l = {initLoop};
      if (inner)
        l.append(inner->getAllInitLoops());
      return l;
    }

    SmallVector<Value> getAllNestedAllocs() const {
      SmallVector<Value> a = {outerAlloc};
      if (inner)
        a.append(inner->getAllNestedAllocs());
      else if (innerAllocValue)
        a.push_back(innerAllocValue);
      return a;
    }

    Type getFinalElementType() const { return elementType; }
  };

  std::optional<NestedAllocPattern> detectNestedAllocPattern(Value alloc);
  void transformNestedAccesses(Value outerAlloc, Value ndPtr);

  /// Extract the logical (pre-malloc) size for a dynamically computed value
  Value getOriginalSizeValue(Value size, OpBuilder &builder, Location loc);
};

//===----------------------------------------------------------------------===//
// Pass Entry Point
//===----------------------------------------------------------------------===//
void CanonicalizeMemrefsPass::runOnOperation() {
  module = getOperation();
  opsToRemove.clear();

  ARTS_INFO_HEADER(CanonicalizeMemrefsPass);
  ARTS_DEBUG_REGION(module.dump(););

  OpBuilder builder(module.getContext());

  /// Phase 1: Detect and convert nested allocation patterns
  ARTS_INFO("Phase 1: Detecting nested allocation patterns");
  DenseMap<scf::ForOp, SmallVector<NestedAllocPattern>> nestedPatternsByLoop;

  module.walk([&](memref::AllocOp allocOp) {
    Value alloc = allocOp.getResult();
    /// Skip allocations inside loops - these are inner allocations
    if (allocOp->getParentOfType<scf::ForOp>())
      return;
    if (auto pattern = detectNestedAllocPattern(alloc)) {
      nestedPatternsByLoop[pattern->initLoop].push_back(std::move(*pattern));
    }
  });

  if (nestedPatternsByLoop.empty()) {
    ARTS_DEBUG(" - No nested allocation patterns found");
    ARTS_INFO_FOOTER(CanonicalizeMemrefsPass);
    return;
  }

  /// Convert nested patterns to a single N-D stack allocation and rewrite
  /// accesses to index directly into that allocation.
  ARTS_INFO("Phase 2: Converting nested patterns to N-D alloca and rewriting "
            "accesses");
  for (auto &[loop, patterns] : nestedPatternsByLoop) {
    for (auto &pattern : patterns) {
      auto allocOp = pattern.outerAlloc.getDefiningOp<memref::AllocOp>();
      if (!allocOp)
        continue;

      Location loc = allocOp->getLoc();
      SmallVector<Value> allSizes = pattern.getAllSizes();
      if (allSizes.empty())
        continue;

      builder.setInsertionPoint(pattern.initLoop);

      /// Extract logical sizes
      SmallVector<Value> originalSizes;
      originalSizes.reserve(allSizes.size());
      for (Value size : allSizes)
        originalSizes.push_back(getOriginalSizeValue(size, builder, loc));

      SmallVector<int64_t> shape(pattern.getDepth(), ShapedType::kDynamic);
      MemRefType ndType = MemRefType::get(shape, pattern.getFinalElementType());
      Value ndAlloc =
          builder.create<memref::AllocaOp>(loc, ndType, originalSizes);

      /// Rewrite cascaded memory accesses to direct N-D indexing
      transformNestedAccesses(pattern.outerAlloc, ndAlloc);

      /// Schedule the initialization loop and legacy allocations for removal.
      auto legacyAllocs = pattern.getAllNestedAllocs();
      for (Value legacy : legacyAllocs) {
        if (!legacy)
          continue;
        if (Operation *def = legacy.getDefiningOp())
          opsToRemove.insert(def);
      }

      /// Clean up obsolete outer allocation
      Value outer = pattern.outerAlloc;
      if (Operation *defOp = outer.getDefiningOp())
        opsToRemove.insert(defOp);
    }
  }

  ARTS_INFO("Phase 3: Removing legacy allocations");
  removeOps(module, opsToRemove, true);
  ARTS_INFO_FOOTER(CanonicalizeMemrefsPass);
  ARTS_DEBUG_REGION(module.dump(););
}

//===----------------------------------------------------------------------===//
// Size Value Extraction
/// Extract logical allocation size from sizeof-scaled expressions
//===----------------------------------------------------------------------===//
Value CanonicalizeMemrefsPass::getOriginalSizeValue(Value size,
                                                    OpBuilder &builder,
                                                    Location loc) {
  if (!size)
    return size;

  if (auto div = size.getDefiningOp<arith::DivUIOp>())
    if (Value original =
            arts::extractOriginalSize(div.getLhs(), div.getRhs(), builder, loc))
      return original;

  if (auto div = size.getDefiningOp<arith::DivSIOp>())
    if (Value original =
            arts::extractOriginalSize(div.getLhs(), div.getRhs(), builder, loc))
      return original;

  return size;
}

//===----------------------------------------------------------------------===//
// Nested Allocation Pattern Detection
/// Detect multi-dimensional arrays allocated as nested arrays-of-pointers
//===----------------------------------------------------------------------===//
std::optional<CanonicalizeMemrefsPass::NestedAllocPattern>
CanonicalizeMemrefsPass::detectNestedAllocPattern(Value alloc) {
  auto currentType = alloc.getType().dyn_cast<MemRefType>();
  if (!currentType)
    return std::nullopt;

  auto elementType = currentType.getElementType();
  auto innerMemRefType = elementType.dyn_cast<MemRefType>();
  if (!innerMemRefType)
    return std::nullopt;

  /// Get this dimension size
  Value thisSize;
  auto allocOp = alloc.getDefiningOp<memref::AllocOp>();
  if (!allocOp)
    return std::nullopt;
  Location loc = allocOp->getLoc();
  if (currentType.isDynamicDim(0) && !allocOp.getDynamicSizes().empty()) {
    thisSize = allocOp.getDynamicSizes()[0];
  } else if (!currentType.isDynamicDim(0)) {
    OpBuilder builder(module.getContext());
    builder.setInsertionPointAfter(allocOp);
    thisSize =
        builder.create<arith::ConstantIndexOp>(loc, currentType.getDimSize(0));
  } else {
    return std::nullopt;
  }

  /// Find the initialization loop and inner allocations
  scf::ForOp initLoop = nullptr;
  Value storedValue;

  for (Operation *user : alloc.getUsers()) {
    if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
      if (storeOp.getMemref() != alloc)
        continue;

      storedValue = storeOp.getValue();
      auto innerAlloc = storedValue.getDefiningOp<memref::AllocOp>();
      if (!innerAlloc)
        continue;

      auto forOp = storeOp->getParentOfType<scf::ForOp>();
      if (!forOp)
        continue;

      initLoop = forOp;
      auto innerAllocType = innerAlloc.getResult().getType().cast<MemRefType>();

      /// Get inner dimension size and hoist it if needed
      Value innerSize;
      if (innerAllocType.isDynamicDim(0) &&
          !innerAlloc.getDynamicSizes().empty()) {
        innerSize = innerAlloc.getDynamicSizes()[0];
      } else if (!innerAllocType.isDynamicDim(0)) {
        OpBuilder builder(innerAlloc->getContext());
        builder.setInsertionPoint(innerAlloc);
        innerSize = builder.create<arith::ConstantIndexOp>(
            innerAlloc->getLoc(), innerAllocType.getDimSize(0));
      }

      if (!innerSize)
        continue;

      /// Recursively detect inner pattern
      auto innerPattern = detectNestedAllocPattern(innerAlloc.getResult());

      NestedAllocPattern pattern;
      pattern.outerAlloc = alloc;
      pattern.dimSize = thisSize;
      pattern.initLoop = initLoop;
      if (innerPattern) {
        pattern.inner =
            std::make_unique<NestedAllocPattern>(std::move(*innerPattern));
        pattern.elementType = innerPattern->getFinalElementType();
      } else {
        pattern.elementType = innerAllocType.getElementType();
        pattern.innerDimSize = innerSize;
        pattern.innerAllocValue = innerAlloc.getResult();
      }

      ARTS_DEBUG(" - Detected nested allocation pattern: depth="
                 << pattern.getDepth()
                 << ", element_type=" << pattern.elementType);

      return pattern;
    }
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Transform Nested Memory Accesses
/// Convert cascaded load/store chains into direct N-dimensional indexing
/// Pattern: load(load(A[i])[j]) -> load(A_2D[i,j])
//===----------------------------------------------------------------------===//
void CanonicalizeMemrefsPass::transformNestedAccesses(Value outerAlloc,
                                                      Value ndPtr) {

  MemRefType ndType = ndPtr.getType().cast<MemRefType>();
  int rank = ndType.getRank();
  int expectedChainLength = rank - 1;
  OpBuilder builder(module.getContext());

  struct ChainInfo {
    SmallVector<Value> indices;
    SmallVector<Operation *> loads;
  };

  /// Recursive function to collect index chain
  auto collectChain = [&](Value mem, auto &&self) -> std::optional<ChainInfo> {
    Value base = arts::getUnderlyingValue(mem);
    if (base == outerAlloc)
      return ChainInfo{{}, {}};

    if (auto defOp = mem.getDefiningOp<memref::LoadOp>()) {
      auto indicesRange = defOp.getIndices();
      SmallVector<Value> thisIndices(indicesRange.begin(), indicesRange.end());
      if (thisIndices.size() != 1)
        return std::nullopt;

      auto prevOpt = self(defOp.getMemref(), self);
      if (!prevOpt)
        return std::nullopt;

      ChainInfo prev = *prevOpt;
      prev.indices.push_back(thisIndices[0]);
      prev.loads.push_back(defOp);
      return prev;
    }

    return std::nullopt;
  };

  module.walk([&](Operation *op) -> WalkResult {
    Value mem;
    SmallVector<Value> indices;
    bool isLoad = false;
    Value storedValue;
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      mem = loadOp.getMemref();
      indices.assign(loadOp.getIndices().begin(), loadOp.getIndices().end());
      isLoad = true;
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      mem = storeOp.getMemref();
      indices.assign(storeOp.getIndices().begin(), storeOp.getIndices().end());
      isLoad = false;
      storedValue = storeOp.getValue();
    } else if (auto dbControlOp = dyn_cast<arts::DbControlOp>(op)) {
      dbControlOp.getPtrMutable().assign(ndPtr);
      return WalkResult::advance();
    } else {
      return WalkResult::advance();
    }

    /// Collect the index chain
    auto chainOpt = collectChain(mem, collectChain);
    if (!chainOpt)
      return WalkResult::advance();

    SmallVector<Value> fullIndices = chainOpt->indices;
    if (static_cast<int>(fullIndices.size()) != expectedChainLength)
      return WalkResult::advance();

    /// Check if the current indices are a single index
    if (indices.size() != 1)
      return WalkResult::advance();

    /// Combine the index chain with the current indices
    fullIndices.push_back(indices[0]);

    builder.setInsertionPoint(op);
    if (isLoad) {
      auto newLoad =
          builder.create<memref::LoadOp>(op->getLoc(), ndPtr, fullIndices);
      op->getResult(0).replaceAllUsesWith(newLoad);
    } else {
      builder.create<memref::StoreOp>(op->getLoc(), storedValue, ndPtr,
                                      fullIndices);
    }

    /// Schedule the operation for removal
    opsToRemove.insert(op);
    for (auto *load : chainOpt->loads)
      opsToRemove.insert(load);

    return WalkResult::advance();
  });
}

//===----------------------------------------------------------------------===//
// Pass creation
//===----------------------------------------------------------------------===//
namespace mlir {
namespace arts {
std::unique_ptr<Pass> createCanonicalizeMemrefsPass() {
  return std::make_unique<CanonicalizeMemrefsPass>();
}
} // namespace arts
} // namespace mlir