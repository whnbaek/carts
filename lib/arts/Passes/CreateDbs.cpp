///==========================================================================
/// File: CreateDbs.cpp
///
/// This pass creates ARTS Dbs (DB) operations to handle memory
/// dependencies for EDTs (Event-Driven Tasks). Each EDT may execute in a
/// separate memory environment, so external memory references must be properly
/// managed.
///
/// Pass Overview:
/// 1. Identify external memory allocations used by EDTs
/// 2. Create arts.db_alloc operations for each allocation
/// 3. Analyze memory access patterns within each EDT
/// 4. Insert arts.db_acquire operations before EDTs with computed dependencies
/// 5. Update load/store operations to use acquired memory views
/// 6. Insert arts.db_release operations before EDT terminators
/// 7. Insert arts.db_free operations for db_alloc operations
///
/// This pass assumes the worst-case scenario for access modes, always using
/// inout.
//==========================================================================

#include "ArtsPassDetails.h"
#include "arts/Analysis/StringAnalysis.h"
#include "arts/ArtsDialect.h"
#include "arts/Utils/ArtsUtils.h"

#include "arts/Passes/ArtsPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

/// Debug
#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(create_dbs);

using namespace mlir;
using namespace mlir::arts;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//
namespace {

/// Cast a value to index type if needed.
static Value castToIndex(Value value, OpBuilder &builder, Location loc) {
  if (!value)
    return value;
  if (value.getType().isIndex())
    return value;
  if (value.getType().isIntOrIndex())
    return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                              value);
  return value;
}

/// Extract original size from (N * scale) / scale pattern.
/// Common in malloc size calculations: malloc(N * sizeof(T)) / sizeof(T) -> N.
static Value extractOriginalSize(Value numerator, Value denominator,
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

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct CreateDbsPass : public arts::CreateDbsBase<CreateDbsPass> {
  CreateDbsPass(bool identifyDbs) { this->identifyDbs = identifyDbs; }

  void runOnOperation() override;

private:
  bool identifyDbs;
  ModuleOp module;
  DenseMap<Value, DbAllocOp> allocToDbAlloc;
  DenseMap<Value, Value> dbPtrToOriginalAlloc;
  SmallVector<DbAllocOp, 8> createdDbAllocs;
  SetVector<Operation *> opsToRemove;
  StringAnalysis *stringAnalysis = nullptr;

  /// Core helper functions
  void cleanupControlDbOps();
  SetVector<Value> collectUsedAllocations();
  void createDbAllocOps(const SetVector<Value> &allocs, OpBuilder &builder);
  SetVector<Value> getEdtExternalValues(EdtOp edt);
  void processEdtDeps(EdtOp edt, OpBuilder &builder);

  /// Infer allocation type from the defining operation
  DbAllocType inferAllocType(Value basePtr);

  /// Helper: mark an operation for deferred removal
  void markForRemoval(Operation *op) {
    if (!op)
      return;
    opsToRemove.insert(op);
  }

  /// Access adjustment
  void adjustAccesses(EdtOp edt, Value originalValue, Value dbAcquire,
                      OpBuilder &builder);

  /// Extract the logical (pre-malloc) size for a dynamically computed value
  Value getOriginalSizeValue(Value size, OpBuilder &builder, Location loc);

  /// DbAlloc cleanup
  void insertDbFreeOperations(DbAllocOp dbAlloc, OpBuilder &builder);

  /// Remove dealloc operations for allocations converted to DBs
  void removeDeallocOperations();

  /// Helper function to check if a value is related to the original allocation
  bool isRelatedToAllocation(Value value, DbAllocOp dbAlloc);
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass Entry Point
//===----------------------------------------------------------------------===//
void CreateDbsPass::runOnOperation() {
  module = getOperation();
  allocToDbAlloc.clear();
  createdDbAllocs.clear();
  opsToRemove.clear();
  if (stringAnalysis) {
    delete stringAnalysis;
    stringAnalysis = nullptr;
  }
  ARTS_INFO_HEADER(CreateDbsPass);
  ARTS_DEBUG_REGION(module.dump(););

  OpBuilder builder(module.getContext());

  /// Run StringAnalysis to detect string memrefs
  stringAnalysis = new StringAnalysis(module);
  stringAnalysis->run();
  assert(stringAnalysis && "StringAnalysis must be created");

  /// Phase 1: Cleanup existing Control DB operations
  ARTS_INFO("Phase 1: Cleaning up existing Control DB operations");
  cleanupControlDbOps();

  /// Phase 2: Collect allocations used in EDTs
  ARTS_INFO("Phase 2: Collecting allocations used in EDTs");
  SetVector<Value> usedAllocs = collectUsedAllocations();

  ARTS_DEBUG(" - Found " << usedAllocs.size()
                         << " allocations to convert to DB");

  /// Convert allocations to DBs
  createDbAllocOps(usedAllocs, builder);

  /// Phase 3: Process EDTs for dependencies
  SmallVector<EdtOp> allEdts;
  module.walk([&](EdtOp edt) { allEdts.push_back(edt); });
  ARTS_INFO("Phase 3: Processing " << allEdts.size() << " EDT operations");
  for (auto it = allEdts.rbegin(); it != allEdts.rend(); ++it)
    processEdtDeps(*it, builder);

  /// Phase 4: Insert DbFree operations
  ARTS_INFO("Phase 4: Inserting DbFree operations for "
            << createdDbAllocs.size() << " DB allocations");
  for (DbAllocOp dbAlloc : createdDbAllocs)
    insertDbFreeOperations(dbAlloc, builder);

  /// Phase 5: Remove dealloc operations for converted allocations
  ARTS_INFO("Phase 5: Removing dealloc operations for converted allocations");
  removeDeallocOperations();

  /// Phase 6: Clean up legacy allocations
  ARTS_INFO(" - Cleaning up legacy allocations");
  removeOps(module, opsToRemove, true);

  /// Phase 7: Enabling verification for EDTs
  module.walk([](EdtOp edt) { edt.removeNoVerifyAttr(); });

  /// Phase 8: Simplify the module
  DominanceInfo domInfo(module);
  arts::simplifyIR(module, domInfo);

  ARTS_INFO_FOOTER(CreateDbsPass);
  ARTS_DEBUG_REGION(module.dump(););
  delete stringAnalysis;
  stringAnalysis = nullptr;
}

//===----------------------------------------------------------------------===//
// Allocation Type Inference
//===----------------------------------------------------------------------===//
DbAllocType CreateDbsPass::inferAllocType(Value basePtr) {
  if (!basePtr)
    return DbAllocType::heap;

  if (auto definingOp = basePtr.getDefiningOp()) {
    if (isa<memref::AllocaOp>(definingOp))
      return DbAllocType::stack;
    if (isa<memref::AllocOp>(definingOp))
      return DbAllocType::heap;
    if (isa<memref::GetGlobalOp>(definingOp))
      return DbAllocType::global;
  }
  return DbAllocType::unknown;
}

//===----------------------------------------------------------------------===//
// Cleanup Control DB Operations
//===----------------------------------------------------------------------===//
void CreateDbsPass::cleanupControlDbOps() {
  /// Erase all db_control ops from previous passes, replacing uses with pointer
  module.walk([&](DbControlOp dbControl) {
    dbControl.getSubview().replaceAllUsesWith(dbControl.getPtr());
    dbControl.erase();
  });

  /// Clear all existing EDT dependencies and block args that are not used
  /// within the block
  module.walk([&](EdtOp edt) {
    Block &block = edt.getBody().front();

    /// Only erase block arguments that are not used within the block
    block.eraseArguments([&](BlockArgument arg) {
      /// Check if this argument is used anywhere in the block
      bool isUsed = false;
      block.walk([&](Operation *op) {
        for (Value operand : op->getOperands()) {
          if (operand == arg) {
            isUsed = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });

      /// Only erase if the argument is not used
      return !isUsed;
    });

    /// Preserve the route operand when clearing dependencies
    SmallVector<Value> preservedOperands;
    if (Value route = edt.getRoute())
      preservedOperands.push_back(route);
    edt->setOperands(preservedOperands);
  });
}

//===----------------------------------------------------------------------===//
// Collect Allocations Used in EDTs
//===----------------------------------------------------------------------===//
SetVector<Value> CreateDbsPass::collectUsedAllocations() {
  SetVector<Value> allUsedAllocs;
  module.walk([&](EdtOp edt) {
    edt.walk([&](Operation *op) {
      /// Check all operands of all operations for memory references
      for (Value operand : op->getOperands()) {
        if (!operand.getType().isa<MemRefType>())
          continue;
        /// Get the underlying value of the memory reference
        Value base = arts::getUnderlyingValue(operand);
        if (base)
          allUsedAllocs.insert(base);
      }
    });
  });
  return allUsedAllocs;
}

//===----------------------------------------------------------------------===//
// Create DB Allocation Operations
//===----------------------------------------------------------------------===//
void CreateDbsPass::createDbAllocOps(const SetVector<Value> &allocs,
                                     OpBuilder &builder) {
  ARTS_DEBUG("Creating DbAlloc operations for " << allocs.size()
                                                << " allocations");

  for (Value alloc : allocs) {
    Operation *allocOp = alloc.getDefiningOp();
    if (!allocOp)
      continue;

    auto memRefType = alloc.getType().cast<MemRefType>();

    Location loc = allocOp->getLoc();
    builder.setInsertionPointAfter(allocOp);

    /// Infer allocation type based on the defining operation
    DbAllocType allocType = inferAllocType(alloc);
    ArtsMode mode = ArtsMode::inout;
    DbMode dbMode = DbMode::write;
    Type elementType = memRefType.getElementType();
    while (auto nestedMemRef = elementType.dyn_cast<MemRefType>())
      elementType = nestedMemRef.getElementType();

    /// Extract dimension sizes from the memref type
    SmallVector<Value> sizes, elementSizes;
    const int rank = memRefType.getRank();
    sizes.reserve(rank);

    for (int i = 0; i < rank; ++i) {
      if (memRefType.isDynamicDim(i)) {
        sizes.push_back(builder.create<arts::DbDimOp>(loc, alloc, (int64_t)i));
      } else {
        sizes.push_back(builder.create<arith::ConstantIndexOp>(
            loc, memRefType.getDimSize(i)));
      }
    }

    /// If this is a string memref, allocate a single DB with full payload:
    /// move all logical sizes into elementSizes and clear outer sizes
    if (stringAnalysis && stringAnalysis->isStringMemRef(alloc)) {
      elementSizes = sizes;
      sizes.clear();
      sizes.push_back(builder.create<arith::ConstantIndexOp>(loc, 1));
    }

    auto zeroRoute = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    Type pointerType;
    if (stringAnalysis && stringAnalysis->isStringMemRef(alloc)) {
      pointerType = arts::getElementMemRefType(elementType, elementSizes);
    } else {
      /// For regular memrefs, use the original memref type
      pointerType = memRefType;
    }
    auto dbAllocOp = builder.create<DbAllocOp>(loc, mode, zeroRoute, allocType,
                                               dbMode, elementType, pointerType,
                                               sizes, elementSizes);

    alloc.replaceUsesWithIf(dbAllocOp.getPtr(), [&](OpOperand &use) -> bool {
      Operation *user = use.getOwner();
      /// Skip operands to the db_alloc, db_dim ops and EDT regions.
      return !isa<DbAllocOp>(user) && !isa<DbDimOp>(user) &&
             !user->getParentOfType<EdtOp>();
    });

    /// Store mappings for later use
    allocToDbAlloc[alloc] = dbAllocOp;
    dbPtrToOriginalAlloc[dbAllocOp.getPtr()] = alloc;
    createdDbAllocs.push_back(dbAllocOp);
  }
}

//===----------------------------------------------------------------------===//
// Extract External Values from EDT
//===----------------------------------------------------------------------===//
SetVector<Value> CreateDbsPass::getEdtExternalValues(EdtOp edt) {
  SetVector<Value> externalValues;
  auto &edtRegion = edt.getRegion();

  /// Collect external values used by this EDT (excluding nested EDTs)
  edt.walk([&](Operation *op) -> WalkResult {
    if (isa<EdtOp>(op) && op != edt.getOperation())
      return WalkResult::skip();

    Value mem;
    if (auto load = dyn_cast<memref::LoadOp>(op))
      mem = load.getMemref();
    else if (auto store = dyn_cast<memref::StoreOp>(op))
      mem = store.getMemref();
    else
      return WalkResult::advance();

    /// Mark as external if defined outside the EDT region
    Value underlyingValue = arts::getUnderlyingValue(mem);
    if (!underlyingValue)
      return WalkResult::advance();
    Operation *definingOp = underlyingValue.getDefiningOp();
    if (definingOp && !edtRegion.isAncestor(definingOp->getParentRegion()))
      externalValues.insert(underlyingValue);
    return WalkResult::advance();
  });

  return externalValues;
}

//===----------------------------------------------------------------------===//
// Process EDT Dependencies
/// For each EDT, analyze external memory dependencies and insert
/// acquire/release
//===----------------------------------------------------------------------===//
void CreateDbsPass::processEdtDeps(EdtOp edt, OpBuilder &builder) {
  SetVector<Value> externalValues = getEdtExternalValues(edt);

  if (externalValues.empty())
    return;

  ARTS_DEBUG(" - Processing EDT with " << externalValues.size()
                                       << " external dependencies");

  /// For each external memory value, create acquire and release operations
  for (Value value : externalValues) {
    /// Locate the source Db (db_alloc or db_acquire)
    Operation *sourceOp = nullptr;
    if (allocToDbAlloc.count(value))
      sourceOp = allocToDbAlloc[value];
    else
      sourceOp = arts::getUnderlyingDb(value);

    /// If no DB was found (e.g., unrelated pointer or not yet converted),
    /// skip this external value. Downstream passes may handle it or it is not
    /// a datablock candidate.
    if (!sourceOp)
      continue;
    assert(sourceOp->getNumResults() == 2 &&
           "Source operation must have 2 results");
    auto sourceGuid = sourceOp->getResult(0);
    auto sourcePtr = sourceOp->getResult(1);
    assert((sourceGuid && sourcePtr) && "Source guid and ptr must be non-null");

    /// Create acquire operation (pass both source guid and ptr)
    builder.setInsertionPoint(edt);

    SmallVector<Value> indices, offsets, sizes;

    /// For string memrefs, the source has outer size 1, so acquire should use
    /// size 1 For non-string memrefs, we defer slicing to downstream passes
    /// (empty offsets/sizes)
    if (stringAnalysis && stringAnalysis->isStringMemRef(value)) {
      /// Acquire the whole string DB (outer size 1, offset 0)
      offsets.push_back(
          builder.create<arith::ConstantIndexOp>(edt.getLoc(), 0));
      sizes.push_back(builder.create<arith::ConstantIndexOp>(edt.getLoc(), 1));
    }

    DbAcquireOp acquire =
        builder.create<DbAcquireOp>(edt.getLoc(), ArtsMode::inout, sourceGuid,
                                    sourcePtr, indices, offsets, sizes);

    /// Add acquire pointer as EDT dependency
    SmallVector<Value> newOperands;
    if (Value route = edt.getRoute())
      newOperands.push_back(route);
    auto existingDeps = edt.getDependenciesAsVector();
    newOperands.append(existingDeps.begin(), existingDeps.end());
    newOperands.push_back(acquire.getPtr());
    edt->setOperands(newOperands);

    /// Add corresponding block argument for the acquired view
    auto sourceType = sourcePtr.getType().dyn_cast<MemRefType>();
    BlockArgument dbAcquireArg =
        edt.getBody().front().addArgument(sourceType, edt.getLoc());

    /// Rewrite memory accesses to use the acquired view
    adjustAccesses(edt, value, dbAcquireArg, builder);

    /// Insert release before EDT terminator
    builder.setInsertionPoint(edt.getBody().front().getTerminator());
    builder.create<DbReleaseOp>(edt.getLoc(), dbAcquireArg);
  }
}

//===----------------------------------------------------------------------===//
// Adjust Memory Accesses
/// Rewrite load/store operations to use acquired Db
//===----------------------------------------------------------------------===//

/// Helper function to adjust memref indices when rank changes
template <typename MemRefOp>
void adjustMemrefIndices(MemRefOp op, Value originalMemref, Value dbAcquire,
                         OpBuilder &builder) {
  auto originalMemRefType = originalMemref.getType().cast<MemRefType>();
  auto newMemRefType = dbAcquire.getType().cast<MemRefType>();

  if (originalMemRefType.getRank() < newMemRefType.getRank()) {
    // Add zero indices for the additional dimensions
    SmallVector<Value> indices;
    indices.reserve(newMemRefType.getRank());

    auto zeroIndex = builder.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    for (int i = 0; i < newMemRefType.getRank(); ++i)
      indices.push_back(zeroIndex);
    op.getIndicesMutable().assign(indices);
  }
}

void CreateDbsPass::adjustAccesses(EdtOp edt, Value originalValue,
                                   Value dbAcquire, OpBuilder &builder) {
  edt->getRegion(0).walk([&](Operation *op) -> WalkResult {
    if (isa<EdtOp>(op))
      return WalkResult::skip();

    Value mem;
    if (auto load = dyn_cast<memref::LoadOp>(op))
      mem = load.getMemref();
    else if (auto store = dyn_cast<memref::StoreOp>(op))
      mem = store.getMemref();
    else if (auto gep = dyn_cast<DbGepOp>(op))
      mem = gep.getBasePtr();
    else
      return WalkResult::advance();

    if (arts::getUnderlyingValue(mem) != originalValue)
      return WalkResult::advance();

    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      load.getMemrefMutable().assign(dbAcquire);
      adjustMemrefIndices(load, mem, dbAcquire, builder);
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      store.getMemrefMutable().assign(dbAcquire);
      adjustMemrefIndices(store, mem, dbAcquire, builder);
    }
    return WalkResult::advance();
  });

  /// Replace remaining uses of the original value with the db acquire if they
  /// are in the edt region
  originalValue.replaceUsesWithIf(dbAcquire, [&](OpOperand &use) -> bool {
    Operation *user = use.getOwner();
    return user->getParentOfType<EdtOp>() == edt;
  });
}

//===----------------------------------------------------------------------===//
// Size Value Extraction
/// Extract logical allocation size from sizeof-scaled expressions
//===----------------------------------------------------------------------===//
Value CreateDbsPass::getOriginalSizeValue(Value size, OpBuilder &builder,
                                          Location loc) {
  if (!size)
    return size;

  if (auto div = size.getDefiningOp<arith::DivUIOp>())
    if (Value original =
            extractOriginalSize(div.getLhs(), div.getRhs(), builder, loc))
      return original;

  if (auto div = size.getDefiningOp<arith::DivSIOp>())
    if (Value original =
            extractOriginalSize(div.getLhs(), div.getRhs(), builder, loc))
      return original;

  return castToIndex(stripNumericCasts(size), builder, loc);
}

//===----------------------------------------------------------------------===//
// Insert DB Free Operations
/// Insert db_free for each Db, replacing existing dealloc operations
//===----------------------------------------------------------------------===//
void CreateDbsPass::insertDbFreeOperations(DbAllocOp dbAlloc,
                                           OpBuilder &builder) {
  Region *region = dbAlloc->getParentRegion();
  if (!region)
    return;

  Operation *regionOwner = region->getParentOp();
  if (!regionOwner)
    return;

  /// Insert a conservative free at the end of the owning function/region.
  if (auto parentFunc = dyn_cast<func::FuncOp>(regionOwner)) {
    /// Skip outlined EDT functions
    if (parentFunc.getName().startswith("__arts_edt_"))
      return;
    Block &lastBlock = parentFunc.getBody().back();
    Operation *terminator = lastBlock.getTerminator();
    if (!terminator)
      return;
    OpBuilder::InsertionGuard IG(builder);
    builder.setInsertionPoint(terminator);
    builder.create<DbFreeOp>(dbAlloc.getLoc(), dbAlloc.getGuid());
    builder.create<DbFreeOp>(dbAlloc.getLoc(), dbAlloc.getPtr());
    return;
  }

  /// If not within a function, skip insertion to avoid unsafe mutations.
}

//===----------------------------------------------------------------------===//
// Remove Dealloc Operations
/// Remove memref.dealloc operations for allocations converted to DBs
//===----------------------------------------------------------------------===//
void CreateDbsPass::removeDeallocOperations() {
  /// Walk the module to find memref.dealloc operations
  module.walk([&](memref::DeallocOp deallocOp) {
    Value deallocValue = deallocOp.getMemref();

    ARTS_DEBUG("Checking dealloc: " << deallocOp
                                    << " on value: " << deallocValue);

    /// Check if this dealloc corresponds to an allocation we converted to DB
    Value underlyingValue = arts::getUnderlyingValue(deallocValue);
    ARTS_DEBUG("Underlying value: " << underlyingValue);

    if (underlyingValue) {
      /// Check if this underlying value was converted to a DB
      if (allocToDbAlloc.count(underlyingValue)) {
        ARTS_DEBUG("Marking dealloc operation for removal (original alloc): "
                   << deallocOp);
        markForRemoval(deallocOp);
        return;
      }
    }

    /// Check if this dealloc is on a DB pointer that we created
    if (dbPtrToOriginalAlloc.count(deallocValue)) {
      ARTS_DEBUG(
          "Marking dealloc operation for removal (DB ptr): " << deallocOp);
      markForRemoval(deallocOp);
      return;
    }
  });
}

//===----------------------------------------------------------------------===//
// Allocation Relationship Check
/// Check if a value is related to a specific Db allocation
//===----------------------------------------------------------------------===//
bool CreateDbsPass::isRelatedToAllocation(Value value, DbAllocOp dbAlloc) {
  /// First check if the value is directly from this DbAllocOp
  if (auto dbAllocOp = value.getDefiningOp<DbAllocOp>()) {
    if (dbAllocOp == dbAlloc)
      return true;
  }

  /// Use getUnderlyingValue to find the root allocation for both values
  Value valueRoot = arts::getUnderlyingValue(value);
  Value dbAllocRoot = arts::getUnderlyingValue(dbAlloc.getAddress());

  /// If either root is null, we can't determine the relationship
  if (!valueRoot || !dbAllocRoot)
    return false;

  /// Check if the root allocations are the same
  return valueRoot == dbAllocRoot;
}

//===----------------------------------------------------------------------===//
// Pass creation
//===----------------------------------------------------------------------===//
namespace mlir {
namespace arts {
std::unique_ptr<Pass> createCreateDbsPass(bool identifyDbs) {
  return std::make_unique<CreateDbsPass>(identifyDbs);
}
} // namespace arts
} // namespace mlir
