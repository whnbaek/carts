///==========================================================================
/// File: DbLowering.cpp
///
/// Ensures datablock (DB) allocations use heap-resident handles.
/// - Stack-allocated DB sources are replaced with heap allocations while
///   preserving the typed descriptor structure.
/// - Heap/global sources are left typed but recreated so downstream users
///   see a uniform heap-based handle.
/// Pointer-result uses are updated; GUID uses remain unchanged.
/// db_acquire/release ops are recreated with the new handles.
///==========================================================================
#include "ArtsPassDetails.h"
#include "arts/ArtsDialect.h"
#include "arts/Codegen/ArtsCodegen.h"
#include "arts/Passes/ArtsPasses.h"
#include "arts/Utils/ArtsDebug.h"
#include "arts/Utils/ArtsUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "polygeist/Ops.h"

ARTS_DEBUG_SETUP(db_lowering);
using namespace mlir;
using namespace arts;
namespace {
struct DbLoweringPass : public arts::DbLoweringBase<DbLoweringPass> {
  ~DbLoweringPass() override { delete AC; }
  void runOnOperation() override;

private:
  /// Processes all DB allocation ops, converting to opaque pointers.
  void convertDbAllocOps();
  /// Updates all users of the old DB alloc to use the new one.
  void updateUsers(DbAllocOp oldAllocOp, DbAllocOp newAllocOp);

private:
  SetVector<Operation *> opsToRemove;
  ModuleOp module;
  ArtsCodegen *AC = nullptr;
};
} // namespace

//===----------------------------------------------------------------------===//
/// Lower datablock allocations to use opaque pointers
//===----------------------------------------------------------------------===//
void DbLoweringPass::runOnOperation() {
  module = getOperation();
  AC = new ArtsCodegen(module, false);
  ARTS_INFO_HEADER(DbLowering);
  ARTS_DEBUG_REGION(module.dump(););
  convertDbAllocOps();
  removeOps(module, opsToRemove);
  ARTS_INFO_FOOTER(DbLoweringPass);
  ARTS_DEBUG_REGION(module.dump(););
}

//===----------------------------------------------------------------------===//
/// Convert all DB allocation operations to use opaque pointers
//===----------------------------------------------------------------------===//
void DbLoweringPass::convertDbAllocOps() {
  SmallVector<DbAllocOp, 8> dbAllocOps;
  module.walk([&](arts::DbAllocOp allocOp) { dbAllocOps.push_back(allocOp); });
  ARTS_INFO("Found " << dbAllocOps.size()
                     << " DB allocation operations to lower");
  if (dbAllocOps.empty())
    return;

  for (auto oldOp : dbAllocOps) {
    ARTS_DEBUG("DbAllocOp: " << oldOp);
    AC->setInsertionPointAfter(oldOp);
    auto oldPtrType = llvm::dyn_cast<MemRefType>(oldOp.getPtr().getType());
    if (!oldPtrType) {
      ARTS_DEBUG("  - Skipping non-memref pointer result");
      continue;
    }

    const bool alreadyOpaque =
        oldPtrType.getElementType().isa<LLVM::LLVMPointerType>();
    const bool alreadyHeap = oldOp.getAllocType() == DbAllocType::heap;
    if (alreadyOpaque && alreadyHeap) {
      ARTS_DEBUG("  - Allocation already lowered; skipping");
      continue;
    }

    /// Create new DbAllocOp with opaque pointer element type
    SmallVector<Value> sizes(oldOp.getSizes().begin(), oldOp.getSizes().end());
    SmallVector<Value> elementSizes(oldOp.getElementSizes().begin(),
                                    oldOp.getElementSizes().end());
    MemRefType allocatedElementType = oldOp.getAllocatedElementType();
    Type elementType = LLVM::LLVMPointerType::get(
        AC->getContext(), allocatedElementType.getMemorySpaceAsInt());
    SmallVector<int64_t> shape;
    shape.assign(sizes.size(), ShapedType::kDynamic);
    auto ptrType = MemRefType::get(shape, elementType);
    DbAllocOp newOp = AC->create<DbAllocOp>(
        oldOp.getLoc(), oldOp.getMode(), oldOp.getRoute(), DbAllocType::heap,
        oldOp.getDbMode(), oldOp.getElementType(), ptrType, sizes,
        elementSizes);
    ARTS_DEBUG("  - New DbAllocOp: " << newOp);
    updateUsers(oldOp, newOp);
    opsToRemove.insert(oldOp);
  }
}

//===----------------------------------------------------------------------===//
/// Update all users of the old allocation to use the new lowered allocation
//===----------------------------------------------------------------------===//
void DbLoweringPass::updateUsers(DbAllocOp oldAllocOp, DbAllocOp newAllocOp) {
  /// Replace all uses of the old GUID with the new GUID.
  Value oldGuid = oldAllocOp.getGuid();
  Value newGuid = newAllocOp.getGuid();
  assert(oldGuid && newGuid && "expected guid values");
  oldGuid.replaceAllUsesWith(newGuid);

  /// Replace all uses of the old pointer with the new pointer.
  Value oldPtr = oldAllocOp.getPtr();
  Value newPtr = newAllocOp.getPtr();
  assert(oldPtr && newPtr && "expected pointer values");

  auto getLLVMPtr = [&](Value base, ValueRange opIndices,
                        Location loc) -> Value {
    if (opIndices.empty())
      return AC->castToLLVMPtr(base, loc);

    SmallVector<Value> sizes = getSizesFromDb(base);
    SmallVector<Value> strides = AC->computeStridesFromSizes(sizes, loc);
    SmallVector<Value> indices(opIndices.begin(), opIndices.end());
    return AC->create<DbGepOp>(loc, AC->getLLVMPointerType(base), base, indices,
                               strides);
  };

  /// Process each use of the old pointer
  for (auto &use : llvm::make_early_inc_range(oldPtr.getUses())) {
    Operation *userOp = use.getOwner();
    AC->setInsertionPoint(userOp);

    if (auto dbRefOp = dyn_cast<DbRefOp>(userOp)) {
      Value llvmPtr =
          getLLVMPtr(newPtr, dbRefOp.getIndices(), dbRefOp.getLoc());
      auto loadOp = AC->create<LLVM::LoadOp>(
          dbRefOp.getLoc(), dbRefOp.getResult().getType(), llvmPtr);
      dbRefOp.getResult().replaceAllUsesWith(loadOp.getResult());
      opsToRemove.insert(dbRefOp);
      continue;
    }

    if (auto loadOp = dyn_cast<memref::LoadOp>(userOp)) {
      Value llvmPtr = getLLVMPtr(newPtr, loadOp.getIndices(), loadOp.getLoc());
      auto newLoad = AC->create<LLVM::LoadOp>(
          loadOp.getLoc(), loadOp.getResult().getType(), llvmPtr);
      loadOp.getResult().replaceAllUsesWith(newLoad.getResult());
      opsToRemove.insert(loadOp);
      continue;
    }

    if (auto storeOp = dyn_cast<memref::StoreOp>(userOp)) {
      Value llvmPtr =
          getLLVMPtr(newPtr, storeOp.getIndices(), storeOp.getLoc());
      AC->create<LLVM::StoreOp>(storeOp.getLoc(), storeOp.getValueToStore(),
                                llvmPtr);
      opsToRemove.insert(storeOp);
      continue;
    }

    if (auto acquireOp = dyn_cast<DbAcquireOp>(userOp)) {
      SmallVector<Value> indices(acquireOp.getIndices().begin(),
                                 acquireOp.getIndices().end());
      SmallVector<Value> offsets(acquireOp.getOffsets().begin(),
                                 acquireOp.getOffsets().end());
      SmallVector<Value> sizes(acquireOp.getSizes().begin(),
                               acquireOp.getSizes().end());
      auto newAcquireOp =
          AC->create<DbAcquireOp>(acquireOp.getLoc(), acquireOp.getMode(),
                                  newGuid, newPtr, indices, offsets, sizes);
      ARTS_DEBUG("  - New DbAcquireOp: " << newAcquireOp);

      /// Find the EDT that uses this acquire op and the corresponding block argument
      auto [edtUser, blockArg] = arts::getEdtBlockArgumentForAcquire(acquireOp);
      assert(edtUser && blockArg && "Expected EDT user with block argument");

      /// Update the block argument type to match the new lowered type
      blockArg.setType(newPtr.getType());

      /// Handle the uses of the block argument within the EDT
      for (auto &blockUse : llvm::make_early_inc_range(blockArg.getUses())) {
        Operation *blockUserOp = blockUse.getOwner();
        AC->setInsertionPoint(blockUserOp);

        if (auto dbRefOp = dyn_cast<DbRefOp>(blockUserOp)) {
          Value llvmPtr =
              getLLVMPtr(blockArg, dbRefOp.getIndices(), dbRefOp.getLoc());
          auto loadOp = AC->create<LLVM::LoadOp>(
              dbRefOp.getLoc(), dbRefOp.getResult().getType(), llvmPtr);
          dbRefOp.getResult().replaceAllUsesWith(loadOp.getResult());
          opsToRemove.insert(dbRefOp);
          continue;
        }

        if (auto loadOp = dyn_cast<memref::LoadOp>(blockUserOp)) {
          Value llvmPtr =
              getLLVMPtr(blockArg, loadOp.getIndices(), loadOp.getLoc());
          auto newLoad = AC->create<LLVM::LoadOp>(
              loadOp.getLoc(), loadOp.getResult().getType(), llvmPtr);
          loadOp.getResult().replaceAllUsesWith(newLoad.getResult());
          opsToRemove.insert(loadOp);
          continue;
        }

        if (auto storeOp = dyn_cast<memref::StoreOp>(blockUserOp)) {
          Value llvmPtr =
              getLLVMPtr(blockArg, storeOp.getIndices(), storeOp.getLoc());
          AC->create<LLVM::StoreOp>(storeOp.getLoc(), storeOp.getValueToStore(),
                                    llvmPtr);
          opsToRemove.insert(storeOp);
          continue;
        }

        if (auto m2p = dyn_cast<polygeist::Memref2PointerOp>(blockUserOp)) {
          Value llvmPtr = AC->castToLLVMPtr(blockArg, m2p.getLoc());
          m2p.getResult().replaceAllUsesWith(llvmPtr);
          opsToRemove.insert(m2p);
          continue;
        }

        if (auto releaseOp = dyn_cast<DbReleaseOp>(blockUserOp)) {
          opsToRemove.insert(releaseOp);
          continue;
        }
      }

      /// Replace the old acquire op with the new one
      acquireOp->replaceAllUsesWith(newAcquireOp);
      opsToRemove.insert(acquireOp);
      continue;
    }

    if (auto m2p = dyn_cast<polygeist::Memref2PointerOp>(userOp)) {
      Value llvmPtr = AC->castToLLVMPtr(newPtr, m2p.getLoc());
      m2p.getResult().replaceAllUsesWith(llvmPtr);
      opsToRemove.insert(m2p);
      continue;
    }

    if (auto freeOp = dyn_cast<DbFreeOp>(userOp)) {
      auto newFree = AC->create<DbFreeOp>(freeOp.getLoc(), newPtr);
      freeOp->replaceAllUsesWith(newFree);
      opsToRemove.insert(freeOp);
      continue;
    }
  }
}
//===----------------------------------------------------------------------===//
// Pass creation
//===----------------------------------------------------------------------===//
namespace mlir {
namespace arts {
std::unique_ptr<Pass> createDbLoweringPass() {
  return std::make_unique<DbLoweringPass>();
}
} // namespace arts
} // namespace mlir
