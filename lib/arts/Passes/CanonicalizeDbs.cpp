///==========================================================================
/// File: CanonicalizeDbs.cpp
///
/// Normalize ARTS datablock handles and their uses to a typed memref form.
/// This pass:
///   - Forces stack-resident handles to become heap-backed so the runtime
///     consistently receives a pointer-based handle.
///   - Rebuilds `arts.db_alloc` so its pointer result preserves the outer
///     datablock shape while its element type is the elementSizes memref
///   - Rewrites users (loads, stores, acquires, GEPs, releases, region args)
///     to first select the datablock, then operate on the elementSizes memref,
///     keeping the computation entirely in typed memref IR for analysis.
///
/// The goal is to keep dimensional information available for DB analyses and
/// optimizations. A later `DbLowering` pass will switch to the opaque
/// pointer representation required by the runtime.
///==========================================================================

#include "ArtsPassDetails.h"
#include "arts/ArtsDialect.h"
#include "arts/Codegen/ArtsCodegen.h"
#include "arts/Passes/ArtsPasses.h"
#include "arts/Utils/ArtsDebug.h"
#include "arts/Utils/ArtsUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

ARTS_DEBUG_SETUP(canonicalize_dbs);

using namespace mlir;
using namespace arts;

namespace {
struct CanonicalizeDbsPass
    : public arts::CanonicalizeDbsBase<CanonicalizeDbsPass> {
  ~CanonicalizeDbsPass() override { delete AC; }
  void runOnOperation() override;

private:
  void canonicalizeDbAllocOps();
  void updateAllocUsers(DbAllocOp oldAllocOp, DbAllocOp newAllocOp);

  ModuleOp module;
  ArtsCodegen *AC = nullptr;
  SetVector<Operation *> opsToRemove;
};
} // namespace

/// Entry point: run the canonicalization on the whole module and clean up.
void CanonicalizeDbsPass::runOnOperation() {
  module = getOperation();
  AC = new ArtsCodegen(module, false);

  ARTS_INFO_HEADER(CanonicalizeDbs);
  ARTS_DEBUG_REGION(module.dump(););

  canonicalizeDbAllocOps();
  removeOps(module, opsToRemove);

  ARTS_INFO_FOOTER(CanonicalizeDbs);
  ARTS_DEBUG_REGION(module.dump(););
}

void CanonicalizeDbsPass::canonicalizeDbAllocOps() {
  SmallVector<DbAllocOp, 8> allocOps;
  module.walk([&](DbAllocOp alloc) { allocOps.push_back(alloc); });
  if (allocOps.empty())
    return;

  ARTS_INFO("Found " << allocOps.size()
                     << " datablock allocation(s) to canonicalize");

  for (DbAllocOp oldOp : allocOps) {
    ARTS_DEBUG("Canonicalizing DbAlloc: " << oldOp);
    AC->setInsertionPointAfter(oldOp);

    /// Old sizes is empty, add a dimension to the sizes
    SmallVector<Value> newSizes;
    SmallVector<Value> oldSizes = getSizesFromDb(oldOp);
    if (oldSizes.empty())
      newSizes.push_back(AC->createIndexConstant(1, oldOp.getLoc()));
    else
      newSizes.assign(oldSizes.begin(), oldSizes.end());

    /// Old element sizes is empty, add a dimension to the element sizes
    SmallVector<Value> newElementSizes;
    SmallVector<Value> oldElementSizes = oldOp.getElementSizes();
    if (oldElementSizes.empty())
      newElementSizes.push_back(AC->createIndexConstant(1, oldOp.getLoc()));
    else
      newElementSizes.assign(oldElementSizes.begin(), oldElementSizes.end());

    /// Route get default value of 0
    Value route = AC->createIntConstant(0, AC->Int32, oldOp.getLoc());

    /// Create the new DbAllocOp
    DbAllocOp newOp = AC->create<DbAllocOp>(
        oldOp.getLoc(), oldOp.getMode(), route, DbAllocType::heap,
        oldOp.getDbMode(), oldOp.getElementType(), oldOp.getAddress(), newSizes,
        newElementSizes);
    ARTS_DEBUG("  New DbAllocOp: " << newOp);

    /// Update the users of the old DbAllocOp
    updateAllocUsers(oldOp, newOp);
    opsToRemove.insert(oldOp);
  }
}

/// This function transforms all uses of the old stack-based datablock
/// allocation to use the new heap-based memref-of-memrefs representation.
void CanonicalizeDbsPass::updateAllocUsers(DbAllocOp oldAllocOp,
                                           DbAllocOp newAllocOp) {
  /// Phase 1: Replace GUID uses - these are straightforward
  Value oldGuid = oldAllocOp.getGuid();
  Value newGuid = newAllocOp.getGuid();
  oldGuid.replaceAllUsesWith(newGuid);

  /// Phase 2: Replace pointer and transform its uses
  Value oldPtr = oldAllocOp.getPtr();
  Value newPtr = newAllocOp.getPtr();

  /// Worklist of pointers to process
  auto elementSizes = getElementSizesFromDb(oldAllocOp);
  auto elementMemRefType =
      newAllocOp.getAllocatedElementType().getElementType();

  /// Process each use of the old pointer
  for (auto &use : llvm::make_early_inc_range(oldPtr.getUses())) {
    Operation *userOp = use.getOwner();
    AC->setInsertionPoint(userOp);

    if (auto loadOp = dyn_cast<memref::LoadOp>(userOp)) {
      SmallVector<Value> indices(loadOp.getIndices().begin(),
                                 loadOp.getIndices().end());
      auto [outerIdx, innerIdx] = splitDbIndices(
          oldAllocOp, indices, AC->getBuilder(), loadOp.getLoc());
      auto newDbRef = AC->create<DbRefOp>(loadOp.getLoc(), elementMemRefType,
                                          newPtr, outerIdx);
      auto newLoad =
          AC->create<memref::LoadOp>(loadOp.getLoc(), newDbRef, innerIdx);
      loadOp.replaceAllUsesWith(newLoad.getResult());
      opsToRemove.insert(loadOp);
      continue;
    }

    if (auto storeOp = dyn_cast<memref::StoreOp>(userOp)) {
      SmallVector<Value> indices(storeOp.getIndices().begin(),
                                 storeOp.getIndices().end());
      auto [outerIdx, innerIdx] = splitDbIndices(
          oldAllocOp, indices, AC->getBuilder(), storeOp.getLoc());
      auto newDbRef = AC->create<DbRefOp>(storeOp.getLoc(), elementMemRefType,
                                          newPtr, outerIdx);
      AC->create<memref::StoreOp>(storeOp.getLoc(), storeOp.getValueToStore(),
                                  newDbRef, innerIdx);
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
      /// Find the EDT that uses this acquire op and the corresponding block
      /// argument
      auto [edtUser, blockArg] = arts::getEdtBlockArgumentForAcquire(acquireOp);
      assert(edtUser && blockArg && "Expected EDT user with block argument");

      /// Update the block argument type to match the new canonicalized database
      /// type
      blockArg.setType(newPtr.getType());

      /// Now handle the uses of the block argument within the EDT
      for (auto &use : llvm::make_early_inc_range(blockArg.getUses())) {
        Operation *userOp = use.getOwner();
        AC->setInsertionPoint(userOp);

        if (auto loadOp = dyn_cast<memref::LoadOp>(userOp)) {
          SmallVector<Value> indices(loadOp.getIndices().begin(),
                                     loadOp.getIndices().end());
          auto [outerIdx, innerIdx] = splitDbIndices(
              acquireOp, indices, AC->getBuilder(), loadOp.getLoc());
          auto newDbRef = AC->create<DbRefOp>(
              loadOp.getLoc(), elementMemRefType, blockArg, outerIdx);
          auto newLoad =
              AC->create<memref::LoadOp>(loadOp.getLoc(), newDbRef, innerIdx);
          loadOp.replaceAllUsesWith(newLoad.getResult());
          opsToRemove.insert(loadOp);
          continue;
        }

        if (auto storeOp = dyn_cast<memref::StoreOp>(userOp)) {
          SmallVector<Value> indices(storeOp.getIndices().begin(),
                                     storeOp.getIndices().end());
          auto [outerIdx, innerIdx] = splitDbIndices(
              acquireOp, indices, AC->getBuilder(), storeOp.getLoc());
          auto newDbRef = AC->create<DbRefOp>(
              storeOp.getLoc(), elementMemRefType, blockArg, outerIdx);
          AC->create<memref::StoreOp>(
              storeOp.getLoc(), storeOp.getValueToStore(), newDbRef, innerIdx);
          opsToRemove.insert(storeOp);
          continue;
        }

        if (auto m2p = dyn_cast<polygeist::Memref2PointerOp>(userOp)) {
          SmallVector<Value> indices;
          indices.push_back(AC->createIndexConstant(0, m2p.getLoc()));
          auto newDbRef = AC->create<DbRefOp>(m2p.getLoc(), elementMemRefType,
                                              blockArg, indices);
          auto newM2p = AC->castToLLVMPtr(newDbRef, m2p.getLoc());
          m2p.getResult().replaceAllUsesWith(newM2p);
          opsToRemove.insert(m2p);
          continue;
        }

        if (auto releaseOp = dyn_cast<DbReleaseOp>(userOp)) {
          AC->create<DbReleaseOp>(releaseOp.getLoc(), blockArg);
          opsToRemove.insert(releaseOp);
          continue;
        }
      }

      /// Let's replace the old acquire op with the new one
      acquireOp->replaceAllUsesWith(newAcquireOp);
      opsToRemove.insert(acquireOp);
      continue;
    }

    if (auto m2p = dyn_cast<polygeist::Memref2PointerOp>(userOp)) {
      SmallVector<Value> indices;
      indices.push_back(AC->createIndexConstant(0, m2p.getLoc()));
      auto newDbRef =
          AC->create<DbRefOp>(m2p.getLoc(), elementMemRefType, newPtr, indices);
      auto newM2p = AC->castToLLVMPtr(newDbRef, m2p.getLoc());
      m2p.getResult().replaceAllUsesWith(newM2p);
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

  /// For all other uses, DbRefOp is used to select the element memref
  AC->setInsertionPointAfter(newAllocOp);
  SmallVector<Value> indices(newAllocOp.getSizes().size());
  std::generate(indices.begin(), indices.end(), [&]() {
    return AC->createIndexConstant(0, newAllocOp.getLoc());
  });
  auto newDbRef = AC->create<DbRefOp>(newAllocOp.getLoc(), elementMemRefType,
                                      newPtr, indices);
  oldPtr.replaceAllUsesWith(newDbRef);
}

namespace mlir {
namespace arts {
std::unique_ptr<Pass> createCanonicalizeDbsPass() {
  return std::make_unique<CanonicalizeDbsPass>();
}
} // namespace arts
} // namespace mlir
