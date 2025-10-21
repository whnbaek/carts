///==========================================================================
// File: ConvertArtsToLLVM.cpp
//
// This file implements a pass to convert ARTS dialect operations into
// LLVM dialect operations. Since preprocessing is handled by a separate
// pass, this pass focuses purely on ARTS-specific conversion logic.
///==========================================================================

/// Dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "polygeist/Ops.h"
/// Arts
#include "ArtsPassDetails.h"
#include "arts/ArtsDialect.h"
#include "arts/Codegen/ArtsCodegen.h"
#include "arts/Passes/ArtsPasses.h"
/// Others
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

/// Debug
#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(convert_arts_to_llvm);

using namespace mlir;
using namespace arts;

//===----------------------------------------------------------------------===//
// Constants and Configuration
//===----------------------------------------------------------------------===//
namespace {
/// Configuration constants
} // namespace

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Base class for ARTS to LLVM conversion patterns with shared ArtsCodegen
template <typename OpType>
class ArtsToLLVMPattern : public OpRewritePattern<OpType> {
protected:
  ArtsCodegen *AC;

public:
  ArtsToLLVMPattern(MLIRContext *context, ArtsCodegen *ac)
      : OpRewritePattern<OpType>(context), AC(ac) {}
};

//===----------------------------------------------------------------------===//
// Helpers for DB sizes and single-element detection
//===----------------------------------------------------------------------===//
namespace {
template <typename OpType>
static inline void getDbInfo(OpType op, SmallVector<Value> &sizesOut,
                             SmallVector<Value> &offsetsOut,
                             SmallVector<Value> &indicesOut,
                             bool &isSingleElement) {
  sizesOut.assign(op.getSizes().begin(), op.getSizes().end());

  if (auto acqOp = dyn_cast<DbAcquireOp>(op.getOperation())) {
    offsetsOut.assign(acqOp.getOffsets().begin(), acqOp.getOffsets().end());
    indicesOut.assign(acqOp.getIndices().begin(), acqOp.getIndices().end());
  } else {
    offsetsOut.clear();
    indicesOut.clear();
  }

  isSingleElement = false;
  if (sizesOut.empty()) {
    isSingleElement = true;
    return;
  }
  if (sizesOut.size() == 1) {
    if (auto constTotal = sizesOut[0].getDefiningOp<arith::ConstantIndexOp>())
      isSingleElement = (constTotal.value() == 1);
  }
}
} // namespace

/// Pattern to convert arts.get_total_workers operations
struct GetTotalWorkersPattern : public ArtsToLLVMPattern<GetTotalWorkersOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(GetTotalWorkersOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering GetTotalWorkers Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto result = AC->getTotalWorkers(op.getLoc());
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Pattern to convert arts.get_total_nodes operations
struct GetTotalNodesPattern : public ArtsToLLVMPattern<GetTotalNodesOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(GetTotalNodesOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering GetTotalNodes Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto result = AC->getTotalNodes(op.getLoc());
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Pattern to convert arts.get_current_worker operations
struct GetCurrentWorkerPattern : public ArtsToLLVMPattern<GetCurrentWorkerOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(GetCurrentWorkerOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering GetCurrentWorker Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto result = AC->getCurrentWorker(op.getLoc());
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Pattern to convert arts.get_current_node operations
struct GetCurrentNodePattern : public ArtsToLLVMPattern<GetCurrentNodeOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(GetCurrentNodeOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering GetCurrentNode Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto result = AC->getCurrentNode(op.getLoc());
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Event and Barrier Patterns
//===----------------------------------------------------------------------===//

/// Pattern to convert arts.barrier operations
struct BarrierPattern : public ArtsToLLVMPattern<BarrierOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(BarrierOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering Barrier Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    AC->createRuntimeCall(types::ARTSRTL_artsYield, {}, op.getLoc());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to convert arts.atomic_add operations
struct AtomicAddPattern : public ArtsToLLVMPattern<AtomicAddOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(AtomicAddOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering AtomicAdd Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);

    auto loc = op.getLoc();
    auto addr = op.getAddr();
    auto value = op.getValue();

    // Convert memref to LLVM pointer
    Value llvmPtr = rewriter.create<polygeist::Memref2PointerOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getContext()), addr);

    // Create LLVM atomic add operation
    rewriter.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::add, llvmPtr,
                                       value, LLVM::AtomicOrdering::seq_cst);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to convert arts.record_dep operations
struct RecordDepPattern : public ArtsToLLVMPattern<RecordDepOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(RecordDepOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering RecordInDep Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto edtGuid = op.getEdtGuid();
    auto loc = op.getLoc();

    /// Create shared slot counter for all dependencies
    auto slotTy = MemRefType::get({}, AC->Int32);
    Value sharedSlotAlloc = AC->create<memref::AllocaOp>(loc, slotTy);
    Value zeroI32 = AC->createIntConstant(0, AC->Int32, loc);
    AC->create<memref::StoreOp>(loc, zeroI32, sharedSlotAlloc);

    /// Add dependencies for each datablock using shared slot counter
    for (Value dbGuid : op.getDatablocks())
      recordDepsForDb(dbGuid, edtGuid, sharedSlotAlloc, loc);

    rewriter.eraseOp(op);
    return success();
  }

private:
  void recordDepsForDb(Value dbGuid, Value edtGuid, Value sharedSlotAlloc,
                       Location loc) const {
    auto dbAcquireOp = dbGuid.getDefiningOp<DbAcquireOp>();
    assert(dbAcquireOp && "DbAcquireOp not found");

    SmallVector<Value> dbSizes, dbOffsets, dbIndices;
    bool isSingle = false;
    getDbInfo(dbAcquireOp, dbSizes, dbOffsets, dbIndices, isSingle);

    /// Use shared slot counter and iterate over all db elements
    AC->iterateDbElements(dbGuid, edtGuid, dbSizes, dbOffsets, isSingle, loc,
                          [&](Value linearIndex) {
                            recordSingleDb(dbGuid, edtGuid, sharedSlotAlloc,
                                           linearIndex, loc);
                          });
  }

  void recordSingleDb(Value dbGuid, Value edtGuid, Value slotAlloc,
                      Value linearIndex, Location loc) const {
    auto dbGuidValue =
        AC->create<memref::LoadOp>(loc, dbGuid, ValueRange{linearIndex});

    Value edtGuidValue = edtGuid;
    if (auto mt = edtGuid.getType().dyn_cast<MemRefType>()) {
      auto zeroIndex = AC->createIndexConstant(0, loc);
      edtGuidValue =
          AC->create<memref::LoadOp>(loc, edtGuid, ValueRange{zeroIndex});
    }
    edtGuidValue = AC->castToInt(AC->Int64, edtGuidValue, loc);

    auto currentSlotI32 = AC->create<memref::LoadOp>(loc, slotAlloc);
    AC->createRuntimeCall(types::ARTSRTL_artsDbAddDependence,
                          {dbGuidValue, edtGuidValue, currentSlotI32}, loc);

    /// Increment the shared slot counter for next dependency
    auto oneI32 = AC->createIntConstant(1, AC->Int32, loc);
    auto incrementedSlot =
        AC->create<arith::AddIOp>(loc, currentSlotI32, oneI32);
    AC->create<memref::StoreOp>(loc, incrementedSlot, slotAlloc);
  }
};

/// Pattern to convert arts.increment_dep operations
struct IncrementDepPattern : public ArtsToLLVMPattern<IncrementDepOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(IncrementDepOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering IncrementOutLatch Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto loc = op.getLoc();

    /// Create shared slot counter for all latch increments
    auto slotTy = MemRefType::get({}, AC->Int32);
    Value sharedSlotAlloc = AC->create<memref::AllocaOp>(loc, slotTy);
    Value zeroI32 = AC->createIntConstant(0, AC->Int32, loc);
    AC->create<memref::StoreOp>(loc, zeroI32, sharedSlotAlloc);

    /// Increment latch for each datablock using shared slot counter
    for (Value dbGuid : op.getDatablocks())
      incLatchForDb(dbGuid, sharedSlotAlloc, loc);

    rewriter.eraseOp(op);
    return success();
  }

private:
  void incLatchForDb(Value dbGuid, Value sharedSlotAlloc, Location loc) const {
    auto dbAcquireOp = dbGuid.getDefiningOp<DbAcquireOp>();
    assert(dbAcquireOp && "DbAcquireOp not found");

    SmallVector<Value> dbSizes, dbOffsets, dbIndices;
    bool isSingle = false;
    getDbInfo(dbAcquireOp, dbSizes, dbOffsets, dbIndices, isSingle);

    /// Use shared slot counter and iterate over all db elements
    AC->iterateDbElements(dbGuid, Value(), dbSizes, dbOffsets, isSingle, loc,
                          [&](Value linearIndex) {
                            incSingleLatch(dbGuid, sharedSlotAlloc, linearIndex,
                                           loc);
                          });
  }

  void incSingleLatch(Value dbGuid, Value slotAlloc, Value linearIndex,
                      Location loc) const {
    auto dbGuidValue =
        AC->create<memref::LoadOp>(loc, dbGuid, ValueRange{linearIndex});
    AC->createRuntimeCall(types::ARTSRTL_artsDbIncrementLatch, {dbGuidValue},
                          loc);

    /// Increment the shared slot counter for next latch increment
    auto currentSlotI32 = AC->create<memref::LoadOp>(loc, slotAlloc);
    auto oneI32 = AC->createIntConstant(1, AC->Int32, loc);
    auto incrementedSlot =
        AC->create<arith::AddIOp>(loc, currentSlotI32, oneI32);
    AC->create<memref::StoreOp>(loc, incrementedSlot, slotAlloc);
  }
};

//===----------------------------------------------------------------------===//
// EDT  Patterns
//===----------------------------------------------------------------------===//
/// Pattern to convert arts.edt_param_pack operations
struct EdtParamPackPattern : public ArtsToLLVMPattern<EdtParamPackOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(EdtParamPackOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering EdtParamPack Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);

    auto loc = op.getLoc();
    auto params = op.getParams();
    auto resultType = op.getMemref().getType().dyn_cast<MemRefType>();
    if (!resultType)
      return op.emitError("Expected MemRef type for result");

    // Check if result type has dynamic dimensions
    bool hasDynamicDims = resultType.getNumDynamicDims() > 0;

    memref::AllocOp allocOp;
    if (hasDynamicDims) {
      /// Dynamic memref: allocate with size and store parameters
      auto numParams = AC->createIndexConstant(params.size(), loc);
      allocOp =
          AC->create<memref::AllocOp>(loc, resultType, ValueRange{numParams});

      for (unsigned i = 0; i < params.size(); ++i) {
        auto index = AC->createIndexConstant(i, loc);
        auto castParam = AC->castParameter(AC->Int64, params[i], loc);
        AC->create<memref::StoreOp>(loc, castParam, allocOp, ValueRange{index});
      }
    } else {
      /// Empty parameter pack: allocate dynamic memref<?xi64> with size 0
      auto dynamicType = MemRefType::get({ShapedType::kDynamic}, AC->Int64);
      auto zeroIndex = AC->createIndexConstant(0, loc);
      allocOp =
          AC->create<memref::AllocOp>(loc, dynamicType, ValueRange{zeroIndex});
    }

    rewriter.replaceOp(op, allocOp);
    return success();
  }
};

/// Pattern to convert arts.edt_param_unpack operations
struct EdtParamUnpackPattern : public ArtsToLLVMPattern<EdtParamUnpackOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(EdtParamUnpackOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering EdtParamUnpack Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto paramMemref = op.getMemref();
    auto results = op.getUnpacked();

    SmallVector<Value> newResults;
    for (unsigned i = 0; i < results.size(); ++i) {
      auto idx = AC->createIndexConstant(i, op.getLoc());
      auto loadedParam =
          AC->create<memref::LoadOp>(op.getLoc(), paramMemref, ValueRange{idx});
      auto castedParam =
          AC->castParameter(results[i].getType(), loadedParam, op.getLoc());
      newResults.push_back(castedParam);
    }

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

/// Pattern to convert arts.dep_gep operations
struct DepGepOpPattern : public ArtsToLLVMPattern<DepGepOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DepGepOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DepGep Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto depStruct = op.getDepStruct();
    auto offset = op.getOffset();
    auto indices = op.getIndices();
    auto strides = op.getStrides();
    auto loc = op.getLoc();

    /// Cast dep_struct to typed pointer to LLVM pointer
    Value typedDepPtr = AC->castToLLVMPtr(depStruct, loc);

    /// Compute linearized index: offset + sum_i indices[i] * strides[i]
    Value linearIndex = offset;
    if (linearIndex.getType() != AC->Int64)
      linearIndex = AC->castToInt(AC->Int64, linearIndex, loc);

    for (size_t i = 0; i < indices.size(); ++i) {
      Value idx = indices[i];
      if (idx.getType() != AC->Int64)
        idx = AC->castToInt(AC->Int64, idx, loc);
      Value strideVal = (i < strides.size())
                            ? strides[i]
                            : AC->createIntConstant(1, AC->Int64, loc);
      if (strideVal.getType() != AC->Int64)
        strideVal = AC->castToInt(AC->Int64, strideVal, loc);

      Value contrib = AC->create<arith::MulIOp>(loc, idx, strideVal);
      linearIndex = AC->create<arith::AddIOp>(loc, linearIndex, contrib);
    }

    Value depEntryPtr = AC->create<LLVM::GEPOp>(
        loc, AC->llvmPtr, AC->ArtsEdtDep, typedDepPtr, ValueRange{linearIndex});

    /// Extract ptr field (field #2) from dependency structure
    auto c0 = AC->createIntConstant(0, AC->Int64, loc);
    auto c2 = AC->createIntConstant(2, AC->Int64, loc);
    auto dataPtr = AC->create<LLVM::GEPOp>(loc, AC->llvmPtr, AC->ArtsEdtDep,
                                           depEntryPtr, ValueRange{c0, c2});

    /// Return pointer to the array element
    rewriter.replaceOp(op, dataPtr);
    return success();
  }
};

/// Pattern to convert arts.db_gep operations to LLVM GEP using element strides
struct DbGepOpPattern : public ArtsToLLVMPattern<arts::DbGepOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(arts::DbGepOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DbGep Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto base = op.getBasePtr();
    auto indices = op.getIndices();
    auto strides = op.getStrides();
    auto loc = op.getLoc();

    /// Cast base to LLVM pointer type
    Value basePtr = AC->castToLLVMPtr(base, loc);

    /// If base is a memref of pointers, we must load the pointer element;
    /// otherwise we compute the address within the element buffer.
    auto baseMT = base.getType().dyn_cast<MemRefType>();

    /// Pad strides with 1s to match indices length
    SmallVector<Value> paddedStrides(strides.begin(), strides.end());
    while (paddedStrides.size() < indices.size())
      paddedStrides.push_back(AC->createIndexConstant(1, loc));

    /// Compute linear element index using provided strides
    Value linearIdx =
        AC->computeLinearIndexFromStrides(paddedStrides, indices, loc);
    Value idx64 = AC->castToInt(AC->Int64, linearIdx, loc);

    /// Use typed GEP based on the element type; default to pointer elements
    Type elemTy = baseMT ? baseMT.getElementType() : AC->llvmPtr;
    Value elemAddr = AC->create<LLVM::GEPOp>(loc, AC->llvmPtr, elemTy, basePtr,
                                             ValueRange{idx64});

    rewriter.replaceOp(op, elemAddr);
    return success();
  }
};

/// Pattern to convert arts.edt_create operations (key EDT creation)
struct EdtCreatePattern : public ArtsToLLVMPattern<EdtCreateOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(EdtCreateOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering EdtCreate Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    /// Get outlined function name
    auto funcNameAttr = op->getAttrOfType<StringAttr>("outlined_func");
    if (!funcNameAttr)
      return op.emitError("Missing outlined_func attribute");

    auto outlined =
        AC->getModule().lookupSymbol<func::FuncOp>(funcNameAttr.getValue());
    if (!outlined)
      return op.emitError("EDT Outlined function not found");

    /// Build artsEdtCreate arguments; route required, default 0 if missing
    auto funcPtr = AC->createFnPtr(outlined, op.getLoc());
    Value route = op.getRoute();
    if (!route)
      route = AC->createIntConstant(0, AC->Int32, op.getLoc());
    auto paramv = op.getParamMemref();
    auto depc = op.getDepCount();

    /// Calculate parameter count from memref size
    Value paramc;
    if (auto memrefType = paramv.getType().dyn_cast<MemRefType>()) {
      if (memrefType.hasStaticShape() && memrefType.getNumElements() == 0) {
        paramc = AC->createIntConstant(0, AC->Int32, op.getLoc());
      } else {
        /// Dynamic memref case - get size from memref
        auto zeroIndex = AC->createIndexConstant(0, op.getLoc());
        auto memrefSize =
            AC->create<memref::DimOp>(op.getLoc(), paramv, zeroIndex);
        paramc =
            AC->create<arith::IndexCastOp>(op.getLoc(), AC->Int32, memrefSize);
      }
    } else {
      paramc = AC->createIntConstant(0, AC->Int32, op.getLoc());
    }

    /// Create artsEdtCreate or artsEdtCreateWithEpoch call based on epoch GUID
    /// availability
    func::CallOp callOp;
    if (op.getEpochGuid()) {
      /// Use artsEdtCreateWithEpoch when epoch GUID is provided
      callOp = AC->createRuntimeCall(
          types::ARTSRTL_artsEdtCreateWithEpoch,
          {funcPtr, route, paramc, paramv, depc, op.getEpochGuid()},
          op.getLoc());
    } else {
      /// Use regular artsEdtCreate when no epoch GUID
      callOp = AC->createRuntimeCall(types::ARTSRTL_artsEdtCreate,
                                     {funcPtr, route, paramc, paramv, depc},
                                     op.getLoc());
    }
    rewriter.replaceOp(op, callOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// DB Patterns
//===----------------------------------------------------------------------===//
/// Pattern to convert arts.db_alloc operations
struct DbAllocPattern : public ArtsToLLVMPattern<DbAllocOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DbAllocOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DbAlloc Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto loc = op.getLoc();
    Value guidMemref, dbMemref;

    auto dbMode = AC->createIntConstant(
        static_cast<int>(op.getDbModeAttr().getValue()), AC->Int32, loc);
    Value route = op.getRoute();
    if (!route)
      route = AC->createIntConstant(0, AC->Int32, loc);
    Value elementSize = AC->computeElementTypeSize(op.getElementType(), loc);

    /// Compute payload size from elementSizes (product of all payload
    /// dimensions)
    Value payloadSize = AC->createIndexConstant(1, loc);
    for (Value payloadDim : op.getElementSizes())
      payloadSize = AC->create<arith::MulIOp>(loc, payloadSize, payloadDim);

    /// Total datablock size = elementSize * payloadSize
    Value totalDbSize =
        AC->create<arith::MulIOp>(loc, elementSize, payloadSize);

    SmallVector<Value> dbSizes, dbOffsets, dbIndices;
    bool isSingleElement = false;
    getDbInfo(op, dbSizes, dbOffsets, dbIndices, isSingleElement);

    if (isSingleElement) {
      ARTS_DEBUG("Creating single DB");
      /// Allocate a single GUID as 1-element array
      auto guidType = MemRefType::get({1}, AC->Int64);
      guidMemref = AC->create<memref::AllocOp>(loc, guidType);
      /// Allocate a single DB Pointer as 1-element array
      auto payloadPtrType = MemRefType::get({1}, AC->llvmPtr);
      dbMemref = AC->create<memref::AllocOp>(loc, payloadPtrType);
      createSingleDb(dbMemref, guidMemref, dbMode, route, totalDbSize, loc);
    } else {
      ARTS_DEBUG("Creating multi-dim DB");
      /// Compute total number of elements
      Value totalElems = AC->computeTotalElements(dbSizes, loc);
      /// Allocate 1D linear array of GUIDs
      auto guidType = MemRefType::get({ShapedType::kDynamic}, AC->Int64);
      guidMemref =
          AC->create<memref::AllocOp>(loc, guidType, ValueRange{totalElems});
      /// Allocate 1D linear array of DB Pointers
      auto payloadPtrType =
          MemRefType::get({ShapedType::kDynamic}, AC->llvmPtr);
      dbMemref = AC->create<memref::AllocOp>(loc, payloadPtrType,
                                             ValueRange{totalElems});
      createMultiDbs(dbMemref, guidMemref, dbSizes, dbMode, route, totalDbSize,
                     loc);
    }

    rewriter.replaceOp(op, {guidMemref, dbMemref});
    return success();
  }

private:
  void createSingleDb(Value dbMemref, Value guidMemref, Value dbMode,
                      Value route, Value elementSize, Location loc,
                      ArrayRef<Value> sizes = {},
                      ArrayRef<Value> indices = {}) const {
    /// Reserve GUID for the DB
    auto guid = AC->createRuntimeCall(types::ARTSRTL_artsReserveGuidRoute,
                                      {dbMode, route}, loc)
                    .getResult(0);
    /// Create the DB with GUID
    auto elemSize64 = AC->castToInt(AC->Int64, elementSize, loc);
    auto dbCall = AC->createRuntimeCall(types::ARTSRTL_artsDbCreateWithGuid,
                                        {guid, elemSize64}, loc);
    auto dbPtr = dbCall.getResult(0);

    /// Store the DB pointer and GUID in the linearized memrefs
    if (indices.empty()) {
      auto zeroIndex = AC->createIndexConstant(0, loc);
      AC->create<memref::StoreOp>(loc, dbPtr, dbMemref, ValueRange{zeroIndex});
      AC->create<memref::StoreOp>(loc, guid, guidMemref, ValueRange{zeroIndex});
    } else {
      auto linearIndex = AC->computeLinearIndex(sizes, indices, loc);
      AC->create<memref::StoreOp>(loc, guid, guidMemref,
                                  ValueRange{linearIndex});
      AC->create<memref::StoreOp>(loc, dbPtr, dbMemref,
                                  ValueRange{linearIndex});
    }
  }

  void createMultiDbs(Value dbMemref, Value guidMemref, ArrayRef<Value> sizes,
                      Value dbMode, Value route, Value elementSize,
                      Location loc) const {
    const unsigned rank = sizes.size();
    auto stepConstant = AC->createIndexConstant(1, loc);

    std::function<void(unsigned, SmallVector<Value, 4> &)> createDbsRecursive =
        [&](unsigned dim, SmallVector<Value, 4> &indices) {
          if (dim == rank) {
            createSingleDb(dbMemref, guidMemref, dbMode, route, elementSize,
                           loc, sizes, indices);
            return;
          }

          /// Create loop for current dimension
          auto lowerBound = AC->createIndexConstant(0, loc);
          auto upperBound = sizes[dim];
          auto loopOp =
              AC->create<scf::ForOp>(loc, lowerBound, upperBound, stepConstant);

          auto &loopBlock = loopOp.getRegion().front();
          AC->setInsertionPointToStart(&loopBlock);
          indices.push_back(loopOp.getInductionVar());
          createDbsRecursive(dim + 1, indices);
          indices.pop_back();
          AC->setInsertionPointAfter(loopOp);
        };

    SmallVector<Value, 4> initIndices;
    createDbsRecursive(0, initIndices);
  }
};

/// Pattern to convert arts.db_acquire operations
struct DbAcquirePattern : public ArtsToLLVMPattern<DbAcquireOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DbAcquireOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DbAcquire Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto loc = op.getLoc();

    Value sourceGuid = op.getSourceGuid();
    Value sourcePtr = op.getSourcePtr();

    /// Get source sizes from the defining op
    auto sourceOp = sourceGuid.getDefiningOp();
    SmallVector<Value> sourceSizes;
    if (auto allocOp = dyn_cast_or_null<DbAllocOp>(sourceOp))
      sourceSizes.assign(allocOp.getSizes().begin(), allocOp.getSizes().end());
    else if (auto acqOp = dyn_cast_or_null<DbAcquireOp>(sourceOp))
      sourceSizes.assign(acqOp.getSizes().begin(), acqOp.getSizes().end());

    if (!sourceSizes.empty()) {
      SmallVector<Value> indices(op.getIndices().begin(),
                                 op.getIndices().end());
      SmallVector<Value> strides =
          AC->computeStridesFromSizes(sourceSizes, loc);
      /// Guid llvm type
      auto guidMT = op.getGuid().getType().dyn_cast<MemRefType>();
      auto llvmGuidType = LLVM::LLVMPointerType::get(
          AC->getContext(), guidMT.getMemorySpaceAsInt());
      auto loadedGuid =
          AC->create<DbGepOp>(loc, llvmGuidType, sourceGuid, indices, strides);
      /// Ptr llvm type
      auto ptrMT = op.getPtr().getType().dyn_cast<MemRefType>();
      auto llvmPtrType = LLVM::LLVMPointerType::get(
          AC->getContext(), ptrMT.getMemorySpaceAsInt());
      auto loadedPtr =
          AC->create<DbGepOp>(loc, llvmPtrType, sourcePtr, indices, strides);
      /// Convert to memref type
      auto guidMemref =
          AC->create<polygeist::Pointer2MemrefOp>(loc, guidMT, loadedGuid);
      auto ptrMemref =
          AC->create<polygeist::Pointer2MemrefOp>(loc, ptrMT, loadedPtr);
      rewriter.replaceOp(op, ValueRange{guidMemref, ptrMemref});
    } else {
      /// Fallback: just forward the source values
      rewriter.replaceOp(op, ValueRange{sourceGuid, sourcePtr});
    }

    return success();
  }
};

/// Pattern to convert arts.db_release operations
struct DbReleasePattern : public ArtsToLLVMPattern<DbReleaseOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DbReleaseOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DbRelease Op " << op);
    /// Simply erase the DbReleaseOp
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to convert arts.db_free operations
struct DbFreePattern : public ArtsToLLVMPattern<DbFreeOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DbFreeOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DbFree Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    AC->create<memref::DeallocOp>(op.getLoc(), op.getSource());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to convert arts.db_control operations
struct DbControlPattern : public ArtsToLLVMPattern<DbControlOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DbControlOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DbControl Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    /// Control DBs are not supported yet
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to convert arts.db_num_elements operations
struct DbNumElementsPattern : public ArtsToLLVMPattern<DbNumElementsOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DbNumElementsOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DbNumElements Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto loc = op.getLoc();

    /// Get the sizes from the operation arguments
    SmallVector<Value> sizes = op.getSizes();

    /// If no sizes are provided, return constant 1 (i32)
    if (sizes.empty()) {
      rewriter.replaceOp(op, AC->createIntConstant(1, AC->Int32, loc));
      return success();
    }

    /// If all sizes are constants, fold to a single i32 constant
    bool allConst = true;
    int64_t folded = 1;
    for (Value sz : sizes) {
      if (auto cst = dyn_cast_or_null<arith::ConstantOp>(sz.getDefiningOp())) {
        if (auto intAttr = cst.getValue().dyn_cast<IntegerAttr>()) {
          folded *= intAttr.getInt();
          continue;
        }
      }
      allConst = false;
      break;
    }
    if (allConst) {
      rewriter.replaceOp(op, AC->createIntConstant(folded, AC->Int32, loc));
      return success();
    }

    /// Otherwise, compute product at runtime as a plain i32 value
    Value productVal = AC->createIntConstant(1, AC->Int32, loc);
    for (Value sz : sizes) {
      Value szI32 = AC->castToInt(AC->Int32, sz, loc);
      productVal = AC->create<arith::MulIOp>(loc, productVal, szI32);
    }

    rewriter.replaceOp(op, productVal);
    return success();
  }
};

/// Pattern to convert arts.alloc operations (ARTS memory allocation)
struct AllocPattern : public ArtsToLLVMPattern<AllocOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(AllocOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering Alloc Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto resultType = op.getResult().getType().dyn_cast<MemRefType>();
    if (!resultType)
      return op.emitError("Expected MemRef type for result");
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Terminator Patterns
//===----------------------------------------------------------------------===//

/// Pattern to convert arts.yield operations (terminator)
struct YieldPattern : public ArtsToLLVMPattern<YieldOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(YieldOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering Yield Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    /// arts.yield is a terminator that should be erased during conversion
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//
namespace {
struct ConvertArtsToLLVMPass
    : public arts::ConvertArtsToLLVMBase<ConvertArtsToLLVMPass> {

  explicit ConvertArtsToLLVMPass(bool debug = false) : debugMode(debug) {}

  void runOnOperation() override;

private:
  LogicalResult initializeCodegen();
  void populateCorePatterns(RewritePatternSet &patterns);
  void populateDbPatterns(RewritePatternSet &patterns);

  //// Member variables
  ArtsCodegen *AC = nullptr;
  bool debugMode = false;
};
} // namespace

//===----------------------------------------------------------------------===//
// Core Pass Implementation
//===----------------------------------------------------------------------===//

void ConvertArtsToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *context = &getContext();

  ARTS_INFO_HEADER(ConvertArtsToLLVMPass);
  ARTS_DEBUG_REGION(module.dump(););

  //// Initialize codegen infrastructure
  if (failed(initializeCodegen())) {
    ARTS_ERROR("Failed to initialize ArtsCodegen");
    return signalPassFailure();
  }

  //// Apply patterns with greedy rewriter (two runs)
  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = true;

  /// Run 1: core patterns
  {
    ARTS_INFO("Running core patterns");
    RewritePatternSet corePatterns(context);
    populateCorePatterns(corePatterns);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(corePatterns),
                                            config))) {
      ARTS_ERROR("Failed to apply core ARTS to LLVM conversion patterns");
      delete AC;
      return signalPassFailure();
    }
  }

  /// Run 2: Db Patterns
  {
    ARTS_INFO("Running db patterns");
    RewritePatternSet dbPatterns(context);
    populateDbPatterns(dbPatterns);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(dbPatterns),
                                            config))) {
      ARTS_ERROR("Failed to apply DbAcquire/DbRelease conversion patterns");
      delete AC;
      return signalPassFailure();
    }
  }

  /// Run 3: Other Patterns
  {
    ARTS_INFO("Running other patterns");
    RewritePatternSet otherPatterns(context);
    otherPatterns.add<DbAllocPattern, DbFreePattern>(context, AC);
    otherPatterns.add<DbNumElementsPattern>(context, AC);
    otherPatterns.add<YieldPattern>(context, AC);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(otherPatterns),
                                            config))) {
      ARTS_ERROR("Failed to apply other conversion patterns");
      delete AC;
      return signalPassFailure();
    }
  }
  //// Initialize runtime
  AC->initRT(AC->getUnknownLoc());

  //// Cleanup
  delete AC;
  AC = nullptr;

  ARTS_INFO_FOOTER(ConvertArtsToLLVMPass);
  ARTS_DEBUG_REGION(module.dump(););
}

LogicalResult ConvertArtsToLLVMPass::initializeCodegen() {
  ModuleOp module = getOperation();
  AC = new ArtsCodegen(module, debugMode);

  ARTS_DEBUG_TYPE("ArtsCodegen initialized successfully");
  return success();
}

void ConvertArtsToLLVMPass::populateCorePatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  /// Runtime helper patterns
  patterns.add<GetTotalWorkersPattern, GetTotalNodesPattern,
               GetCurrentWorkerPattern, GetCurrentNodePattern>(context, AC);

  /// Synchronization patterns
  patterns.add<BarrierPattern, AtomicAddPattern>(context, AC);

  /// EDT patterns
  patterns.add<EdtParamPackPattern, EdtParamUnpackPattern>(context, AC);
  patterns.add<EdtCreatePattern>(context, AC);

  /// Dependency patterns
  patterns.add<DepGepOpPattern>(context, AC);
  patterns.add<RecordDepPattern, IncrementDepPattern>(context, AC);
}

void ConvertArtsToLLVMPass::populateDbPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  /// DB patterns
  patterns.add<DbControlPattern>(context, AC);
  patterns.add<DbAcquirePattern, DbReleasePattern>(context, AC);
  patterns.add<DbGepOpPattern>(context, AC);
}

//===----------------------------------------------------------------------===//
// Pass Functions
//===----------------------------------------------------------------------===//
namespace mlir {
namespace arts {
std::unique_ptr<Pass> createConvertArtsToLLVMPass() {
  return std::make_unique<ConvertArtsToLLVMPass>();
}

std::unique_ptr<Pass> createConvertArtsToLLVMPass(bool debug) {
  return std::make_unique<ConvertArtsToLLVMPass>(debug);
}
} // namespace arts
} // namespace mlir
