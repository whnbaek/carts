///==========================================================================
/// File: ArtsDialect.cpp
/// Defines the Arts dialect and the operations within it.
///==========================================================================
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cassert>

#include "arts/ArtsDialect.h"
#include "arts/Utils/ArtsUtils.h"
#include "polygeist/Ops.h"

using namespace mlir;
using namespace mlir::arts;

//===----------------------------------------------------------------------===//
// Arts dialect.
//===----------------------------------------------------------------------===//
void ArtsDialect::initialize() {
  /// Register operations
  addOperations<
#define GET_OP_LIST
#include "arts/ArtsOps.cpp.inc"
      >();

  /// Register types
  addTypes<
#define GET_TYPEDEF_LIST
#include "arts/ArtsOpsTypes.cpp.inc"
      >();

  /// Register attributes
  addAttributes<
#define GET_ATTRDEF_LIST
#include "arts/ArtsOpsAttributes.cpp.inc"
      >();
}

#include "arts/ArtsOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Arts Dialect Operations - method definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "arts/ArtsOps.cpp.inc"

namespace {
struct FoldDbDimFromDbOps : public OpRewritePattern<DbDimOp> {
  using OpRewritePattern<DbDimOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(DbDimOp op,
                                PatternRewriter &rewriter) const override {
    auto cstIdx = op.getDim().getDefiningOp<arith::ConstantIndexOp>();
    if (!cstIdx)
      return failure();
    int64_t idx = cstIdx.getValue().cast<IntegerAttr>().getInt();

    if (auto dbAlloc = op.getSource().getDefiningOp<DbAllocOp>()) {
      auto sizes = dbAlloc.getSizes();
      if ((long long)sizes.size() > idx) {
        rewriter.replaceOp(op, sizes[idx]);
        return success();
      }
    }
    if (auto dbAcq = op.getSource().getDefiningOp<DbAcquireOp>()) {
      auto sizes = dbAcq.getSizes();
      if ((long long)sizes.size() > idx) {
        rewriter.replaceOp(op, sizes[idx]);
        return success();
      }
    }
    /// Fallback: query with memref.dim if source is not a DB op
    rewriter.replaceOpWithNewOp<memref::DimOp>(op, op.getSource(), idx);
    return success();
  }
};
} // namespace

void DbDimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<FoldDbDimFromDbOps>(context);
}

bool isArtsRegion(Operation *op) { return isa<EdtOp>(op) || isa<EpochOp>(op); }
bool isArtsOp(Operation *op) {
  return isArtsRegion(op) ||
         isa<arts::EdtOp, arts::EpochOp, arts::BarrierOp, arts::AllocOp,
             arts::DbAllocOp, arts::DbAcquireOp, arts::DbReleaseOp,
             arts::DbFreeOp, arts::DbControlOp, arts::GetTotalWorkersOp,
             arts::GetTotalNodesOp, arts::GetCurrentWorkerOp,
             arts::GetCurrentNodeOp>(op);
}

//===----------------------------------------------------------------------===//
// Arts Dialect Types - method definitions
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "arts/ArtsOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Arts Dialect Attributes - method definitions
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "arts/ArtsOpsAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// Arts Dialect Enums - method definitions
//===----------------------------------------------------------------------===//
#include "arts/ArtsOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// UndefOp
//===----------------------------------------------------------------------===//
class UndefToLLVM final : public OpRewritePattern<UndefOp> {
public:
  using OpRewritePattern<UndefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UndefOp uop,
                                PatternRewriter &rewriter) const override {
    auto ty = uop.getResult().getType();
    if (!LLVM::isCompatibleType(ty))
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::UndefOp>(uop, ty);
    return success();
  }
};

void UndefOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<UndefToLLVM>(context);
}

//===----------------------------------------------------------------------===//
// EdtOp
//===----------------------------------------------------------------------===//
SmallVector<Value> mlir::arts::EdtOp::getDependenciesAsVector() {
  SmallVector<Value> deps(getDependencies().begin(), getDependencies().end());
  return deps;
}

LogicalResult EdtOp::verify() {
  /// Skip verification if no_verify attribute is present
  if (getNoVerifyAttr())
    return success();

  Block &block = getBody().front();
  auto blockArgs = block.getArguments();
  auto deps = getDependenciesAsVector();

  /// Check that dependencies and block arguments match
  if (blockArgs.size() != deps.size()) {
    return emitOpError("block arguments (")
           << blockArgs.size() << ") must match dependencies (" << deps.size()
           << ")";
  }

  /// Collect all values defined outside the EDT region
  DenseSet<Value> externalValues;
  for (Value dep : deps)
    externalValues.insert(dep);

  /// Walk all operations in the EDT body
  auto walkResult = getBody().walk([&](Operation *op) {
    /// Skip the block itself
    if (op == &block.front())
      return WalkResult::advance();

    /// Check each operand
    for (Value operand : op->getOperands()) {
      /// Skip if operand is defined inside the region
      if (operand.getParentRegion() == &getBody())
        continue;

      /// Skip block arguments - these are allowed
      if (llvm::is_contained(blockArgs, operand))
        continue;

      /// Check if this is a dependency value used directly
      if (externalValues.contains(operand)) {
        op->emitOpError("EDT region uses external dependency value '")
            << operand << "' directly instead of block argument. "
            << "All dependency accesses must go through block arguments.";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return failure();

  return success();
}

void EdtOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  /// Collect all effects from the dependencies using safe method
  auto dependencies = getDependenciesAsVector();

  /// If no dependencies, assume it has effects on all the memrefs
  /// and add read/write effects
  if (dependencies.empty()) {
    effects.emplace_back(MemoryEffects::Read::get(),
                         ::mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(),
                         ::mlir::SideEffects::DefaultResource::get());
    return;
  }

  /// Otherwise, collect effects from each dependency
  /// for (auto dep : dependencies) {
  /// TODO: Implement getEffects for DbDepOp
  /// if (auto dbOp = dep.getDefiningOp<DbDepOp>())
  ///   dbOp.getEffects(effects);
  /// }
}

//===----------------------------------------------------------------------===//
// ARTS Operation Builders
//===----------------------------------------------------------------------===//
void DbReleaseOp::build(OpBuilder &builder, OperationState &state,
                        Value source) {
  state.addOperands(source);
}

void DbDimOp::build(OpBuilder &builder, OperationState &state, Value source,
                    int64_t dim) {
  state.addOperands(source);
  auto c = builder.create<arith::ConstantIndexOp>(state.location, dim);
  state.addOperands(c.getResult());
  state.addTypes(builder.getIndexType());
}

void DbGepOp::build(OpBuilder &builder, OperationState &state, Type ptr,
                    Value base_ptr, SmallVector<Value> indices,
                    SmallVector<Value> strides) {
  state.addOperands(base_ptr);
  state.addOperands(indices);
  state.addOperands(strides);
  /// Set operand segment sizes: [base_ptr, indices..., strides...]
  SmallVector<int32_t, 3> segments = {1, (int32_t)indices.size(),
                                      (int32_t)strides.size()};
  state.addAttribute(getOperandSegmentSizesAttrName(state.name),
                     builder.getDenseI32ArrayAttr(segments));
  state.addTypes(ptr);
}

void DbControlOp::build(OpBuilder &builder, OperationState &state,
                        ArtsMode mode, Value ptr,
                        SmallVector<Value> pinnedIndices) {
  auto ptrOp = ptr.getDefiningOp();
  assert(ptrOp && "Input must be a defining operation.");

  /// If the defining op is a load op, obtain the base memref and pinned
  /// indices.
  Value baseMemRef;
  auto processLoad = [&](auto loadOp) {
    baseMemRef = loadOp.getMemref();
    if (pinnedIndices.empty())
      pinnedIndices.assign(loadOp.getIndices().begin(),
                           loadOp.getIndices().end());
  };

  if (auto loadOp = dyn_cast<memref::LoadOp>(ptrOp))
    processLoad(loadOp);
  else
    baseMemRef = ptr;

  /// Ensure the base memref type.
  auto baseType = baseMemRef.getType().dyn_cast<MemRefType>();
  assert(baseType && "Input must be a MemRefType.");

  int64_t rank = baseType.getRank();
  const auto pinnedCount = static_cast<int64_t>(pinnedIndices.size());
  assert(pinnedCount <= rank &&
         "Pinned indices exceed the rank of the memref.");

  /// Compute sizes and subshape.
  SmallVector<Value, 4> indices(pinnedCount), sizes(rank - pinnedCount);
  SmallVector<Value, 4> offsets(
      rank - pinnedCount,
      builder.create<arith::ConstantIndexOp>(state.location, 0));
  SmallVector<int64_t, 4> subShape;
  for (int64_t i = 0, j = 0; i < rank; ++i) {
    if (i < pinnedCount) {
      indices[i] = pinnedIndices[i];
    } else {
      bool isDyn = baseType.isDynamicDim(i);
      int64_t dimSize = baseType.getDimSize(i);
      Value dimVal =
          isDyn
              ? (Value)builder.create<arts::DbDimOp>(state.location, baseMemRef,
                                                     (int64_t)i)
              : builder.create<arith::ConstantIndexOp>(state.location, dimSize)
                    .getResult();
      sizes[j] = dimVal;
      /// For the normal subview type, preserve the static dim if available.
      subShape.push_back(isDyn ? ShapedType::kDynamic : dimSize);
      j++;
    }
  }

  /// Compute the element type size.
  auto elementType = baseType.getElementType();
  auto elementTypeSize =
      builder
          .create<polygeist::TypeSizeOp>(state.location, builder.getIndexType(),
                                         elementType)
          .getResult();

  state.addAttribute("mode", ArtsModeAttr::get(builder.getContext(), mode));
  state.addOperands(baseMemRef);
  state.addAttribute("elementType", TypeAttr::get(elementType));
  state.addOperands(elementTypeSize);
  state.addOperands(indices);
  state.addOperands(offsets);
  state.addOperands(sizes);

  /// Build operand segment sizes
  state.addAttribute(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr({1, /// ptr
                                    1, /// elementTypeSize
                                    static_cast<int32_t>(indices.size()),
                                    static_cast<int32_t>(offsets.size()),
                                    static_cast<int32_t>(sizes.size())}));

  auto resultType = MemRefType::get(subShape, elementType);
  state.addTypes(resultType);
}

//===----------------------------------------------------------------------===//
// EDT Ops
//===----------------------------------------------------------------------===//
void EdtOp::build(OpBuilder &builder, OperationState &state, EdtType type,
                  EdtConcurrency concurrency) {
  build(builder, state, type, concurrency, ValueRange{});
}

void EdtOp::build(OpBuilder &builder, OperationState &state, EdtType type,
                  EdtConcurrency concurrency, ValueRange dependencies) {
  Value route = builder.create<arith::ConstantIntOp>(state.location, 0, 32);
  build(builder, state, type, concurrency, route, dependencies);
}

void EdtOp::build(OpBuilder &builder, OperationState &state, EdtType type,
                  EdtConcurrency concurrency, Value route,
                  ValueRange dependencies) {
  state.addAttribute("type", EdtTypeAttr::get(builder.getContext(), type));
  state.addAttribute("concurrency", EdtConcurrencyAttr::get(
                                        builder.getContext(), concurrency));
  state.addOperands(route);
  state.addOperands(dependencies);

  /// Create the region with a block
  Region *bodyRegion = state.addRegion();
  Block *bodyBlock = new Block();
  bodyRegion->push_back(bodyBlock);
}

//===----------------------------------------------------------------------===//
// EDT Create Ops
//===----------------------------------------------------------------------===//

void EdtCreateOp::build(OpBuilder &builder, OperationState &state,
                        Value param_memref, Value depCount) {
  Value route = builder.create<arith::ConstantIntOp>(state.location, 0, 32);
  state.addTypes(builder.getI64Type());
  state.addOperands({param_memref, depCount, route});
}

void EdtCreateOp::build(OpBuilder &builder, OperationState &state,
                        Value param_memref, Value depCount, Value route) {
  state.addTypes(builder.getI64Type());
  state.addOperands({param_memref, depCount, route});
}

//===----------------------------------------------------------------------===//
// DB operations builders
//===----------------------------------------------------------------------===//
/// DbAllocOp builders
static void
buildDbAllocOpCommon(OpBuilder &builder, OperationState &state, ArtsMode mode,
                     Value route, DbAllocType allocType, DbMode dbMode,
                     Type elementType, Value address, SmallVector<Value> sizes,
                     SmallVector<Value> elementSizes, Type pointerType) {
  /// Auto-compute GUID type to match datablock dimensionality
  /// 0 or 1 size -> memref<?xi64>, n sizes -> memref<?x?x...xi64>
  SmallVector<int64_t> guidShape;
  size_t numDims = std::max(sizes.size(), size_t(1));
  for (size_t i = 0; i < numDims; ++i)
    guidShape.push_back(ShapedType::kDynamic);
  Type guidType = MemRefType::get(guidShape, builder.getI64Type());

  /// Build ptr type from address shape (if any) or dynamic sizes
  Type pointerElementType;
  if (elementSizes.empty())
    pointerElementType = elementType;
  else
    pointerElementType = arts::getElementMemRefType(elementType, elementSizes);

  Type ptrType;
  if (pointerType) {
    ptrType = pointerType;
  } else {
    SmallVector<int64_t> shape;
    shape.assign(sizes.size(), ShapedType::kDynamic);
    ptrType = MemRefType::get(shape, MemRefType::get({}, pointerElementType));
  }
  state.addTypes({guidType, ptrType});

  /// Add attributes
  ArtsModeAttr modeAttr = ArtsModeAttr::get(builder.getContext(), mode);
  DbAllocTypeAttr allocTypeAttr =
      DbAllocTypeAttr::get(builder.getContext(), allocType);
  DbModeAttr dbModeAttr = DbModeAttr::get(builder.getContext(), dbMode);
  TypeAttr elementTypeAttr = TypeAttr::get(elementType);

  state.addAttribute("mode", modeAttr);
  state.addAttribute("allocType", allocTypeAttr);
  state.addAttribute("dbMode", dbModeAttr);
  state.addAttribute("elementType", elementTypeAttr);

  /// Add operands
  state.addOperands(route);
  if (address)
    state.addOperands(address);
  state.addOperands(sizes);
  state.addOperands(elementSizes);

  /// Build operand segment sizes: [route=1, address(0/1), sizes,
  /// elementSizes]
  state.addAttribute("operandSegmentSizes",
                     builder.getDenseI32ArrayAttr(
                         {1, static_cast<int32_t>(address ? 1 : 0),
                          static_cast<int32_t>(sizes.size()),
                          static_cast<int32_t>(elementSizes.size())}));
}

void DbAllocOp::build(OpBuilder &builder, OperationState &state, ArtsMode mode,
                      Value route, DbAllocType allocType, DbMode dbMode,
                      Type elementType, Value address, SmallVector<Value> sizes,
                      SmallVector<Value> elementSizes) {
  buildDbAllocOpCommon(builder, state, mode, route, allocType, dbMode,
                       elementType, address, sizes, elementSizes, nullptr);
}

void DbAllocOp::build(OpBuilder &builder, OperationState &state, ArtsMode mode,
                      Value route, DbAllocType allocType, DbMode dbMode,
                      Type elementType, SmallVector<Value> sizes,
                      SmallVector<Value> elementSizes) {
  buildDbAllocOpCommon(builder, state, mode, route, allocType, dbMode,
                       elementType, Value{}, sizes, elementSizes, nullptr);
}

void DbAllocOp::build(OpBuilder &builder, OperationState &state, ArtsMode mode,
                      Value route, DbAllocType allocType, DbMode dbMode,
                      Type elementType, Type pointerType,
                      SmallVector<Value> sizes,
                      SmallVector<Value> elementSizes) {
  buildDbAllocOpCommon(builder, state, mode, route, allocType, dbMode,
                       elementType, Value{}, sizes, elementSizes, pointerType);
}

MemRefType DbAllocOp::getAllocatedElementType() {
  return MemRefType::get(
      {}, arts::getElementMemRefType(
              getElementType(), SmallVector<Value>(getElementSizes().begin(),
                                                   getElementSizes().end())));
}

/// DbAcquireOp builders
void DbAcquireOp::build(OpBuilder &builder, OperationState &state,
                        ArtsMode mode, Value sourceGuid, Value sourcePtr,
                        SmallVector<Value> indices, SmallVector<Value> offsets,
                        SmallVector<Value> sizes) {
  auto sourceMemRefType = llvm::cast<MemRefType>(sourcePtr.getType());
  const int64_t sourceRank = sourceMemRefType.getRank();
  const int64_t pinnedDims = static_cast<int64_t>(indices.size());
  assert(pinnedDims <= sourceRank && "Number of indices exceeds source rank");
  const int64_t remainingRank = sourceRank - pinnedDims;

  /// If sizes are not provided, infer sizes for remaining dimensions from
  /// source
  if (sizes.empty()) {
    sizes.reserve(remainingRank);
    for (int64_t d = 0; d < remainingRank; ++d) {
      int64_t srcDim = pinnedDims + d;
      if (sourceMemRefType.isDynamicDim(srcDim)) {
        sizes.push_back(builder.create<arts::DbDimOp>(state.location, sourcePtr,
                                                      (int64_t)srcDim));
      } else {
        sizes.push_back(
            builder
                .create<arith::ConstantIndexOp>(
                    state.location, sourceMemRefType.getDimSize(srcDim))
                .getResult());
      }
    }
  }

  /// Ensure offsets are present for remaining dimensions when omitted
  if (offsets.empty() && remainingRank > 0) {
    offsets.reserve(remainingRank);
    for (int64_t d = 0; d < remainingRank; ++d)
      offsets.push_back(
          builder.create<arith::ConstantIndexOp>(state.location, 0));
  }

  /// GUID type uses dimensionality equal to number of sizes (at least 1)
  SmallVector<int64_t> guidShape;
  size_t numGuidDims = std::max(sizes.size(), size_t(1));
  guidShape.assign(numGuidDims, ShapedType::kDynamic);
  Type guidType = MemRefType::get(guidShape, builder.getI64Type());

  /// Result ptr type: when sizes are provided, use dynamic dimensions
  /// since the actual sizes are determined at runtime
  SmallVector<int64_t> subShape;
  subShape.reserve(remainingRank);
  if (!sizes.empty()) {
    /// When explicit sizes are provided, result has dynamic dimensions
    subShape.assign(remainingRank, ShapedType::kDynamic);
  } else {
    /// When no sizes provided, preserve static dims from source where
    /// possible
    for (int64_t d = 0; d < remainingRank; ++d) {
      int64_t srcDim = pinnedDims + d;
      subShape.push_back(sourceMemRefType.isDynamicDim(srcDim)
                             ? ShapedType::kDynamic
                             : sourceMemRefType.getDimSize(srcDim));
    }
  }
  Type elementType = sourceMemRefType.getElementType();
  Type ptrType = MemRefType::get(subShape, elementType);

  state.addTypes({guidType, ptrType});

  /// Add operands and attributes
  state.addAttribute("mode", ArtsModeAttr::get(builder.getContext(), mode));
  state.addOperands(sourceGuid);
  state.addOperands(sourcePtr);
  state.addOperands(indices);
  state.addOperands(offsets);
  state.addOperands(sizes);

  /// Build operand segment sizes: [sourceGuid=1, sourcePtr=1, indices,
  /// offsets, sizes]
  state.addAttribute(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(indices.size()),
                                    static_cast<int32_t>(offsets.size()),
                                    static_cast<int32_t>(sizes.size())}));
}

LogicalResult DbAcquireOp::verify() {
  /// Check that the number of offsets matches the number of sizes
  auto dbSizes = getSizes().size();
  auto dbOffsets = getOffsets().size();
  if (dbOffsets != dbSizes) {
    return emitOpError("Number of offsets (")
           << dbOffsets << ") must match Number of sizes (" << dbSizes << ")";
  }

  /// Get the source pointer type and rank
  auto srcRank = getSizesFromDb(getSourcePtr()).size();
  auto dbIndices = getIndices().size();
  auto remainingRank = srcRank - dbIndices;

  /// Check that source rank minus Db indices equals number of sizes
  if (remainingRank != dbSizes) {
    return emitOpError("Source rank (")
           << srcRank << ") - (" << dbIndices << ") = " << remainingRank
           << " must equal Number of sizes (" << dbSizes << ")";
  }

  /// Check if the source has a single size of 1 and we are using indices
  if (dbIndices > 0) {
    Operation *sourceDb = getUnderlyingDb(getSourcePtr());
    if (dbHasSingleSize(sourceDb)) {
      return emitOpError(
          "Cannot use indices on single-element datablock (size=1)");
    }
  }

  return success();
}

/// DbRefOp builders
void DbRefOp::build(OpBuilder &builder, OperationState &state, Value source,
                    ArrayRef<Value> indices) {
  /// The dbref
  DbAllocOp dbAllocOp = dyn_cast<DbAllocOp>(arts::getUnderlyingDbAlloc(source));
  assert(dbAllocOp && "Expected dbAllocOp");
  state.addTypes(dbAllocOp.getAllocatedElementType().getElementType());
  state.addOperands(source);
  state.addOperands(indices);
  state.addAttribute(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr({1, static_cast<int32_t>(indices.size())}));
}

LogicalResult DbRefOp::verify() {
  auto underlyingDbOp = arts::getUnderlyingDb(getSource());
  if (!underlyingDbOp)
    return emitOpError("source must be a datablock operation");

  /// Verify outer indices
  auto outerSizes = getSizesFromDb(underlyingDbOp);
  if (getIndices().size() != outerSizes.size())
    return emitOpError("expects ")
           << outerSizes.size() << " index operands (got "
           << getIndices().size() << ")";

  /// Verify output
  DbAllocOp dbAllocOp = nullptr;
  if (auto dbAcquire = dyn_cast<DbAcquireOp>(underlyingDbOp)) {
    dbAllocOp = dyn_cast<DbAllocOp>(
        arts::getUnderlyingDbAlloc(dbAcquire.getSourcePtr()));
  } else
    dbAllocOp = dyn_cast<DbAllocOp>(underlyingDbOp);

  /// Verify output type
  auto resultType = getResult().getType().cast<MemRefType>();
  if (resultType != dbAllocOp.getAllocatedElementType().getElementType())
    return emitOpError("result type must match source element type");
  return success();
}

/// Custom builders for DbNumElementsOp that automatically extract sizes
void DbNumElementsOp::build(OpBuilder &builder, OperationState &state,
                            DbAllocOp allocOp) {
  /// Extract sizes from the DbAllocOp
  SmallVector<Value> sizes = allocOp.getSizes();
  build(builder, state, sizes);
}

void DbNumElementsOp::build(OpBuilder &builder, OperationState &state,
                            DbAcquireOp acquireOp) {
  /// Extract sizes from the DbAcquireOp
  SmallVector<Value> sizes = acquireOp.getSizes();
  build(builder, state, sizes);
}

///==========================================================================
/// Utility functions for Arts dialect operations
///==========================================================================

/// Helper function to extract sizes from a datablock pointer
/// by finding the original DbAllocOp or DbAcquireOp that created it
SmallVector<Value> getSizesFromDb(Operation *dbOp) {
  if (auto allocOp = dyn_cast<DbAllocOp>(dbOp)) {
    return SmallVector<Value>(allocOp.getSizes().begin(),
                              allocOp.getSizes().end());
  }
  if (auto acquireOp = dyn_cast<DbAcquireOp>(dbOp)) {
    return SmallVector<Value>(acquireOp.getSizes().begin(),
                              acquireOp.getSizes().end());
  }
  return {};
}

SmallVector<Value> getElementSizesFromDb(Operation *dbOp) {
  if (auto allocOp = dyn_cast<DbAllocOp>(dbOp)) {
    return SmallVector<Value>(allocOp.getElementSizes().begin(),
                              allocOp.getElementSizes().end());
  }
  return {};
}

SmallVector<Value> getSizesFromDb(Value datablockPtr) {
  /// Use getUnderlyingDb to find the original DB operation
  Operation *underlyingDb = arts::getUnderlyingDb(datablockPtr);
  if (!underlyingDb)
    return {};

  return getSizesFromDb(underlyingDb);
}

/// Check if a datablock operation has a single size of 1
bool dbHasSingleSize(Operation *dbOp) {
  if (!dbOp)
    return false;

  SmallVector<Value> sizes = getSizesFromDb(dbOp);

  if (sizes.empty())
    return false;

  /// Check if there's exactly one size and it's constant 1
  if (sizes.size() != 1)
    return false;

  if (auto constSize = sizes[0].getDefiningOp<arith::ConstantIndexOp>())
    return constSize.value() == 1;

  return false;
}

SmallVector<Value> getOffsetsFromDb(Value datablockPtr) {
  Operation *underlyingDb = arts::getUnderlyingDb(datablockPtr);
  if (!underlyingDb)
    return {};

  if (auto acquireOp = dyn_cast<DbAcquireOp>(underlyingDb))
    return SmallVector<Value>(acquireOp.getOffsets().begin(),
                              acquireOp.getOffsets().end());

  return {};
}

//===----------------------------------------------------------------------===//
// ForOp Canonicalization
//===----------------------------------------------------------------------===//

void arts::ForOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  /// No canonicalization patterns for now
  /// Future patterns could include:
  ///  - Loop unrolling for small iteration counts
  /// - Loop fusion for adjacent loops
  /// - Loop invariant code motion
}

//===----------------------------------------------------------------------===//
// ReductionOp Builders
//===----------------------------------------------------------------------===//

void ReductionOp::build(OpBuilder &builder, OperationState &state, Value array,
                        Value result, Value lower_bound, Value upper_bound,
                        Value step, ReductionKind reduction_kind) {
  state.addOperands({array, result, lower_bound, upper_bound, step});
  state.addAttribute(
      "reduction_kind",
      ReductionKindAttr::get(builder.getContext(), reduction_kind));
}
