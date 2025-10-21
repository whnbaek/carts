///==========================================================================
/// File: EdtLowering.cpp
/// Complete implementation of EDT lowering pass that transforms arts.edt
/// operations into runtime-compatible function calls.
///
/// This pass implements a 6-step EDT lowering process:
/// 1. Analyze EDT region for free variables and deps
/// 2. Outline EDT region to function with ARTS runtime signature
/// 3. Insert parameter packing before EDT (edt_param_pack) - It should include
///    all parameters from the EDT + unique datablock sizes and indices for all
///    deps
/// 4. Insert parameter/dependency unpacking in outlined function
/// 5. Replace EDT with edt_create call returning GUID
/// 6. Add dependency management (record_in_dep, increment_out_latch)
///==========================================================================

#include "ArtsPassDetails.h"
#include "arts/Analysis/ArtsAnalysisManager.h"
#include "arts/ArtsDialect.h"
#include "arts/Codegen/ArtsCodegen.h"
#include "arts/Passes/ArtsPasses.h"
#include "arts/Utils/ArtsUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "arts/Utils/ArtsDebug.h"
#include "polygeist/Ops.h"
ARTS_DEBUG_SETUP(edt_lowering);

using namespace mlir;
using namespace mlir::func;
using namespace mlir::arts;

namespace {

//===----------------------------------------------------------------------===//
// EdtEnvManager
// Manages the environment analysis for EDT regions by collecting parameters,
// constants, and dependencies used in the region.
//===----------------------------------------------------------------------===//
class EdtEnvManager {
public:
  EdtEnvManager(EdtOp edtOp) : edtOp(edtOp) { analyze(); }

  /// Analyze the region and collect parameters/constants
  void analyze() {
    getUsedValuesDefinedAbove(edtOp.getRegion(), capturedValues);

    /// Checks if the value is a constant, if so, ignore it
    auto isConstant = [&](Value val) {
      auto defOp = val.getDefiningOp();
      if (!defOp)
        return false;

      auto constantOp = dyn_cast<arith::ConstantOp>(defOp);
      if (!constantOp)
        return false;
      constants.insert(val);
      return true;
    };

    /// Classify variables into parameters, constants, and captured values
    for (Value val : capturedValues) {
      if (isConstant(val))
        continue;

      /// Only treat integers, indices, or floats as parameters
      if (val.getType().isIntOrIndexOrFloat())
        parameters.insert(val);

      /// Ignore other types, they might be dependencies
    }

    /// Only treat explicit EDT operands as deps
    for (Value operand : edtOp.getDependencies())
      deps.insert(operand);
  }

  ArrayRef<Value> getParameters() const { return parameters.getArrayRef(); }
  ArrayRef<Value> getConstants() const { return constants.getArrayRef(); }
  ArrayRef<Value> getCapturedValues() const {
    return capturedValues.getArrayRef();
  }
  ArrayRef<Value> getDependencies() const { return deps.getArrayRef(); }
  const DenseMap<Value, unsigned> &getValueToPackIndex() const {
    return valueToPackIndex;
  }
  DenseMap<Value, unsigned> &getValueToPackIndex() { return valueToPackIndex; }

private:
  EdtOp edtOp;
  SetVector<Value> capturedValues, parameters, constants, deps;
  DenseMap<Value, unsigned> valueToPackIndex;
};


//===----------------------------------------------------------------------===//
// EDT Lowering Pass Implementation
//===----------------------------------------------------------------------===//
struct EdtLoweringPass : public arts::EdtLoweringBase<EdtLoweringPass> {
  explicit EdtLoweringPass() {}
  ~EdtLoweringPass() { delete AC; }
  void runOnOperation() override;

private:
  /// Core transformation methods
  LogicalResult lowerEdt(EdtOp edtOp);

  /// Function outlining with ARTS signature
  func::FuncOp createOutlinedFunction(EdtOp edtOp, EdtEnvManager &envManager);

  /// Parameter handling
  Value packParams(Location loc, EdtEnvManager &envManager,
                   SmallVector<Type> &packTypes);

  /// Region outlining
  LogicalResult outlineRegionToFunction(EdtOp edtOp, func::FuncOp targetFunc,
                                        EdtEnvManager &envManager,
                                        const SmallVector<Type> &packTypes,
                                        size_t numUserParams);

  void transformDepUses(ArrayRef<Value> originalDeps, Value depv,
                        ArrayRef<Value> allParams, EdtEnvManager &envManager,
                        ArrayRef<Value> depIdentifiers);

  /// Dependency satisfaction
  LogicalResult insertDepManagement(Location loc, Value edtGuid,
                                    const SmallVector<Value> &deps);

  /// Attributes
  unsigned functionCounter = 0;
  ModuleOp module;
  ArtsCodegen *AC = nullptr;
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

void EdtLoweringPass::runOnOperation() {
  module = getOperation();
  AC = new ArtsCodegen(module, false);

  ARTS_INFO_HEADER(EdtLoweringPass);
  ARTS_DEBUG_REGION(module.dump(););

  /// Collect and lower all task EDTs
  {
    ARTS_DEBUG_HEADER(TaskEdtLowering);
    SmallVector<EdtOp, 8> taskEdts;
    module.walk<WalkOrder::PostOrder>([&](EdtOp edtOp) {
      assert(edtOp.getType() == EdtType::task && "Expected task EDT");
      taskEdts.push_back(edtOp);
    });
    ARTS_INFO("Found " << taskEdts.size() << " task EDTs to lower");
    for (EdtOp edtOp : taskEdts) {
      if (failed(lowerEdt(edtOp))) {
        edtOp.emitError("Failed to lower task EDT");
        return signalPassFailure();
      }
    }
  }

  ARTS_INFO_FOOTER(EdtLoweringPass);
  ARTS_DEBUG_REGION(module.dump(););
}

//===----------------------------------------------------------------------===//
/// Lower EDT operations to runtime calls
///
/// Transforms arts.edt operations into outlined functions and runtime calls.
/// Creates an outlined function with the EDT body, packs parameters and
/// dependencies, and replaces the EDT with arts.edt_create and dependency
/// management calls. Example:
///   %result = arts.edt(%dep) {
///   ^bb0(%arg: memref<f64>):
///     %val = memref.load %arg[] : memref<f64>
///     arts.return %val : f64
///   }
/// becomes:
///   %param_pack = arts.edt_param_pack ...
///   %edt_guid = arts.edt_create %param_pack, %dep_count, %route
///   arts.record_in_dep %edt_guid, %dep
///   arts.increment_out_latch %edt_guid
//===----------------------------------------------------------------------===//
LogicalResult EdtLoweringPass::lowerEdt(EdtOp edtOp) {
  OpBuilder::InsertionGuard IG(AC->getBuilder());
  AC->setInsertionPoint(edtOp);
  Location loc = edtOp.getLoc();
  /// Analyze EDT environment (parameters, constants, deps)
  EdtEnvManager envManager(edtOp);

  /// Create local variables for parameter packing/dedup
  DenseMap<Value, unsigned> valueToPackIndex;
  SmallVector<Type> packTypes;

  /// Create edt outlined function
  func::FuncOp outlinedFunc = createOutlinedFunction(edtOp, envManager);
  if (!outlinedFunc)
    return edtOp.emitError("Failed to create outlined function");

  /// Pack parameters
  Value paramPack = packParams(loc, envManager, packTypes);

  /// Outline region to function and replace EDT
  if (failed(outlineRegionToFunction(edtOp, outlinedFunc, envManager, packTypes,
                                     envManager.getParameters().size()))) {
    return edtOp.emitError("Failed to outline region to function");
  }

  /// Calculate total dependency count (sum of elements in all deps)
  Value depCount = AC->createIntConstant(0, AC->Int32, loc);
  for (Value dep : envManager.getDependencies()) {
    SmallVector<Value> sizes = getSizesFromDb(dep);
    Value numElements = AC->create<DbNumElementsOp>(loc, sizes);
    numElements = AC->castToInt(AC->Int32, numElements, loc);
    depCount = AC->create<arith::AddIOp>(loc, depCount, numElements);
  }

  /// Create the outline operation at the same location as the EDT
  AC->setInsertionPoint(edtOp);
  Value routeVal = edtOp.getRoute();
  if (!routeVal)
    routeVal = AC->createIntConstant(0, AC->Int32, loc);

  /// Create the outline operation at the same location as the EDT
  auto outlineOp = AC->create<EdtCreateOp>(loc, paramPack, depCount, routeVal);

  outlineOp->setAttr("outlined_func",
                     AC->getBuilder().getStringAttr(outlinedFunc.getName()));

  /// Insert dependency management after the outline op
  Value edtGuid = outlineOp.getGuid();
  AC->setInsertionPointAfter(outlineOp);
  SmallVector<Value> depsVec;
  for (Value d : edtOp.getDependencies())
    depsVec.push_back(d);
  if (failed(insertDepManagement(loc, edtGuid, depsVec)))
    return edtOp.emitError("Failed to insert dependency management");

  /// Replace all uses of EDT with the outlined function result
  if (edtOp->getNumResults() > 0) {
    SmallVector<Value> replacementValues = {outlineOp.getResult()};
    edtOp->replaceAllUsesWith(replacementValues);
  }
  /// Remove original EDT
  edtOp.erase();

  return success();
}


//===----------------------------------------------------------------------===//
/// Create outlined function for EDT body
///
/// Generates a new private function with the ARTS EDT signature to contain
/// the outlined EDT region.
//===----------------------------------------------------------------------===//
func::FuncOp
EdtLoweringPass::createOutlinedFunction(EdtOp edtOp,
                                        EdtEnvManager &envManager) {
  Location loc = edtOp.getLoc();
  OpBuilder::InsertionGuard IG(AC->getBuilder());
  AC->setInsertionPoint(module);
  std::string funcName = "__arts_edt_" + std::to_string(++functionCounter);
  auto outlinedFunc = AC->create<func::FuncOp>(loc, funcName, AC->EdtFn);
  outlinedFunc.setPrivate();

  ARTS_INFO("Created outlined function: " << funcName);
  return outlinedFunc;
}

//===----------------------------------------------------------------------===//
/// Pack EDT parameters and dependency metadata
///
/// Creates a parameter pack containing user-defined parameters, constants,
/// and metadata for datablock dependencies (indices, offsets, sizes).
//===----------------------------------------------------------------------===//
Value EdtLoweringPass::packParams(Location loc, EdtEnvManager &envManager,
                                  SmallVector<Type> &packTypes) {
  const auto &parameters = envManager.getParameters();
  const auto &deps = envManager.getDependencies();

  SmallVector<Value> packValues;
  auto &valueToPackIndex = envManager.getValueToPackIndex();

  /// Pack user parameters first
  for (Value v : parameters) {
    /// Skip llvm.mlir.undef values - they can be easily recreated
    if (auto defOp = v.getDefiningOp()) {
      if (defOp->getName().getStringRef() == "llvm.mlir.undef")
        continue;
    }
    valueToPackIndex.try_emplace(v, packValues.size());
    packTypes.push_back(v.getType());
    packValues.push_back(v);
  }

  /// Insert indices/offsets/sizes for deps into packValues if not already
  /// present
  for (Value dep : deps) {
    auto dbAcquireOp = dep.getDefiningOp<DbAcquireOp>();
    if (!dbAcquireOp)
      continue;

    auto appendIfMissing = [&](Value val) {
      if (!val)
        return;
      /// Skip constants; they will be recreated in outlined function
      if (val.getDefiningOp<arith::ConstantOp>())
        return;
      if (valueToPackIndex.count(val) == 0) {
        valueToPackIndex[val] = packValues.size();
        packTypes.push_back(val.getType());
        packValues.push_back(val);
      }
    };

    for (Value idx : dbAcquireOp.getIndices())
      appendIfMissing(idx);
    for (Value off : dbAcquireOp.getOffsets())
      appendIfMissing(off);
    for (Value sz : dbAcquireOp.getSizes())
      appendIfMissing(sz);
  }

  if (packValues.empty()) {
    ARTS_INFO("No parameters to pack, creating empty EdtParamPackOp");
    auto emptyType = MemRefType::get({0}, AC->Int64);
    return AC->create<EdtParamPackOp>(loc, TypeRange{emptyType}, ValueRange{});
  }

  ARTS_DEBUG_REGION({
    ARTS_INFO("Creating parameter pack for " << packValues.size() << " items");
    for (size_t i = 0; i < packValues.size(); ++i)
      ARTS_DEBUG("  packValues[" << i << "]: " << packValues[i]);
  });

  auto memrefType = MemRefType::get({ShapedType::kDynamic}, AC->Int64);
  auto packOp = AC->create<EdtParamPackOp>(loc, TypeRange{memrefType},
                                           ValueRange(packValues));
  return packOp;
}

//===----------------------------------------------------------------------===//
/// Outline EDT region to target function
///
/// Moves the EDT body region into the outlined function, performs parameter
/// and dependency unpacking, and updates value references to work with the
/// new function context.
//===----------------------------------------------------------------------===//
LogicalResult EdtLoweringPass::outlineRegionToFunction(
    EdtOp edtOp, func::FuncOp targetFunc, EdtEnvManager &envManager,
    const SmallVector<Type> &packTypes, size_t numUserParams) {
  Location loc = edtOp.getLoc();
  auto &builder = AC->getBuilder();
  OpBuilder::InsertionGuard IG(builder);
  auto *entryBlock = targetFunc.addEntryBlock();
  AC->setInsertionPointToStart(entryBlock);

  /// Insert parameter and dependency unpacking
  auto args = entryBlock->getArguments();
  Value paramv = args[1];
  Value depv = args[3];

  const auto &parameters = envManager.getParameters();
  SmallVector<Value> deps;
  for (Value d : edtOp.getDependencies())
    deps.push_back(d);
  ARTS_INFO("analyzeDependencies returned " << deps.size() << " dependencies");
  for (size_t i = 0; i < deps.size(); ++i)
    ARTS_DEBUG("  dep[" << i << "]: " << deps[i]);
  SmallVector<Value> unpackedParams, allParams;

  if (!packTypes.empty()) {
    auto paramUnpackOp = AC->create<EdtParamUnpackOp>(loc, packTypes, paramv);
    auto results = paramUnpackOp.getResults();
    allParams.assign(results.begin(), results.end());
    unpackedParams.append(allParams.begin(), allParams.begin() + numUserParams);
  }

  /// Store dependency information for direct use in strict operand order
  SmallVector<Value> originalDeps;
  for (Value d : edtOp.getDependencies())
    originalDeps.push_back(d);
  Region &edtRegion = edtOp.getRegion();
  Block &edtBlock = edtRegion.front();

  /// Create compact per-dependency placeholders for later dep_gep rewrite.
  SmallVector<Value> depPlaceholders(originalDeps.size());
  for (auto it : llvm::enumerate(originalDeps))
    depPlaceholders[it.index()] =
        AC->create<UndefOp>(loc, edtBlock.getArguments()[it.index()].getType());

  /// Clone constants into function
  IRMapping valueMapping;

  /// Map EDT args to placeholders so cloned ops don't reference outer values.
  for (auto [edtArg, placeholder] :
       llvm::zip(edtBlock.getArguments(), depPlaceholders))
    valueMapping.map(edtArg, placeholder);

  /// Also map the original dependency values to their corresponding
  /// placeholders to catch any direct uses that bypassed the block arguments.
  for (auto [originalDep, placeholder] :
       llvm::zip(originalDeps, depPlaceholders))
    valueMapping.map(originalDep, placeholder);

  for (Value constant : envManager.getConstants())
    if (Operation *defOp = constant.getDefiningOp())
      valueMapping.map(constant, builder.clone(*defOp)->getResult(0));

  /// Map parameters directly to their unpacked counterparts; clone constants
  /// and undef.
  size_t unpackedIndex = 0;
  for (Value param : parameters) {
    /// Handle llvm.mlir.undef by recreating it instead of unpacking
    if (auto defOp = param.getDefiningOp()) {
      if (defOp->getName().getStringRef() == "llvm.mlir.undef") {
        valueMapping.map(param, builder.clone(*defOp)->getResult(0));
        continue;
      }
    }
    /// Map to unpacked parameter
    if (unpackedIndex < unpackedParams.size())
      valueMapping.map(param, unpackedParams[unpackedIndex++]);
  }

  for (Value freeVar : envManager.getCapturedValues())
    if (Operation *defOp = freeVar.getDefiningOp())
      if (defOp->hasTrait<OpTrait::ConstantLike>())
        valueMapping.map(freeVar, builder.clone(*defOp)->getResult(0));

  /// Clone region operations
  builder.setInsertionPointToEnd(entryBlock);
  for (Operation &op :
       llvm::make_early_inc_range(edtBlock.without_terminator())) {
    Operation *clonedOp = builder.clone(op, valueMapping);
    for (auto [orig, clone] :
         llvm::zip(op.getResults(), clonedOp->getResults()))
      valueMapping.map(orig, clone);
  }

  /// Add return terminator
  AC->create<func::ReturnOp>(loc);

  /// Transform dependency uses inside outlined region
  transformDepUses(originalDeps, depv, allParams, envManager, depPlaceholders);
  return success();
}

//===----------------------------------------------------------------------===//
/// Insert dependency management operations
///
/// Adds runtime dependency tracking operations (record_in_dep,
/// increment_out_latch) for EDT execution. Separates dependencies by access
/// mode and creates appropriate runtime calls to establish data dependencies
/// between EDTs. Example:
///   EDT with input deps %d1, %d2 and output deps %d3
///   becomes: arts.record_in_dep %edt_guid, [%d1_guid, %d2_guid, %d3_guid]
///            arts.increment_out_latch %edt_guid, [%d3_guid]
//===----------------------------------------------------------------------===//
LogicalResult
EdtLoweringPass::insertDepManagement(Location loc, Value edtGuid,
                                     const SmallVector<Value> &deps) {
  if (deps.empty())
    return success();

  /// Separate deps by mode and extract GUIDs
  SmallVector<Value> inDepGuids, outDepGuids;

  for (Value dep : deps) {
    auto dbAcquireOp = dep.getDefiningOp<DbAcquireOp>();
    assert(dbAcquireOp && "Dependencies must be DbAcquireOp operations");

    /// Get the GUID from the DbAcquireOp
    Value depGuid = dbAcquireOp.getGuid();

    /// Always add to in-dependencies
    inDepGuids.push_back(depGuid);

    /// TODO: Fix this
    /// Only add to out-dependencies if mode is out or inout
    // ArtsMode mode = dbAcquireOp.getMode();
    // if (mode == ArtsMode::out || mode == ArtsMode::inout)
    outDepGuids.push_back(depGuid);
  }

  /// Record all input deps at once using GUIDs
  if (!inDepGuids.empty())
    AC->create<RecordDepOp>(loc, edtGuid, inDepGuids);

  /// Increment output latches for all deps at once using GUIDs
  if (!outDepGuids.empty())
    AC->create<IncrementDepOp>(loc, edtGuid, outDepGuids);

  return success();
}

//===----------------------------------------------------------------------===//
/// Transform dependency uses in outlined function
///
/// Rewrites dependency access operations in the outlined EDT function to use
/// the packed dependency data structure. Computes proper base offsets and
/// strides for each dependency, adjusting indices to account for datablock
/// offsets.
//===----------------------------------------------------------------------===//
void EdtLoweringPass::transformDepUses(ArrayRef<Value> originalDeps, Value depv,
                                       ArrayRef<Value> allParams,
                                       EdtEnvManager &envManager,
                                       ArrayRef<Value> depIdentifiers) {
  /// Get the parameter map
  const auto &paramMap = envManager.getValueToPackIndex();

  /// Resolve the sizes of the dependency within the outlined Edt
  auto resolveSizes = [&](Value dep, Location loc) {
    SmallVector<Value> sizes = getSizesFromDb(dep);
    SmallVector<Value> resolved;
    for (Value sz : sizes) {
      auto it = paramMap.find(sz);
      if (it != paramMap.end() && it->second < allParams.size())
        resolved.push_back(allParams[it->second]);
      else if (auto c = sz.getDefiningOp<arith::ConstantIndexOp>())
        resolved.push_back(AC->createIndexConstant(c.value(), loc));
      else
        resolved.push_back(sz);
    }
    if (resolved.empty())
      resolved.push_back(AC->createIndexConstant(1, loc));
    return resolved;
  };

  /// Compute the base offset of the dependency within the outlined Edt
  /// This corresponds to the sum of number of elements in the previous
  /// dependencies
  auto computeBaseOffset = [&](size_t depIndex, Location loc) {
    Value base = AC->createIndexConstant(0, loc);
    for (size_t i = 0; i < depIndex; ++i) {
      SmallVector<Value> prevResolved = resolveSizes(originalDeps[i], loc);
      Value prevElems = AC->computeTotalElements(prevResolved, loc);
      base = AC->create<arith::AddIOp>(loc, base, prevElems);
    }
    return base;
  };

  /// For each dependency placeholder, rewrite its direct uses.
  for (size_t depIndex = 0; depIndex < depIdentifiers.size(); ++depIndex) {
    Value placeholder = depIdentifiers[depIndex];
    AC->setInsertionPoint(placeholder.getDefiningOp());

    Location loc = placeholder.getLoc();

    /// Get the base offset, sizes, strides, and offsets of the dependency
    ARTS_DEBUG("Processing Dep[" << depIndex
                                 << "]: " << originalDeps[depIndex]);
    Value baseOffset = computeBaseOffset(depIndex, loc);
    SmallVector<Value> depSizes = resolveSizes(originalDeps[depIndex], loc);
    SmallVector<Value> depStrides = AC->computeStridesFromSizes(depSizes, loc);

    bool isSingleElement =
        dbHasSingleSize(originalDeps[depIndex].getDefiningOp<DbAcquireOp>());
    if (isSingleElement) {
      Value depPtr = AC->create<DepGepOp>(loc, AC->llvmPtr, depv, baseOffset,
                                          ValueRange(), ValueRange());
      placeholder.replaceAllUsesWith(depPtr);
      continue;
    }

    /// Get the users of the dependency placeholder
    SmallVector<Operation *, 16> users;
    for (auto &use : placeholder.getUses())
      users.push_back(use.getOwner());

    ARTS_DEBUG(" - Rewriting " << users.size() << " users");

    /// For each user of the dependency placeholder, rewrite the operation
    for (Operation *op : users) {
      if (auto mp = dyn_cast<polygeist::Memref2PointerOp>(op)) {
        if (mp.getSource() != placeholder)
          continue;
        Value depPtr = AC->create<DepGepOp>(loc, AC->llvmPtr, depv, baseOffset,
                                            ValueRange(), ValueRange());
        op->getResult(0).replaceAllUsesWith(depPtr);
        op->erase();
      } else if (auto dbGep = dyn_cast<arts::DbGepOp>(op)) {
        if (dbGep.getBasePtr() != placeholder)
          continue;
        AC->setInsertionPoint(op);
        SmallVector<Value> dbGepIndices(dbGep.getIndices().begin(),
                                        dbGep.getIndices().end());
        Value depPtr = AC->create<DepGepOp>(loc, AC->llvmPtr, depv, baseOffset,
                                            dbGepIndices, depStrides);
        op->getResult(0).replaceAllUsesWith(depPtr);
        op->erase();
      } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        if (store.getMemRef() != placeholder)
          continue;
        AC->setInsertionPoint(op);
        SmallVector<Value> storeIndices(store.getIndices().begin(),
                                        store.getIndices().end());
        Value depPtr = AC->create<DepGepOp>(loc, AC->llvmPtr, depv, baseOffset,
                                            storeIndices, depStrides);
        AC->create<LLVM::StoreOp>(loc, store.getValue(), depPtr);
        op->erase();
      } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
        if (load.getMemRef() != placeholder)
          continue;
        AC->setInsertionPoint(op);
        SmallVector<Value> loadIndices(load.getIndices().begin(),
                                       load.getIndices().end());
        Value depPtr = AC->create<DepGepOp>(loc, AC->llvmPtr, depv, baseOffset,
                                            loadIndices, depStrides);
        Value loaded = AC->create<LLVM::LoadOp>(loc, load.getType(), depPtr);
        op->getResult(0).replaceAllUsesWith(loaded);
        op->erase();
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace arts {

std::unique_ptr<Pass> createEdtLoweringPass() {
  return std::make_unique<EdtLoweringPass>();
}

} // namespace arts
} // namespace mlir
