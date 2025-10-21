///==========================================================================
/// File: ArtsCodegen.cpp
///==========================================================================

#include "arts/Codegen/ArtsCodegen.h"

/// Other dialects
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "polygeist/Dialect.h"
#include "polygeist/Ops.h"
#include "llvm/IR/Attributes.h"
/// Other
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
/// Arts
#include "arts/ArtsDialect.h"
#include "arts/Utils/ArtsUtils.h"
/// Debug

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <functional>

/// DEBUG
#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(arts_codegen);

using namespace mlir;
using namespace mlir::func;
using namespace mlir::LLVM;
using namespace mlir::arts;
using namespace mlir::arith;
using namespace mlir::polygeist;
using namespace mlir::cf;
using namespace mlir::scf;

/// ARTS Codegen
ArtsCodegen::ArtsCodegen(ModuleOp module, bool debug)
    : module(module), builder(OpBuilder(module->getContext())), debug(debug) {
  ARTS_DEBUG_HEADER(ArtsCodegen);
  if (failed(extractDataLayouts())) {
    mlir::emitError(module.getLoc()) << "Failed to extract data layouts";
    llvm_unreachable("Failed to extract data layouts");
    return;
  }
  initializeTypes();
  collectGlobalLLVMStrings();
}

ArtsCodegen::ArtsCodegen(ModuleOp &module, llvm::DataLayout &llvmDL,
                         mlir::DataLayout &mlirDL, bool debug)
    : module(module), builder(OpBuilder(module->getContext())), debug(debug) {
  this->llvmDL = std::make_unique<llvm::DataLayout>(llvmDL);
  this->mlirDL = std::make_unique<mlir::DataLayout>(mlirDL);

  initializeTypes();
  collectGlobalLLVMStrings();
}

ArtsCodegen::~ArtsCodegen() { ARTS_DEBUG_FOOTER(ArtsCodegen); }

func::FuncOp ArtsCodegen::getOrCreateRuntimeFunction(RuntimeFunction FnID) {
  auto cacheIt = runtimeFunctionCache.find(FnID);
  if (cacheIt != runtimeFunctionCache.end())
    return cacheIt->second;

  OpBuilder::InsertionGuard IG(getBuilder());
  getBuilder().setInsertionPointToStart(module.getBody());

  func::FuncOp funcOp;
  /// Try to find the declaration in the module first.
#define ARTS_RTL_FUNCTIONS
#define ARTS_RTL(Enum, Str, ReturnType, ...)                                   \
  case Enum: {                                                                 \
    SmallVector<Type, 4> argumentTypes{__VA_ARGS__};                           \
    auto fnType = getBuilder().getFunctionType(                                \
        argumentTypes, ReturnType.isa<mlir::NoneType>()                        \
                           ? ArrayRef<Type>{}                                  \
                           : ArrayRef<Type>{ReturnType});                      \
    funcOp = module.lookupSymbol<func::FuncOp>(Str);                           \
    if (!funcOp) {                                                             \
      funcOp = create<func::FuncOp>(getUnknownLoc(), Str, fnType);             \
      funcOp.setPrivate();                                                     \
      funcOp->setAttr(                                                         \
          "llvm.linkage",                                                      \
          LLVM::LinkageAttr::get(getContext(), LLVM::Linkage::External));      \
      /* Apply function attributes for optimization */                         \
      applyRuntimeFunctionAttributes(funcOp, Enum);                            \
    }                                                                          \
    break;                                                                     \
  }
  switch (FnID) {
#include "arts/Codegen/ArtsKinds.def"
  }
#undef ARTS_RTL_FUNCTIONS
#undef ARTS_RTL

  assert(funcOp && "Failed to create ARTS runtime function");

  /// Cache the function for future use
  runtimeFunctionCache[FnID] = funcOp;
  return funcOp;
}

void ArtsCodegen::applyRuntimeFunctionAttributes(func::FuncOp funcOp,
                                                 RuntimeFunction fnID) {
  /// Apply function attributes based on ARTS runtime function properties
  auto unitAttr = getBuilder().getUnitAttr();

  /// Helper to apply MLIR attributes from a list of attribute names
  auto applyMLIRAttrs = [&](ArrayRef<StringRef> attrNames) {
    for (StringRef attrName : attrNames)
      funcOp->setAttr(attrName, unitAttr);
  };

  /// Helper to apply argument attributes from a list of attribute names
  auto applyArgMLIRAttrs = [&](unsigned argIndex,
                               ArrayRef<StringRef> attrNames) {
    for (StringRef attrName : attrNames)
      funcOp.setArgAttr(argIndex, attrName, unitAttr);
  };

  /// Define attribute sets
#define ARTS_ATTRS_SET(VarName, AttrList)                                      \
  SmallVector<StringRef> VarName = AttrList;
#include "arts/Codegen/ArtsKinds.def"

  /// Add attributes to the function declaration
  switch (fnID) {
#define ARTS_RTL_ATTRS(Enum, FnAttrSet, RetAttrSet, ArgAttrSets)               \
  case Enum:                                                                   \
    applyMLIRAttrs(FnAttrSet);                                                 \
    /* Apply argument attributes */                                            \
    for (size_t ArgNo = 0; ArgNo < ArgAttrSets.size(); ++ArgNo)                \
      applyArgMLIRAttrs(ArgNo, ArgAttrSets[ArgNo]);                            \
    break;
#define ARTS_RTL_FUNCTIONS
#define ARTS_RTL(Enum, Name, ReturnType, ...) /// No-op, we only want attributes
#define ParamAttrs(...) SmallVector<SmallVector<StringRef>>({__VA_ARGS__})
#include "arts/Codegen/ArtsKinds.def"
#undef ParamAttrs
#undef ARTS_RTL
#undef ARTS_RTL_FUNCTIONS
    /// All cases handled by .def file
  }
#undef ARTS_RTL_ATTRS
}

void ArtsCodegen::initializeTypes() {
  MLIRContext *context = module.getContext();
  llvmPtr = LLVM::LLVMPointerType::get(context);

#define ARTS_TYPE(VarName, InitValue) VarName = InitValue;
#define ARTS_FUNCTION_TYPE(VarName, ReturnType, ...)                           \
  VarName = FunctionType::get(                                                 \
      context, {__VA_ARGS__},                                                  \
      ((ReturnType == Void) ? ArrayRef<Type>{} : ArrayRef<Type>{ReturnType})); \
  VarName##Ptr = MemRefType::get(                                              \
      {ShapedType::kDynamic},                                                  \
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context),            \
                                  VarName.getInputs(), false));
#define ARTS_STRUCT_TYPE(VarName, StructName, Packed, ...)                     \
  VarName = LLVM::LLVMStructType::getLiteral(context, {__VA_ARGS__}, Packed);  \
  VarName##Ptr = MemRefType::get({ShapedType::kDynamic}, VarName);
#include "arts/Codegen/ArtsKinds.def"
}

LogicalResult ArtsCodegen::extractDataLayouts() {
  /// Create LLVM DataLayout from the string attribute
  auto llvmDLAttr = module->getAttrOfType<StringAttr>("llvm.data_layout");
  if (!llvmDLAttr)
    return module.emitError("Module missing required LLVM data layout");
  llvmDL = std::make_unique<llvm::DataLayout>(llvmDLAttr.getValue().str());

  /// Create MLIR DataLayout from the module
  mlirDL = std::make_unique<mlir::DataLayout>(module);
  return success();
}

func::CallOp ArtsCodegen::createRuntimeCall(RuntimeFunction FnID,
                                            ArrayRef<Value> args,
                                            Location loc) {
  func::FuncOp func = getOrCreateRuntimeFunction(FnID);
  assert(func && "Runtime function should exist");
  return create<func::CallOp>(loc, func, args);
}

/// DataBlock management

void ArtsCodegen::addDbDep(Value dbGuid, Value edtGuid, Value edtSlot,
                           Location loc) {
  auto edtSlotInt = castToInt(Int32, edtSlot, loc);
  createRuntimeCall(ARTSRTL_artsDbAddDependence, {dbGuid, edtGuid, edtSlotInt},
                    loc);
}

void ArtsCodegen::incrementDbLatchCount(Value dbGuid, Location loc) {
  createRuntimeCall(ARTSRTL_artsDbIncrementLatch, {dbGuid}, loc);
}

void ArtsCodegen::decrementDbLatchCount(Value dbGuid, Location loc) {
  createRuntimeCall(ARTSRTL_artsDbDecrementLatch, {dbGuid}, loc);
}

/// EDT management

/// Epoch
Value ArtsCodegen::createEpoch(Value finishEdtGuid, Value finishEdtSlot,
                               Location loc) {
  auto finishEdtSlotInt = castToInt(Int32, finishEdtSlot, loc);
  return createRuntimeCall(ARTSRTL_artsInitializeAndStartEpoch,
                           {finishEdtGuid, finishEdtSlotInt}, loc)
      .getResult(0);
}

/// Utils
Value ArtsCodegen::getCurrentEpochGuid(Location loc) {
  return createRuntimeCall(ARTSRTL_artsGetCurrentEpochGuid, {}, loc)
      .getResult(0);
}

Value ArtsCodegen::getCurrentEdtGuid(Location loc) {
  func::FuncOp func = getOrCreateRuntimeFunction(ARTSRTL_artsGetCurrentGuid);

  return create<func::CallOp>(loc, func, ArrayRef<Value>{}).getResult(0);
}

Value ArtsCodegen::getTotalWorkers(Location loc) {
  func::FuncOp func = getOrCreateRuntimeFunction(ARTSRTL_artsGetTotalWorkers);
  return create<func::CallOp>(loc, func, ArrayRef<Value>{}).getResult(0);
}

Value ArtsCodegen::getTotalNodes(Location loc) {
  func::FuncOp func = getOrCreateRuntimeFunction(ARTSRTL_artsGetTotalNodes);
  return create<func::CallOp>(loc, func, ArrayRef<Value>{}).getResult(0);
}

Value ArtsCodegen::getCurrentWorker(Location loc) {
  func::FuncOp func = getOrCreateRuntimeFunction(ARTSRTL_artsGetCurrentWorker);
  return create<func::CallOp>(loc, func, ArrayRef<Value>{}).getResult(0);
}

Value ArtsCodegen::getCurrentNode(Location loc) {
  func::FuncOp func = getOrCreateRuntimeFunction(ARTSRTL_artsGetCurrentNode);
  return create<func::CallOp>(loc, func, ArrayRef<Value>{}).getResult(0);
}

func::CallOp ArtsCodegen::signalEdt(Value edtGuid, Value edtSlot, Value dbGuid,
                                    Location loc) {
  auto edtSlotInt = castToInt(Int32, edtSlot, loc);
  return createRuntimeCall(ARTSRTL_artsSignalEdt, {edtGuid, edtSlotInt, dbGuid},
                           loc);
}

void ArtsCodegen::waitOnHandle(Value epochGuid, Location loc) {
  createRuntimeCall(ARTSRTL_artsWaitOnHandle, {epochGuid}, loc);
}

func::FuncOp ArtsCodegen::insertInitPerWorker(Location loc) {
  OpBuilder::InsertionGuard IG(getBuilder());
  setInsertionPoint(module);
  auto newFn = create<func::FuncOp>(loc, "initPerWorker", InitPerWorkerFn);
  newFn.setPublic();
  module.push_back(newFn);
  auto *entryBlock = newFn.addEntryBlock();
  getBuilder().setInsertionPointToStart(entryBlock);
  create<func::ReturnOp>(loc);
  return newFn;
}

func::FuncOp ArtsCodegen::insertInitPerNode(Location loc,
                                            func::FuncOp callback) {
  OpBuilder::InsertionGuard IG(getBuilder());
  setInsertionPoint(module);

  /// Create the function using the InitPerNodeFn type.
  auto newFn = create<func::FuncOp>(loc, "initPerNode", InitPerNodeFn);
  newFn.setPublic();
  module.push_back(newFn);

  /// Create the entry block.
  Block *entryBlock = newFn.addEntryBlock();
  getBuilder().setInsertionPointToStart(entryBlock);

  /// Retrieve the first argument and compare arg with 1.
  auto arg = entryBlock->getArgument(0);
  auto cmp =
      create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                            castToIndex(arg, loc), createIndexConstant(1, loc));

  /// Create separate then and merge blocks in the function body.
  auto thenBlock = new Block();
  auto mergeBlock = new Block();
  mergeBlock->addArgument(newFn.getArgument(1).getType(), loc);
  mergeBlock->addArgument(newFn.getArgument(2).getType(), loc);
  newFn.getBody().push_back(thenBlock);
  newFn.getBody().push_back(mergeBlock);

  /// In the entry block, do a conditional branch: if cmp true, jump to
  /// thenBlock; otherwise, continue in mergeBlock.
  create<cf::CondBranchOp>(
      loc, cmp, thenBlock, ValueRange{}, mergeBlock,
      ValueRange{newFn.getArgument(1), newFn.getArgument(2)});

  /// thenBlock - If nodeId >= 1, return
  getBuilder().setInsertionPointToStart(thenBlock);
  create<func::ReturnOp>(loc);

  /// mergeBlock - Otherwise, call the callback function and return
  getBuilder().setInsertionPointToStart(mergeBlock);
  if (callback) {
    /// If the callback function receives arguments, pass them to the call,
    /// otherwise, pass an empty ValueRange.
    auto callArgs = ValueRange{};
    if (callback.getNumArguments() > 0)
      callArgs = {newFn.getArgument(1), newFn.getArgument(2)};
    create<func::CallOp>(loc, callback, callArgs);
  }
  createRuntimeCall(ARTSRTL_artsShutdown, {}, loc);
  create<func::ReturnOp>(loc);
  return newFn;
}

func::FuncOp ArtsCodegen::insertArtsMainFn(Location loc,
                                           func::FuncOp callback) {
  OpBuilder::InsertionGuard IG(getBuilder());
  setInsertionPoint(module);

  /// Create the function using the "MainEdt" type.
  auto newFn = create<func::FuncOp>(loc, "artsMain", ArtsMainFn);
  newFn.setPublic();

  /// Create the entry block.
  auto *entryBlock = newFn.addEntryBlock();
  getBuilder().setInsertionPointToStart(entryBlock);

  /// Debug: Print that artsMain is being called
  createPrintfCall(loc, "DEBUG: artsMain function called!\n", {});

  /// Insert call to 'artsRT' function
  auto callArgs = ValueRange{};
  if (callback.getNumArguments() > 0)
    callArgs = {newFn.getArgument(0), newFn.getArgument(1)};
  create<func::CallOp>(loc, callback, callArgs);

  /// Return
  createRuntimeCall(ARTSRTL_artsShutdown, {}, loc);
  create<func::ReturnOp>(loc);
  return newFn;
}

func::FuncOp ArtsCodegen::insertMainFn(Location loc) {
  OpBuilder::InsertionGuard IG(getBuilder());
  setInsertionPoint(module);

  /// Create the function using the "MainFn" type.
  auto newFn = create<func::FuncOp>(loc, "main", MainFn);
  newFn.setPublic();

  /// Create the entry block.
  auto *entryBlock = newFn.addEntryBlock();
  getBuilder().setInsertionPointToStart(entryBlock);

  /// Insert call to 'artsRT' function
  createRuntimeCall(ARTSRTL_artsRT,
                    {newFn.getArgument(0), newFn.getArgument(1)}, loc);

  /// Return 0
  create<func::ReturnOp>(loc, createIntConstant(0, Int32, loc));
  return newFn;
}

void ArtsCodegen::initRT(Location loc) {
  auto mainFn = module.lookupSymbol<func::FuncOp>("main");
  if (!mainFn)
    return;

  /// Rename main function to "mainBody"
  mainFn.setName("mainBody");

  /// Insert init functions (artsMain and main)
  insertArtsMainFn(loc, mainFn);
  insertMainFn(loc);
}

/// Helpers
Value ArtsCodegen::createFnPtr(func::FuncOp funcOp, Location loc) {
  auto FT = funcOp.getFunctionType();
  auto LFT = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(funcOp.getContext()), FT.getInputs(), false);
  auto getFuncOp = create<polygeist::GetFuncOp>(
      loc, LLVM::LLVMPointerType::get(LFT), funcOp.getName());
  return create<polygeist::Pointer2MemrefOp>(
             loc, MemRefType::get({ShapedType::kDynamic}, LFT), getFuncOp)
      .getResult();
}

Value ArtsCodegen::createIndexConstant(int value, Location loc) {
  return create<arith::ConstantIndexOp>(loc, value);
}

Value ArtsCodegen::createIntConstant(int value, Type type, Location loc) {
  assert(type.isa<IntegerType>() && "Expected integer type");
  auto v = create<arith::ConstantOp>(loc, type,
                                     getBuilder().getIntegerAttr(type, value));
  return v;
}

Value ArtsCodegen::createPtr(Value source, Location loc) {
  if (source.getType().isa<LLVM::LLVMPointerType>())
    return source;
  /// If it is not a pointer, cast it to a pointer
  auto srcTy = source.getType().dyn_cast<MemRefType>();
  /// Return source if not a MemRef type
  if (!srcTy)
    return source;
  auto valPtr = create<polygeist::Memref2PointerOp>(
      loc, LLVM::LLVMPointerType::get(srcTy), source);
  auto valTy = mlir::MemRefType::get(srcTy.getShape(), VoidPtr,
                                     srcTy.getLayout(), srcTy.getMemorySpace());
  return create<polygeist::Pointer2MemrefOp>(loc, valTy, valPtr);
}

/// Casting
Value ArtsCodegen::castParameter(mlir::Type targetType, Value source,
                                 Location loc) {
  assert(targetType.isIntOrIndexOrFloat() && "Target type should be a number");
  /// If target type is an integer
  if (targetType.isa<IntegerType>())
    return castToInt(targetType, source, loc);
  /// If target type is an index
  if (targetType.isa<IndexType>())
    return castToIndex(source, loc);
  /// If target type is a float
  if (targetType.isa<FloatType>())
    return castToFloat(targetType, source, loc);
  return source;
}

Value ArtsCodegen::castPtr(mlir::Type targetType, Value source, Location loc) {
  auto srcType = source.getType().dyn_cast<MemRefType>();
  auto dstType = targetType.dyn_cast<MemRefType>();
  assert((srcType && dstType) && "Expected memref type");

  /// Cast if the types are compatible
  if (memref::CastOp::areCastCompatible(srcType, dstType))
    return create<memref::CastOp>(loc, dstType, source);

  /// If the types are not compatible, cast to LLVM pointer and then to memref
  return create<polygeist::Pointer2MemrefOp>(loc, targetType,
                                             createPtr(source, loc));
}

Value ArtsCodegen::castToIndex(Value source, Location loc) {
  if (source.getType().isIndex())
    return source;
  /// If it is not an index, cast it to an index.
  auto indexType = IndexType::get(source.getContext());
  return create<arith::IndexCastOp>(loc, indexType, source).getResult();
}

Value ArtsCodegen::castToFloat(mlir::Type targetType, Value source,
                               Location loc) {
  assert(targetType.isa<FloatType>() && "Target type should be a float");
  if (source.getType().isa<FloatType>())
    return source;
  /// If it is not a float, cast it to a float.
  return create<arith::SIToFPOp>(loc, targetType, source).getResult();
}

Value ArtsCodegen::castToInt(Type targetType, Value source, Location loc) {
  assert(targetType.isa<IntegerType>() &&
         "Target type should be an integer (e.g. i64, i32, etc.)");

  /// If types match exactly, no cast is needed.
  if (source.getType() == targetType)
    return source;

  /// If source is index => use IndexCastOp to target integer type
  if (source.getType().isIndex())
    return create<arith::IndexCastOp>(loc, targetType, source);

  /// If source is float => use FPToSIOp
  if (source.getType().isa<FloatType>())
    return create<arith::FPToSIOp>(loc, targetType, source);

  /// If source is integer => handle extension or truncation
  if (auto srcIntType = source.getType().dyn_cast<IntegerType>()) {
    auto dstIntType = targetType.dyn_cast<IntegerType>();
    if (!dstIntType)
      return source;
    unsigned srcWidth = srcIntType.getWidth();
    unsigned dstWidth = dstIntType.getWidth();

    if (srcWidth == dstWidth)
      return source;
    else if (srcWidth < dstWidth)
      return create<arith::ExtSIOp>(loc, targetType, source);
    return create<arith::TruncIOp>(loc, targetType, source);
  }

  /// If none of the above matched => unsupported type
  ARTS_ERROR("Unsupported type for casting to integer: " << source);
  assert(false && "Unsupported type in castToInt");
  return nullptr;
}

Value ArtsCodegen::castToVoidPtr(Value source, Location loc) {
  auto valPtr = source;
  if (!valPtr.getType().isa<LLVM::LLVMPointerType>())
    valPtr = castToLLVMPtr(source, loc);
  return create<polygeist::Pointer2MemrefOp>(loc, VoidPtr, valPtr);
}

Value ArtsCodegen::castToLLVMPtr(Value source, Location loc) {
  auto ptrType = getLLVMPointerType(source);
  if (!ptrType)
    return source;
  return create<polygeist::Memref2PointerOp>(loc, ptrType, source);
}

void ArtsCodegen::collectGlobalLLVMStrings() {
  for (auto &op : module.getOps()) {
    if (auto global = dyn_cast<LLVM::GlobalOp>(op))
      llvmStringGlobals[global.getName().str()] = global;
  }
}

Value ArtsCodegen::getOrCreateGlobalLLVMString(Location loc, StringRef value) {
  /// Check if we already have this string cached. If so, return the cached
  /// global
  if (llvmStringGlobals.find(value.str()) == llvmStringGlobals.end()) {
    OpBuilder::InsertionGuard IG(getBuilder());
    getBuilder().setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMArrayType::get(Int8, value.size() + 1);

    /// Create an unique name for the global
    std::string globalName = "str_" + std::to_string(llvmStringGlobals.size());
    llvmStringGlobals[value.str()] = create<LLVM::GlobalOp>(
        loc, type, true, LLVM::Linkage::Internal, globalName,
        getBuilder().getStringAttr(value.str() + '\0'));
  }

  LLVM::GlobalOp global = llvmStringGlobals[value.str()];
  return create<LLVM::AddressOfOp>(loc, global);
}

void ArtsCodegen::createPrintfCall(Location loc, llvm::StringRef format,
                                   ValueRange args) {
  if (!debug)
    return;

  /// Create the printf function declaration if it doesn't exist
  auto printfFn = module.lookupSymbol<LLVM::LLVMFuncOp>("printf");
  if (!printfFn) {
    auto printfType = LLVM::LLVMFunctionType::get(Int32, {llvmPtr}, true);
    printfFn = create<LLVM::LLVMFuncOp>(loc, "printf", printfType);
    printfFn.setLinkage(LLVM::Linkage::External);
  }
  assert(printfFn && "printf function not found");

  /// Get or create the format string global and cast it to a generic pointer
  auto formatStrPtr = getOrCreateGlobalLLVMString(loc, format);
  auto castedFormatPtr = create<LLVM::BitcastOp>(loc, llvmPtr, formatStrPtr);

  /// Create the printf call with all arguments
  SmallVector<Value> callArgs;
  callArgs.push_back(castedFormatPtr);
  callArgs.append(args.begin(), args.end());

  /// Create the call operation with a symbol reference
  create<LLVM::CallOp>(loc, TypeRange{getBuilder().getI32Type()},
                       SymbolRefAttr::get(getContext(), printfFn.getName()),
                       callArgs);
}

/// Insertion point management
void ArtsCodegen::setInsertionPoint(Operation *op) {
  getBuilder().setInsertionPoint(op);
}

void ArtsCodegen::setInsertionPointToStart(Block *block) {
  getBuilder().setInsertionPointToStart(block);
}

void ArtsCodegen::setInsertionPointToEnd(Block *block) {
  getBuilder().setInsertionPointToEnd(block);
}

void ArtsCodegen::setInsertionPointAfter(Operation *op) {
  getBuilder().setInsertionPointAfter(op);
}

void ArtsCodegen::setInsertionPoint(ModuleOp &module) {
  getBuilder().setInsertionPointToStart(module.getBody());
}

Operation *ArtsCodegen::clone(Operation &op) { return getBuilder().clone(op); }

Operation *ArtsCodegen::clone(Operation &op, IRMapping &mapper) {
  return getBuilder().clone(op, mapper);
}
//===----------------------------------------------------------------------===//
// Memref helpers
//===----------------------------------------------------------------------===//

LLVM::LLVMPointerType ArtsCodegen::getLLVMPointerType(Value source) {
  if (source.getType().isa<LLVM::LLVMPointerType>())
    return nullptr;
  auto MT = source.getType().dyn_cast<MemRefType>();
  if (!MT)
    return nullptr;

  return LLVM::LLVMPointerType::get(getContext(), MT.getMemorySpaceAsInt());
}

Value ArtsCodegen::computeElementTypeSize(Type elementType, Location loc) {
  /// Insert polygeist type size op
  auto elementSize = create<polygeist::TypeSizeOp>(
      loc, IndexType::get(getContext()), elementType);
  return elementSize;
}

Value ArtsCodegen::computeTotalElements(ValueRange sizes, Location loc) {
  if (sizes.empty())
    return createIndexConstant(1, loc);

  Value total = sizes[0];
  for (size_t i = 1; i < sizes.size(); ++i)
    total = create<arith::MulIOp>(loc, total, sizes[i]);
  return total;
}

Value ArtsCodegen::computeLinearIndex(ArrayRef<Value> sizes,
                                      ArrayRef<Value> indices, Location loc) {
  /// Convert multi-dimensional indices to linear index for 1D memref
  /// Formula: linear_index = i0 * (size1 * size2 * ... * sizeN) +
  ///                        i1 * (size2 * size3 * ... * sizeN) +
  ///                        ...
  ///                        iN
  if (indices.size() <= 1)
    return indices.empty() ? createIndexConstant(0, loc) : indices[0];

  /// If all sizes are constants, we can compute strides statically
  bool allSizesConstant = true;
  SmallVector<int64_t> constSizes;
  for (Value sz : sizes) {
    if (auto constOp = sz.getDefiningOp<arith::ConstantIndexOp>()) {
      constSizes.push_back(constOp.value());
    } else {
      allSizesConstant = false;
      break;
    }
  }

  Value linearIndex = createIndexConstant(0, loc);
  if (allSizesConstant) {
    /// pre-compute strides when all sizes are constants
    SmallVector<int64_t> strides(sizes.size());
    strides.back() = 1;
    for (int i = static_cast<int>(sizes.size()) - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * constSizes[i + 1];

    for (size_t i = 0; i < indices.size(); ++i) {
      if (auto constIdx = indices[i].getDefiningOp<arith::ConstantIndexOp>()) {
        /// Constant folding: compute contribution at compile time
        int64_t contribution = constIdx.value() * strides[i];
        if (contribution != 0) {
          auto constContrib = createIndexConstant(contribution, loc);
          linearIndex = create<arith::AddIOp>(loc, linearIndex, constContrib);
        }
      } else {
        /// Dynamic index: compute stride * index
        auto stride = createIndexConstant(strides[i], loc);
        auto contribution = create<arith::MulIOp>(loc, indices[i], stride);
        linearIndex = create<arith::AddIOp>(loc, linearIndex, contribution);
      }
    }
  } else {
    /// Fallback: original dynamic computation
    for (size_t i = 0; i < indices.size(); ++i) {
      Value stride = createIndexConstant(1, loc);
      for (size_t j = i + 1; j < sizes.size(); ++j)
        stride = create<arith::MulIOp>(loc, stride, sizes[j]);

      auto contribution = create<arith::MulIOp>(loc, indices[i], stride);
      linearIndex = create<arith::AddIOp>(loc, linearIndex, contribution);
    }
  }

  return linearIndex;
}

Value ArtsCodegen::computeLinearIndexFromStrides(ValueRange strides,
                                                 ValueRange indices,
                                                 Location loc) {
  if (indices.empty())
    return createIndexConstant(0, loc);

  Value linearIndex = createIndexConstant(0, loc);

  /// Use strides directly: linear_index = sum(indices[i] * strides[i])
  for (size_t i = 0; i < indices.size() && i < strides.size(); ++i) {
    auto contribution = create<arith::MulIOp>(loc, indices[i], strides[i]);
    linearIndex = create<arith::AddIOp>(loc, linearIndex, contribution);
  }

  return linearIndex;
}

SmallVector<Value> ArtsCodegen::computeStridesFromSizes(ArrayRef<Value> sizes,
                                                        Location loc) {
  if (sizes.empty())
    return {};

  SmallVector<Value> strides(sizes.size());
  Value stride = createIndexConstant(1, loc);

  // Compute strides from right to left (row-major)
  for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    if (i > 0)
      stride = create<arith::MulIOp>(loc, stride, sizes[i]);
  }

  return strides;
}

/// Helper to iterate over all Db elements
void ArtsCodegen::iterateDbElements(
    Value dbGuid, Value edtGuid, ArrayRef<Value> dbSizes,
    ArrayRef<Value> dbOffsets, bool isSingle, Location loc,
    std::function<void(Value)> elementCallback) {

  /// If the db sizes are empty, we are dealing with a single element
  if (dbSizes.empty()) {
    auto zeroIndex = createIndexConstant(0, loc);
    elementCallback(zeroIndex);
    return;
  }

  /// If the db is single, we can use a single loop
  if (isSingle) {
    /// The loop goes from the (offset) to (offset + size)
    auto lowerBound = dbOffsets[0];
    auto upperBound = create<arith::AddIOp>(loc, lowerBound, dbSizes[0]);
    auto stepConstant = createIndexConstant(1, loc);
    auto loopOp = create<scf::ForOp>(loc, lowerBound, upperBound, stepConstant);
    auto &loopBlock = loopOp.getRegion().front();
    setInsertionPointToStart(&loopBlock);
    elementCallback(loopOp.getInductionVar());
    setInsertionPointAfter(loopOp);
    return;
  }

  /// Otherwise, we need to iterate over all dimensions of the db
  iterateMultiDb(dbGuid, edtGuid, dbSizes, dbOffsets, loc, elementCallback);
}

/// Helper to iterate over multi-dimensional db elements
void ArtsCodegen::iterateMultiDb(Value dbGuid, Value edtGuid,
                                 ArrayRef<Value> dbSizes,
                                 ArrayRef<Value> dbOffsets, Location loc,
                                 std::function<void(Value)> elementCallback) {
  const unsigned dbRank = dbSizes.size();
  auto stepConstant = createIndexConstant(1, loc);

  std::function<void(unsigned, SmallVector<Value, 4> &)>
      addDependenciesRecursive = [&](unsigned dim,
                                     SmallVector<Value, 4> &indices) {
        if (dim == dbRank) {
          auto linearIndex = computeLinearIndex(dbSizes, indices, loc);
          elementCallback(linearIndex);
          return;
        }
        /// The loop goes from (offset) for this dimension
        auto lowerBound = dbOffsets[dim];
        auto upperBound = create<arith::AddIOp>(loc, lowerBound, dbSizes[dim]);
        auto loopOp =
            create<scf::ForOp>(loc, lowerBound, upperBound, stepConstant);

        auto &loopBlock = loopOp.getRegion().front();
        setInsertionPointToStart(&loopBlock);
        indices.push_back(loopOp.getInductionVar());
        addDependenciesRecursive(dim + 1, indices);
        indices.pop_back();
        setInsertionPointAfter(loopOp);
      };

  SmallVector<Value, 4> initIndices;
  addDependenciesRecursive(0, initIndices);
}

//===----------------------------------------------------------------------===//
// Debug printing helpers
//===----------------------------------------------------------------------===//

void ArtsCodegen::printDebugInfo(Location loc, const Twine &message,
                                 ValueRange args) {
  if (!debug)
    return;

  SmallString<128> formattedMessage;
  message.toVector(formattedMessage);
  formattedMessage.append("\n");
  createPrintfCall(loc, formattedMessage, args);
}

void ArtsCodegen::printDebugValue(Location loc, const Twine &label,
                                  Value value) {
  if (!debug)
    return;

  SmallString<128> formattedLabel;
  label.toVector(formattedLabel);
  formattedLabel.append(": %lu\n");

  auto castedValue = castToInt(Int64, value, loc);
  createPrintfCall(loc, formattedLabel, {castedValue});
}
