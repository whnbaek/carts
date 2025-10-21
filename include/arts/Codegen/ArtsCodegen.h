///==========================================================================
/// File: ArtsCodegen.h
///==========================================================================
#ifndef CARTS_CODEGEN_ARTSCODEGEN_H
#define CARTS_CODEGEN_ARTSCODEGEN_H

#include "arts/ArtsDialect.h"
#include "arts/Codegen/ArtsTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace arts {
using namespace types;

class ArtsCodegen {
public:
  explicit ArtsCodegen(ModuleOp module, bool debug = false);
  ArtsCodegen(ModuleOp &module, llvm::DataLayout &llvmDL,
              mlir::DataLayout &mlirDL, bool debug = false);
  ~ArtsCodegen();

  /// Getters
  MLIRContext *getContext() { return getBuilder().getContext(); }
  OpBuilder &getBuilder() { return activeRewriter ? *activeRewriter : builder; }
  ModuleOp &getModule() { return module; }

  /// Rewriter management
  void setRewriter(PatternRewriter *rewriter) { activeRewriter = rewriter; }
  void clearRewriter() { activeRewriter = nullptr; }

  /// RAII helper for managing rewriter context
  class RewriterGuard {
  public:
    RewriterGuard(ArtsCodegen &ac, PatternRewriter &rewriter) : AC(ac) {
      AC.setRewriter(&rewriter);
    }
    ~RewriterGuard() { AC.clearRewriter(); }

  private:
    ArtsCodegen &AC;
  };

  /// Create an operation of specific op type at the current insertion point.
  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args &&...args) {
    return getBuilder().create<OpTy>(location, std::forward<Args>(args)...);
  }

  llvm::DataLayout &getLLVMDataLayout() {
    assert(llvmDL && "LLVM DataLayout not initialized");
    return *llvmDL;
  }
  mlir::DataLayout &getMLIRDataLayout() {
    assert(mlirDL && "MLIR DataLayout not initialized");
    return *mlirDL;
  }
  bool isDebug() const { return debug; }
  Location getUnknownLoc() { return UnknownLoc::get(module.getContext()); }

  /// Runtime function management
  func::FuncOp getOrCreateRuntimeFunction(RuntimeFunction FnID);
  func::CallOp createRuntimeCall(RuntimeFunction FnID, ArrayRef<Value> args,
                                 Location loc);

  /// DB management
  void addDbDep(Value dbGuid, Value edtGuid, Value edtSlot, Location loc);
  void incrementDbLatchCount(Value dbGuid, Location loc);
  void decrementDbLatchCount(Value dbGuid, Location loc);

  /// Epoch management
  Value createEpoch(Value finishEdtGuid, Value finishEdtSlot, Location loc);

  /// Utility functions
  Value getCurrentEpochGuid(Location loc);
  Value getCurrentEdtGuid(Location loc);
  Value getTotalWorkers(Location loc);
  Value getTotalNodes(Location loc);
  Value getCurrentWorker(Location loc);
  Value getCurrentNode(Location loc);
  func::CallOp signalEdt(Value edtGuid, Value edtSlot, Value dbGuid,
                         Location loc);
  void waitOnHandle(Value epochGuid, Location loc);

  /// Function creation
  func::FuncOp insertInitPerWorker(Location loc);
  func::FuncOp insertInitPerNode(Location loc, func::FuncOp callback);
  func::FuncOp insertArtsMainFn(Location loc, func::FuncOp callback);
  func::FuncOp insertMainFn(Location loc);
  void initRT(Location loc);

  /// Helper functions
  Value createFnPtr(func::FuncOp funcOp, Location loc);
  Value createIndexConstant(int value, Location loc);
  Value createIntConstant(int value, Type type, Location loc);
  Value createPtr(Value source, Location loc);
  Value castParameter(Type targetType, Value source, Location loc);
  Value castPtr(Type targetType, Value source, Location loc);
  Value castToIndex(Value source, Location loc);
  Value castToFloat(Type targetType, Value source, Location loc);
  Value castToInt(Type targetType, Value source, Location loc);
  Value castToVoidPtr(Value source, Location loc);
  Value castToLLVMPtr(Value source, Location loc);

  /// Memref helpers
  LLVM::LLVMPointerType getLLVMPointerType(Value source);
  Value computeElementTypeSize(Type elementType, Location loc);
  Value computeTotalElements(ValueRange sizes, Location loc);
  Value computeLinearIndex(ArrayRef<Value> sizes, ArrayRef<Value> indices,
                           Location loc);
  Value computeLinearIndexFromStrides(ValueRange strides, ValueRange indices,
                                      Location loc);
  SmallVector<Value> computeStridesFromSizes(ArrayRef<Value> sizes,
                                             Location loc);

  /// Db iteration helpers
  void iterateDbElements(Value dbGuid, Value edtGuid, ArrayRef<Value> dbSizes,
                         ArrayRef<Value> dbOffsets, bool isSingle, Location loc,
                         std::function<void(Value)> elementCallback);
  void iterateMultiDb(Value dbGuid, Value edtGuid, ArrayRef<Value> dbSizes,
                      ArrayRef<Value> dbOffsets, Location loc,
                      std::function<void(Value)> elementCallback);

  /// Debug printing helpers
  void printDebugInfo(Location loc, const Twine &message, ValueRange args = {});
  void printDebugValue(Location loc, const Twine &label, Value value);

  /// Debug
  void collectGlobalLLVMStrings();
  Value getOrCreateGlobalLLVMString(Location loc, StringRef value);
  void createPrintfCall(Location loc, StringRef format, ValueRange args);

  /// Insertion point management
  void setInsertionPoint(Operation *op);
  void setInsertionPointToStart(Block *block);
  void setInsertionPointToEnd(Block *block);
  void setInsertionPointAfter(Operation *op);
  void setInsertionPoint(ModuleOp &module);

  /// Operation management
  Operation *clone(Operation &op);
  Operation *clone(Operation &op, IRMapping &mapper);

/// Types
///{
#define ARTS_TYPE(VarName, InitValue) Type VarName = nullptr;
#define ARTS_FUNCTION_TYPE(VarName, IsVarArg, ReturnType, ...)                 \
  FunctionType VarName = nullptr;                                              \
  MemRefType VarName##Ptr = nullptr;
#define ARTS_STRUCT_TYPE(VarName, StructName, ...)                             \
  LLVM::LLVMStructType VarName = nullptr;                                      \
  MemRefType VarName##Ptr = nullptr;
#include "arts/Codegen/ArtsKinds.def"
  LLVM::LLVMPointerType llvmPtr = nullptr;
  ///}

private:
  ModuleOp module;
  OpBuilder builder;
  PatternRewriter *activeRewriter = nullptr;
  std::unique_ptr<llvm::DataLayout> llvmDL;
  std::unique_ptr<mlir::DataLayout> mlirDL;
  bool debug = false;

  /// Other Attributes
  llvm::DenseMap<RuntimeFunction, func::FuncOp> runtimeFunctionCache;
  llvm::StringMap<LLVM::GlobalOp> llvmStringGlobals;

  /// Helper functions
  void initializeTypes();
  LogicalResult extractDataLayouts();
  void applyRuntimeFunctionAttributes(func::FuncOp funcOp,
                                      RuntimeFunction fnID);
};

} // namespace arts
} // namespace mlir

#endif // CARTS_CODEGEN_ARTSCODEGEN_H
