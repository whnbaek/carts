///==========================================================================
/// File: StringAnalysis.cpp
///==========================================================================

#include "arts/Analysis/StringAnalysis.h"
#include "arts/Utils/ArtsUtils.h"
/// Dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "polygeist/Ops.h"
/// Others
#include "mlir/Transforms/RegionUtils.h"

/// Debug
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "arts/Utils/ArtsDebug.h"
ARTS_DEBUG_SETUP(string_analysis);

using namespace mlir;
using namespace arts;

void StringAnalysis::run() {
  ARTS_DEBUG_HEADER(StringAnalysis);
  trackGlobalUsers();
  trackMemrefInitializations();
  trackCallOps();
  ARTS_DEBUG_REGION({
    if (stringMemRefs.empty()) {
      ARTS_WARN("No string memrefs found");
    } else {
      ARTS_INFO("Found " << stringMemRefs.size() << " string memrefs");
    }
  });
  ARTS_DEBUG_FOOTER(StringAnalysis);
}

bool StringAnalysis::isStringGlobal(LLVM::GlobalOp globalOp) {
  if (auto attr = globalOp.getValueOrNull()) {
    if (auto strAttr = attr.dyn_cast<StringAttr>()) {
      /// We assume a string global is null-terminated.
      return strAttr.getValue().ends_with("\00");
    }
  }
  return false;
}

void StringAnalysis::trackGlobalUsers() {
  module.walk([&](LLVM::AddressOfOp addressOf) {
    auto globalOp =
        module.lookupSymbol<LLVM::GlobalOp>(addressOf.getGlobalName());
    /// Check if the addressOf references a global and if it's a string global.
    if (globalOp && isStringGlobal(globalOp)) {
      SmallVector<Value, 8> worklist{addressOf.getResult()};
      while (!worklist.empty()) {
        Value curr = worklist.pop_back_val();
        for (auto *curUser : curr.getUsers()) {
          // Process pointer arithmetic (GEP) and load operations.
          if (isa<LLVM::GEPOp, LLVM::LoadOp>(curUser)) {
            for (Value result : curUser->getResults()) {
              globalSources.try_emplace(result, globalOp);
              worklist.push_back(result);
            }
          }
        }
      }
    }
  });
}

void StringAnalysis::trackMemrefInitializations() {
  module.walk([&](Operation *op) {
    /// Check for affine store operations.
    if (auto affineStore = dyn_cast<affine::AffineStoreOp>(op)) {
      Value value = affineStore.getValue();
      if (globalSources.count(value))
        stringMemRefs.insert(affineStore.getMemRef());
    }
    /// Also check for non-affine memref store operations.
    else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      Value value = storeOp.getValueToStore();
      if (globalSources.count(value))
        stringMemRefs.insert(storeOp.getMemRef());
    }
  });
}

void StringAnalysis::trackCallOps() {
  module.walk([&](func::CallOp callOp) {
    if (isStringFunction(callOp.getCallee())) {
      for (Value operand : callOp.getOperands()) {
        /// If operand is a direct memref, cache it.
        if (operand.getType().isa<MemRefType>()) {
          stringMemRefs.insert(operand);
        }
        /// Otherwise, if operand is an LLVM pointer, try to recover the
        /// underlying memref.
        else if (operand.getType().isa<LLVM::LLVMPointerType>()) {
          if (auto defOp = operand.getDefiningOp()) {
            /// Check for polygeist conversion.
            if (isa<polygeist::Pointer2MemrefOp>(defOp)) {
              Value memrefOperand = defOp->getOperand(0);
              if (memrefOperand.getType().isa<MemRefType>())
                stringMemRefs.insert(memrefOperand);
            }
          }
        }
      }
    }
  });
}

bool StringAnalysis::isStringFunction(StringRef funcName) {
  /// Set of common string functions.
  const DenseSet<StringRef> stringFunctions = {
      "strlen", "strcpy", "strcat", "strcmp", "printf", "sprintf",
      "puts",   "fputs",  "scanf",  "fscanf", "gets"};
  return stringFunctions.contains(funcName);
}