///==========================================================================
/// File: ArtsTypes.h - ARTS Runtime Type Definitions
///==========================================================================

#ifndef CARTS_CODEGEN_ARTSRTTYPES_H
#define CARTS_CODEGEN_ARTSRTTYPES_H

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/InstrTypes.h"
#include <sys/types.h>

#include "arts/ArtsDialect.h"

namespace mlir {
namespace arts {

// Forward declarations
class DbAllocOp;

namespace types {

/// IDs for all arts runtime library (RTL) functions.
#define ARTS_RTL_FUNCTIONS
enum class RuntimeFunction {
#define ARTS_RTL(Enum, ...) Enum,
#include "arts/Codegen/ArtsKinds.def"
};
#undef ARTS_RTL_FUNCTIONS

#define ARTS_RTL(Enum, ...)                                                    \
  constexpr auto Enum = arts::types::RuntimeFunction::Enum;
#define ARTS_RTL_FUNCTIONS
#include "arts/Codegen/ArtsKinds.def"
#undef ARTS_RTL_FUNCTIONS

// Include ARTS runtime constants
#define ARTS_CONSTANTS
#include "arts/Codegen/ArtsKinds.def"
#undef ARTS_CONSTANTS

} // end namespace types
} // end namespace arts
} // end namespace mlir

/// Specialization of DenseMapInfo for RuntimeFunction enum
namespace llvm {
template <> struct DenseMapInfo<mlir::arts::types::RuntimeFunction> {
  static mlir::arts::types::RuntimeFunction getEmptyKey() {
    return static_cast<mlir::arts::types::RuntimeFunction>(~0U);
  }

  static mlir::arts::types::RuntimeFunction getTombstoneKey() {
    return static_cast<mlir::arts::types::RuntimeFunction>(~0U - 1);
  }

  static unsigned getHashValue(mlir::arts::types::RuntimeFunction val) {
    return static_cast<unsigned>(val);
  }

  static bool isEqual(mlir::arts::types::RuntimeFunction lhs,
                      mlir::arts::types::RuntimeFunction rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

#endif // CARTS_CODEGEN_ARTSRTTYPES_H
