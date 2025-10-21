///==========================================================================
// File: ARTSPassDetails.h
//
// CARTS pass class details
///==========================================================================
#ifndef DIALECT_CARTS_TRANSFORMS_PASSDETAILS_H
#define DIALECT_CARTS_TRANSFORMS_PASSDETAILS_H

#include "arts/ArtsDialect.h"
#include "arts/Passes/ArtsPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class FunctionOpInterface;

// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arts {
/// Register passes
#define GEN_PASS_CLASSES
#include "arts/Passes/ArtsPasses.h.inc"

} // namespace arts
} // namespace mlir

#endif // DIALECT_CARTS_TRANSFORMS_PASSDETAILS_H
