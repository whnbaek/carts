//===- Dialect.h - Arts dialect --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CARTS_DIALECT_H
#define CARTS_DIALECT_H

/// Dialects
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
/// Others
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMapInfo.h"

//===----------------------------------------------------------------------===//
// Arts Dialect
//===----------------------------------------------------------------------===//
#include "arts/ArtsOpsDialect.h.inc"

bool isArtsRegion(mlir::Operation *op);
bool isArtsOp(mlir::Operation *op);

//===----------------------------------------------------------------------===//
// Arts Dialect Types
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "arts/ArtsOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// Arts Dialect Enums
//===----------------------------------------------------------------------===//
#include "arts/ArtsOpsEnums.h.inc"

//===----------------------------------------------------------------------===//
// Arts Dialect Attributes
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "arts/ArtsOpsAttributes.h.inc"

//===----------------------------------------------------------------------===//
// Arts Dialect Operations
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "arts/ArtsOps.h.inc"

//===----------------------------------------------------------------------===//
// Arts Dialect Utility Functions
//===----------------------------------------------------------------------===//

/// Helper function to extract sizes from a datablock pointer
/// by finding the original DbAllocOp or DbAcquireOp that created it
mlir::SmallVector<mlir::Value> getSizesFromDb(mlir::Operation *dbOp);
mlir::SmallVector<mlir::Value> getElementSizesFromDb(mlir::Operation *dbOp);
mlir::SmallVector<mlir::Value> getSizesFromDb(mlir::Value datablockPtr);
mlir::SmallVector<mlir::Value> getOffsetsFromDb(mlir::Value datablockPtr);

/// Check if a datablock operation has a single size of 1
bool dbHasSingleSize(mlir::Operation *dbOp);

#endif // CARTS_DIALECT_H
