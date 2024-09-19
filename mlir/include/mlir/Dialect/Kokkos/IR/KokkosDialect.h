#ifndef MLIR_DIALECT_KOKKOS_DIALECT_H
#define MLIR_DIALECT_KOKKOS_DIALECT_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <optional>

#include "mlir/Dialect/Kokkos/IR/KokkosDialect.h.inc"
#include "mlir/Dialect/Kokkos/IR/KokkosEnums.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Kokkos/IR/Kokkos.h.inc"

namespace mlir {
namespace kokkos {

// Given a CallOp, find the FuncOp corresponding to the callee.
// Since CallOp can only do direct calls, this should always succeed.
func::FuncOp getCalledFunction(func::CallOp callOp);

// Get the top-level "parent" memref of v.
// If v is a block argument or result of an allocation,
// it is its own parent.
// But if it's the result of a view-like op
// (casting, slicing, reshaping) then the memref operand
// of that op is the parent of v.
Value getParentMemref(Value v);

// Determine the correct memory space (Host, Device or DualView)
// for v based on where it gets accessed.
MemorySpace getMemSpace(Value v);

} // namespace kokkos 
} // namespace mlir

#endif // MLIR_DIALECT_KOKKOS_DIALECT_H
