// ===- Partition.h - Partition dialect -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_KOKKOS_DIALECT_H
#define MLIR_DIALECT_KOKKOS_DIALECT_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// Convenience functions can be declared here

} // namespace kokkos 
} // namespace mlir

#endif // MLIR_DIALECT_KOKKOS_DIALECT_H
