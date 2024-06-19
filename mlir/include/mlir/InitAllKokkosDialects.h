//===- InitAllDialects.h - MLIR Dialects Registration -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef KOKKOS_MLIR_INITALLDIALECTS_H_
#define KOKKOS_MLIR_INITALLDIALECTS_H_

#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/PartTensor/IR/PartTensor.h"

namespace mlir {

/// Add all the MLIR dialects to the provided registry.
inline void registerAllKokkosDialects(DialectRegistry &registry) {
  registerAllDialects(registry);
  // clang-format off
  registry.insert<part_tensor::PartTensorDialect>();
  // clang-format on
}

/// Append all the MLIR dialects to the registry contained in the given context.
inline void registerAllKokkosDialects(MLIRContext &context) {
  DialectRegistry registry;
  registerAllKokkosDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace mlir

#endif // KOKKOS_MLIR_INITALLDIALECTS_H_
