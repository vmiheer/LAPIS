//===- LinkAllPassesAndDialects.h - MLIR Registration -----------*- C++ -*-===//
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

#ifndef LAPIS_INITALLKOKKOSPASSES_H
#define LAPIS_INITALLKOKKOSPASSES_H

#include "mlir/InitAllPasses.h"
#include "mlir/Dialect/Kokkos/Pipelines/Passes.h"

namespace mlir {

inline void registerAllKokkosPasses() {
  // Dialect pipelines
  kokkos::registerKokkosPipelines();
}

} // namespace mlir

#endif // LAPIS_INITALLKOKKOSPASSES_H_
