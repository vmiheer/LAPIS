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

#ifndef KOKKOS_MLIR_INITALLPASSES_H_
#define KOKKOS_MLIR_INITALLPASSES_H_

#include "mlir/InitAllPasses.h"
#include "mlir/Dialect/SparseTensor/Pipelines/kmPasses.h"


namespace mlir {


inline void registerAllKokkosPasses() {
  std::cout << " registerAllKokkosPasses called "<< std::endl;
  // Dialect pipelines
  sparse_tensor::registerSparseTensorKokkosPipelines();
}

} // namespace mlir

#endif // KOKKOS_MLIR_INITALLPASSES_H_
