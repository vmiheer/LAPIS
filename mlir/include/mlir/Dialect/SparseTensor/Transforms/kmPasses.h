//===- Passes.h - Sparse tensor pass entry points ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all sparse tensor passes.
//
//===----------------------------------------------------------------------===//

#ifndef LAPIS_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES_H_
#define LAPIS_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES_H_

#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// The SparseKokkos pass.
//===----------------------------------------------------------------------===//
//
void populateSparseKokkosCodegenPatterns(RewritePatternSet &patterns);
std::unique_ptr<Pass> createSparseKokkosCodegenPass();

} // namespace mlir

#endif // LAPIS_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES_H_
