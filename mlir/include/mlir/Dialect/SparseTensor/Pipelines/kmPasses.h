//===- Passes.h - Sparse tensor pipeline entry points -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all sparse tensor pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef LAPIS_DIALECT_SPARSETENSOR_PIPELINES_PASSES_H_
#define LAPIS_DIALECT_SPARSETENSOR_PIPELINES_PASSES_H_

#include "mlir/Dialect/SparseTensor/Pipelines/Passes.h"

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the "sparse-compiler" pipeline to the `OpPassManager`.  This
/// is the standard pipeline for taking sparsity-agnostic IR using
/// the sparse-tensor type and lowering it to LLVM IR with concrete
/// representations and algorithms for sparse tensors.
void buildSparseKokkosCompiler(OpPassManager &pm,
                         const SparseCompilerOptions &options);

/// Registers all pipelines for the `sparse_tensor` dialect.  At present,
/// this includes only "sparse-compiler".
void registerSparseTensorKokkosPipelines();

} // namespace sparse_tensor
} // namespace mlir

#endif // LAPIS_DIALECT_SPARSETENSOR_PIPELINES_PASSES_H_
