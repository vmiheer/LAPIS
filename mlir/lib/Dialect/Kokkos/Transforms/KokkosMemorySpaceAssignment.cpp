//===- KokkosMemorySpaceAssignment.cpp - Pattern for kokkos-assign-memory-spaces pass --------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Kokkos/IR/KokkosDialect.h"
#include "mlir/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace {

struct KokkosMemorySpaceRewriter : public OpRewritePattern<ModuleOp> {
  using OpRewritePattern<ModuleOp>::OpRewritePattern;

  KokkosMemorySpaceRewriter (MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(ModuleOp op, PatternRewriter &rewriter) const override {
    // For each new memref-typed value (block argument or one produced by an op),
    // iterate over all ops that use it and assign its memory space based on whether
    // those ops are in device or host code.
    //
    // Not all ops count as a "use" for this so skip: alloc/alloca/dealloc and slicing/viewing/casting ops.
    // These never actually access data. For example, an alloc may run on host and
    // allocate a device memref, but that doesn't mean the memref should also be accessible on host.
    return failure();
  }
};

} // namespace

void mlir::populateKokkosMemorySpaceAssignmentPatterns(RewritePatternSet &patterns)
{
  patterns.add<MemSpaceConverter>(typeConverter, patterns.getContext(), createSparseDeallocs);
  //patterns.add<KokkosMemorySpaceRewriter>(patterns.getContext());
}

