//===- KokkosDualViewManagement.cpp - Pattern for kokkos-dualview-management pass --------------------===//
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

// Does the value live inside scope, or an ancester of scope?
static bool valueInScope(Value v, Region* scope)
{
  return v.getParentRegion()->isAncestor(scope);
}

struct KokkosDualViewRewriter : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  KokkosDualViewRewriter (MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult insertSyncModifyChild(Region* region, const DenseSet<Value>& memrefs, PatternRewriter& rewriter) const {
    for(Operation& op : region->getOps()) {
      rewriter.setInsertionPoint(&op);
      DenseSet<Value> deviceReads = kokkos::getMemrefsRead(&op, kokkos::ExecutionSpace::Device);
      DenseSet<Value> hostReads = kokkos::getMemrefsRead(&op, kokkos::ExecutionSpace::Host);
      DenseSet<Value> deviceWrites = kokkos::getMemrefsWritten(&op, kokkos::ExecutionSpace::Device);
      DenseSet<Value> hostWrites = kokkos::getMemrefsWritten(&op, kokkos::ExecutionSpace::Host);
      DenseSet<Value> allMemrefs;
      allMemrefs.insert(deviceReads.begin(), deviceReads.end());
      allMemrefs.insert(hostReads.begin(), hostReads.end());
      allMemrefs.insert(deviceWrites.begin(), deviceWrites.end());
      allMemrefs.insert(hostWrites.begin(), hostWrites.end());
      DenseSet<Value> memrefsForChildren;
      for(Value v : allMemrefs) {
        // Skip memrefs that were already handled at a higher scope
        if(!memrefs.contains(v))
          continue;
        // Check conditions for inserting sync/modify before op
        bool dr = deviceReads.contains(v);
        bool dw = deviceWrites.contains(v);
        bool hr = hostReads.contains(v);
        bool hw = hostWrites.contains(v);
        bool inScope = valueInScope(v, region);
        bool usedInOneSpace = !((dr || dw) && (hr || hw));
        bool readOnly = !dw && !hw;
        if(inScope && (usedInOneSpace || readOnly)) {
          // Then we can insert sync and/or modify calls before op.
          // Modifies must go after syncs, otherwise the sync would
          // immediately trigger a copy.
          if(dr) rewriter.create<kokkos::SyncOp>(op.getLoc(), v, kokkos::MemorySpace::Device);
          if(hr) rewriter.create<kokkos::SyncOp>(op.getLoc(), v, kokkos::MemorySpace::Host);
          if(dw) rewriter.create<kokkos::ModifyOp>(op.getLoc(), v, kokkos::MemorySpace::Device);
          if(hw) rewriter.create<kokkos::ModifyOp>(op.getLoc(), v, kokkos::MemorySpace::Host);
        }
        else {
          // Need to handle this memref inside subregions of op.
          memrefsForChildren.insert(v);
        }
      }
      // Recurse into subregions and insert the sync/modify that we couldn't before.
      if(memrefsForChildren.size()) {
        for(Region& subregion : op.getRegions()) {
          if(failed(insertSyncModifyChild(&subregion, memrefsForChildren, rewriter)))
            return failure();
        }
      }
    }
    return success();
  }

  LogicalResult matchAndRewrite(func::FuncOp func, PatternRewriter &rewriter) const override {
    // For each top-level op:
    // - List the memrefs that it reads and writes (including in subregions) separately on host and device
    // - Ignore any memrefs not implemented using DualView.
    // - Then separate out the memrefs whose region scope is not an ancestor of op's scope.
    // - For each memref that is either used in only one space, or read in both spaces,
    //   insert appropriate sync and modify calls before the op.
    // - Put all other memrefs (either used in both spaces, or belonging to a child region) into a list
    //   (memrefsForChildren) and recursively insert DualView handling before ops in child regions.
    for(Region& reg : func->getRegions()) {
      for(Operation& op : reg.getOps()) {
        rewriter.setInsertionPoint(&op);
        DenseSet<Value> deviceReads = kokkos::getMemrefsRead(&op, kokkos::ExecutionSpace::Device);
        DenseSet<Value> hostReads = kokkos::getMemrefsRead(&op, kokkos::ExecutionSpace::Host);
        DenseSet<Value> deviceWrites = kokkos::getMemrefsWritten(&op, kokkos::ExecutionSpace::Device);
        DenseSet<Value> hostWrites = kokkos::getMemrefsWritten(&op, kokkos::ExecutionSpace::Host);
        DenseSet<Value> allMemrefs;
        allMemrefs.insert(deviceReads.begin(), deviceReads.end());
        allMemrefs.insert(hostReads.begin(), hostReads.end());
        allMemrefs.insert(deviceWrites.begin(), deviceWrites.end());
        allMemrefs.insert(hostWrites.begin(), hostWrites.end());
        DenseSet<Value> memrefsForChildren;
        for(Value v : allMemrefs) {
          // Check conditions for inserting sync/modify before op
          bool dr = deviceReads.contains(v);
          bool dw = deviceWrites.contains(v);
          bool hr = hostReads.contains(v);
          bool hw = hostWrites.contains(v);
          bool inScope = valueInScope(v, &reg);
          bool usedInOneSpace = !((dr || dw) && (hr || hw));
          bool readOnly = !dw && !hw;
          if(inScope && (usedInOneSpace || readOnly)) {
            // Then we can insert sync and/or modify calls before op.
            // Modifies must go after syncs, otherwise the sync would
            // immediately trigger a copy.
            if(dr) rewriter.create<kokkos::SyncOp>(op.getLoc(), v, kokkos::MemorySpace::Device);
            if(hr) rewriter.create<kokkos::SyncOp>(op.getLoc(), v, kokkos::MemorySpace::Host);
            if(dw) rewriter.create<kokkos::ModifyOp>(op.getLoc(), v, kokkos::MemorySpace::Device);
            if(hw) rewriter.create<kokkos::ModifyOp>(op.getLoc(), v, kokkos::MemorySpace::Host);
          }
          else {
            // Need to handle this memref inside subregions of op.
            memrefsForChildren.insert(v);
          }
        }
        // Recurse into subregions and insert the sync/modify that we couldn't before.
        if(memrefsForChildren.size()) {
          for(Region& subregion : op.getRegions()) {
            if(failed(insertSyncModifyChild(&subregion, memrefsForChildren, rewriter)))
              return failure();
          }
        }
      }
    }
    return success();
  }
};

} // namespace

void mlir::populateKokkosDualViewManagementPatterns(RewritePatternSet &patterns)
{
  patterns.add<KokkosDualViewRewriter>(patterns.getContext());
}

