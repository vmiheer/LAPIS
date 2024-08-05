//===- KokkosLoopMapping.cpp - Pattern for kokkos-loop-mapping pass --------------------===//
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

// Get the parallel nesting depth of the given Op
// - If Op itself is a kokkos.parallel or scf.parallel, then that counts as 1
// - Otherwise, Op counts for 0
// - Each enclosing parallel counts for 1 more
int getOpParallelDepth(Operation* op)
{
  int depth = 0;
  if(isa<scf::ParallelOp>(op) || isa<kokkos::RangeParallelOp>(op) || isa<kokkos::TeamParallelOp>(op))
    depth++;
  Operation* parent = op->getParentOp();
  if(parent)
    return depth + getOpParallelDepth(parent);
  // op has no parent
  return depth;
}

// Get the number of parallel nesting levels for the given ParallelOp
// - The op itself counts as 1
// - Each additional nesting level counts as another
int getParallelNumLevels(scf::ParallelOp op)
{
  int depth = 1;
  op->walk([&](scf::ParallelOp child) {
    int childDepth = getOpParallelDepth(child);
    if(childDepth > depth)
      depth = childDepth;
  });
  return depth;
}

// Rewrite the given scf.parallel as a kokkos.parallel, with the given execution space and nesting level
// (not for TeamPolicy loops)
LogicalResult scfParallelToKokkosRange(RewriterBase& rewriter, scf::ParallelOp op, kokkos::ExecutionSpace exec, kokkos::ParallelLevel level)
{
  rewriter.setInsertionPoint(op);
  // Create the kokkos.parallel but don't populate the body yet
  auto newOp = rewriter.create<kokkos::RangeParallelOp>(
    op.getLoc(), exec, level, op.getUpperBound(), op.getInitVals(), nullptr);
  // Now inline the old loop's operations into the new loop (replacing all usages of the induction variables)
  rewriter.inlineBlockBefore(op.getBody(), newOp.getBody(), newOp.getBody()->end(), newOp.getInductionVars());
  if(level == kokkos::ParallelLevel::TeamThread) {
    // Ops in this loop are executed in a TeamThread context.
    // This means that kokkos.single is required around any ops with side effects,
    // to make sure that they only happen once per thread as intended.
    SmallVector<Operation*> opsToWrap;
    // The following is careful to not mutate a sequence (block's operations) while iterating over it.
    // Instead, it makes a list of ops to replace upfront and then doing all the replacements without iterating.
    for(Operation& op : newOp.getBody()->getOperations()) {
      // TODO: find a more rigorous way to figure out if an op has side effects that we care about
      if(isa<memref::StoreOp>(op) || isa<memref::AtomicRMWOp>(op)) {
        opsToWrap.push_back(&op);
      }
    }
    for(Operation* op : opsToWrap)
    {
      rewriter.setInsertionPoint(op);
      // The single has the same set of result types as the original op
      auto single = rewriter.create<kokkos::SingleOp>(op->getLoc(), op->getResultTypes(), kokkos::SingleLevel::PerThread);
      auto singleBody = rewriter.createBlock(&single.getRegion());
      rewriter.setInsertionPointToStart(singleBody);
      rewriter.clone(*op);
      rewriter.create<scf::YieldOp>(op->getLoc(), op->getResults());
      rewriter.replaceOp(op, single);
    }
  }
  // The new loop is fully constructed.
  // As a last step, erase the old loop and replace uses of its results with those of the new loop.
  rewriter.replaceOp(op, newOp);
  return success();
}

/*
// Convert an scf.parallel to a TeamPolicy parallel_for, with a 1-dimensional 
LogicalResult scfParallelToKokkosTeam(RewriterBase& rewriter, scf::ParallelOp op, Value leagueSize, Value teamSizeHint, Value vectorLengthHint)
{
  rewriter.setInsertionPoint(op);
  // Create the kokkos.parallel but don't populate the body yet
  auto newOp = rewriter.create<kokkos::TeamParallelOp>(
    op.getLoc(), leagueSize, teamSizeHint, vectorLengthHint, op.getInitVals(), nullptr);
  // Now inline the old loop's operations into the new loop.
  // The team's block arguments are 
  rewriter.inlineBlockBefore(op.getBody(), newOp.getBody(), newOp.getBody()->end(), ValueRange(newOp.getInductionVars());
  // Ops in this loop are executed in a Team context.
  // This means that kokkos.single is required around any ops with side effects,
  // to make sure that they only happen once per team as intended.
  SmallVector<Operation*> opsToWrap;
  // This code is careful to not mutate a sequence (block's operations) while iterating over it.
  // Instead, it makes a list of ops to replace upfront and then does all the replacements without iterating.
  for(Operation& op : newOp.getBody()->getOperations()) {
    // TODO: find a more rigorous way to figure out if an op has side effects that we care about
    if(isa<memref::StoreOp>(op) || isa<memref::AtomicRMWOp>(op)) {
      opsToWrap.push_back(&op);
    }
  }
  for(Operation* op : opsToWrap)
  {
    rewriter.setInsertionPoint(op);
    // The single has the same set of result types as the original op
    auto single = rewriter.create<kokkos::SingleOp>(op->getLoc(), op->getResultTypes(), kokkos::SingleLevel::PerTeam);
    auto singleBody = rewriter.createBlock(&single.getRegion());
    rewriter.setInsertionPointToStart(singleBody);
    rewriter.clone(*op);
    rewriter.create<scf::YieldOp>(op->getLoc(), op->getResults());
    rewriter.replaceOp(op, single);
  }
  // The new loop is fully constructed.
  // As a last step, erase the old loop and replace uses of its results with those of the new loop.
  rewriter.replaceOp(op, newOp);
  return success();
}
*/

LogicalResult scfParallelToSequential(RewriterBase& rewriter, scf::ParallelOp op)
{
  using ValueVector = SmallVector<Value>;
  rewriter.setInsertionPoint(op);
  mlir::scf::LoopNest newLoopNest = scf::buildLoopNest(
    rewriter, op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(), op.getInitVals(),
    [&](OpBuilder& builder, Location loc, ValueRange inductionVars, ValueRange args) -> ValueVector
    {
      ValueVector results;
      // Just clone the statements of op's body.
      // When the scf.reduce is encountered, if any:
      // - for each of its n regions, promote all normal ops into the loop body
      // - reduce has n operands, one per reduction. These are partial reductions to update.
      // - each region i has 1 block, whose 2 args are the partial reductions to join
      //   - LHS = implicit partial reduction
      //   - RHS = ith reduce operand (some value computed in the body)
      // - Replace LHS with the ith loop argument
      // - just delete all scf.reduce.return inside reduce blocks. Instead append the joined result to results.
      IRMapping irMap;
      for (auto p : llvm::zip(op.getInductionVars(), inductionVars)) {
        irMap.map(std::get<0>(p), std::get<1>(p));
      }
      for(Operation& oldOp : op.getBody()->getOperations()) {
        if(auto reduce = dyn_cast<scf::ReduceOp>(oldOp)) {
        }
        else {
          auto newOp = builder.clone(oldOp, irMap);
          for (auto p : llvm::zip(oldOp.getResults(), newOp->getResults())) {
            irMap.map(std::get<0>(p), std::get<1>(p));
          }
        }
      }
      return results;
    });
  rewriter.replaceOp(op, newLoopNest.loops.front());
  return success();
}

struct KokkosLoopRewriter : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  KokkosLoopRewriter(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {
    // Only match with top-level ParallelOps (meaning op is not enclosed in another ParallelOp)
    if(op->getParentOfType<scf::ParallelOp>())
      return failure();
    // Determine the maximum depth of parallel nesting (a simple RangePolicy is 1, etc.)
    int nestingLevel = getParallelNumLevels(op);
    // Now decide whether this op should execute on device (offloaded) or host.
    // Operations that are assumed to be host-only:
    // - func.call
    // - any memref allocation or deallocation
    // This is conservative as there are obviously functions that work safely on the device,
    // but an inlining pass could work around that easily
    bool canBeOffloaded = true;
    op->walk([&](func::CallOp) {
        canBeOffloaded = false;
    });
    op->walk([&](memref::AllocOp) {
        canBeOffloaded = false;
    });
    op->walk([&](memref::AllocaOp) {
        canBeOffloaded = false;
    });
    op->walk([&](memref::DeallocOp) {
        canBeOffloaded = false;
    });
    op->walk([&](memref::ReallocOp) {
        canBeOffloaded = false;
    });
    kokkos::ExecutionSpace exec = canBeOffloaded ? kokkos::ExecutionSpace::Device : kokkos::ExecutionSpace::Host;

    // Possible cases for exec == Device:
    //
    // - Depth 1: RangePolicy (or MDRangePolicy, both have same representation in the dialect)
    // - Depth 2: TeamPolicy with one thread (simd) per inner work-item (best for spmv-like patterns)
    //            TODO: Write a heuristic to choose TeamPolicy/TeamVector instead, for when the inner loop
    //            requires more parallelism
    // - Depth 3: TeamPolicy/TeamThread/ThreadVector nested parallelism
    // - Depth >3: Use TeamPolicy/TeamThread for outermost two loops, and ThreadVector for innermost loop.
    //             Better coalescing that way, if data layout is correct for the loop structure.
    //             Serialize all other loops by replacing them with scf.for.
    //
    // For exec == Host, just parallelize the outermost loop with RangePolicy and serialize the inner loops.
    if(nestingLevel == 1)
    {
      return scfParallelToKokkosRange(rewriter, op, exec, kokkos::ParallelLevel::RangePolicy);
    }
    else if(nestingLevel == 2)
    {
      //TODO
    }
    else if(nestingLevel >= 3)
    {
      //TODO
    }
    if(nestingLevel > 3)
    {
      //TODO
    }
    return success();
  }
};

} // namespace

void mlir::populateKokkosLoopMappingPatterns(RewritePatternSet &patterns)
{
  patterns.add<KokkosLoopRewriter>(patterns.getContext());
}

