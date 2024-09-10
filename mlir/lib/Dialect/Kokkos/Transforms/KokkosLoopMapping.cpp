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
  if(isa<scf::ParallelOp>(op) || isa<kokkos::RangeParallelOp>(op) || isa<kokkos::TeamParallelOp>(op) || isa<kokkos::ThreadParallelOp>(op))
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
  int selfDepth = 0;
  int maxDepth = 0;
  //note: walk starting from op visits op itself
  op->walk([&](scf::ParallelOp child) {
    int d = getOpParallelDepth(child);
    if(op == child) {
      selfDepth = d;
    }
    else {
      if(d > maxDepth)
        maxDepth = d;
    }
  });
  return maxDepth - selfDepth;
}

// Within the context of a kokkos.team_parallel: does op imply team-wide synchronization?
// If false, then a kokkos.team_barrier may be required to make the
// side effects/result(s) of op visible to other threads.
//
// - team_barrier syncs
// - a TeamThread or TeamVector parallel reduce syncs, since these
//   behave like all-reduces
// - a per-team Kokkos::single with at least one result is a broadcast, so it syncs
//
// - a TeamVector or TeamThread parallel for (no reductions) is async
// - a Kokkos::single with no results is async
// - all other ops have no synchronization behavior and are considered async 
/*
bool doesTeamLevelSync(Operation* op)
{
  if(auto rangePar = dyn_cast<kokkos::RangeParallelOp>(op)) {
    auto level = rangePar.getParallelLevel();
    // check if rangePar a) has at least one result and b) is team-level
    return rangePar.getNumResults() &&
      (level == kokkos::ParallelLevel::TeamThread ||
      level == kokkos::ParallelLevel::TeamVector);
  }
  else if(auto single = dyn_cast<kokkos::SingleOp>(op)) {
    return single.getNumResults() && single.getLevel() == kokkos::SingleLevel::PerTeam;
  }
  else if(isa<kokkos::TeamBarrierOp>(op))
    return true;
  return false;
}
*/

// Compute the product of some Index-typed values.
// This assumes no overflow.
template<typename ValRange>
Value buildIndexProduct(Location loc, RewriterBase& rewriter, ValRange vals)
{
  if(vals.size() == size_t(1))
    return vals.front();
  // Pop a value v, multiply the rest, and then multiply v with that
  Value v = vals.front();
  Value prod = buildIndexProduct(loc, rewriter, vals.drop_front(1));
  return rewriter.create<arith::MulIOp>(loc, v, prod);
}

// Return true iff op has side effects that should happen exactly once.
// This means that if it appears in a parallel context, it must be wrapped in a kokkos.single.
//
// (!) Note: here, rely on a key difference between SCF and general Kokkos code.
// SCF's nested parallelism acts like a fork-join model, so values scoped outside
// the innermost parallel level cannot differ across threads.
//
// But in Kokkos, values scoped outside the innermost loop are replicated across threads/vector lanes, and
// those replications can have different values (CUDA model). The Kokkos dialect is capable of representing this case
// (e.g. team rank is a block argument and other values can be computed based on that), but Kokkos dialect
// code lowered from SCF (everything handled by this pass) can never do that.
bool opNeedsSingle(Operation* op)
{
  bool hasSideEffects = false;
  op->walk([&](memref::StoreOp) {
      hasSideEffects = true;
  });
  op->walk([&](memref::AtomicRMWOp) {
      hasSideEffects = true;
  });
  return hasSideEffects;
}

scf::ForOp scfParallelToSequential(RewriterBase& rewriter, scf::ParallelOp op)
{
  using ValueVector = SmallVector<Value>;
  rewriter.setInsertionPoint(op);
  mlir::scf::LoopNest newLoopNest = scf::buildLoopNest(
    rewriter, op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(), op.getInitVals(),
    [&](OpBuilder& builder, Location loc, ValueRange inductionVars, ValueRange args) -> ValueVector
    {
      ValueVector results;
      IRMapping irMap;
      for (auto p : llvm::zip(op.getInductionVars(), inductionVars)) {
        irMap.map(std::get<0>(p), std::get<1>(p));
      }
      for(Operation& oldOp : op.getBody()->getOperations()) {
        if(auto reduce = dyn_cast<scf::ReduceOp>(oldOp)) {
          //TODO: when LLVM is updated, this should be modified to support arbitrary number of reductions
          //The basic structure is the same, but now the reduce can have N operands and N regions, each doing
          //one join
          Value valToJoin = irMap.lookupOrDefault(reduce.getOperand());
          Value partialReduction = args[0];
          for(Region& reduceRegion : reduce->getRegions()) {
            Block& reduceBlock = reduceRegion.front();
            irMap.map(reduceBlock.getArguments()[0], partialReduction);
            irMap.map(reduceBlock.getArguments()[1], valToJoin);
            for(Operation& oldReduceOp : reduceBlock.getOperations()) {
              if(auto reduceReturn = dyn_cast<scf::ReduceReturnOp>(oldReduceOp)) {
                // reduceReturn's operand is the updated partial reduction.
                // Yield this back to the loop (final result or passed to next iteration)
                results.push_back(reduceReturn.getOperand());
              }
              else {
                // All other ops: just inline into new loop body
                auto newOp = builder.clone(oldReduceOp, irMap);
                for (auto p : llvm::zip(oldReduceOp.getResults(), newOp->getResults())) {
                  irMap.map(std::get<0>(p), std::get<1>(p));
                }
              }
            }
          }
        }
        else {
          // For all ops besides reduce, just inline into the new loop
          auto newOp = builder.clone(oldOp, irMap);
          for (auto p : llvm::zip(oldOp.getResults(), newOp->getResults())) {
            irMap.map(std::get<0>(p), std::get<1>(p));
          }
        }
      }
      return results;
    });
  scf::ForOp newOp = newLoopNest.loops.front();
  rewriter.replaceOp(op, newOp);
  return newOp;
}

// Rewrite the given scf.parallel as a kokkos.parallel, with the given execution space and nesting level
// (not for TeamPolicy loops)
// Return the new op, or NULL if failed.
kokkos::RangeParallelOp scfParallelToKokkosRange(RewriterBase& rewriter, scf::ParallelOp op, kokkos::ExecutionSpace exec, kokkos::ParallelLevel level)
{
  rewriter.setInsertionPoint(op);
  // Create the kokkos.parallel but don't populate the body yet
  auto kokkosRange = rewriter.create<kokkos::RangeParallelOp>(
    op.getLoc(), exec, level, op.getUpperBound(), op.getInitVals(), nullptr);
  // Now inline the old loop's operations into the new loop (replacing all usages of the induction variables)
  // Need to omit scf.yield, so can't just use rewriter.inlineBlockBefore
  rewriter.setInsertionPointToStart(&kokkosRange.getLoopBody().front());
  IRMapping irMap;
  for(Operation& oldOp : op.getBody()->getOperations()) {
    if(isa<scf::YieldOp>(oldOp))
      continue;
    // If we are inside a TeamThread loop and oldOp has side effects,
    // then we have to wrap the new op in a PerThread kokkos.single
    if(level == kokkos::ParallelLevel::TeamThread && opNeedsSingle(&oldOp)) {
      // The single has the same set of result types as the original op
      auto single = rewriter.create<kokkos::SingleOp>(oldOp.getLoc(), oldOp.getResultTypes(), kokkos::SingleLevel::PerThread);
      auto singleBody = rewriter.createBlock(&single.getRegion());
      auto ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(singleBody);
      auto singleWrappedOp = rewriter.clone(oldOp, irMap);
      rewriter.create<kokkos::YieldOp>(op->getLoc(), singleWrappedOp->getResults());
      for (auto p : llvm::zip(oldOp.getResults(), single->getResults())) {
        irMap.map(std::get<0>(p), std::get<1>(p));
      }
      rewriter.restoreInsertionPoint(ip);
    }
    else {
      auto newOp = rewriter.clone(oldOp, irMap);
      for (auto p : llvm::zip(oldOp.getResults(), newOp->getResults())) {
        irMap.map(std::get<0>(p), std::get<1>(p));
      }
    }
  }
  // The new loop is fully constructed.
  // Erase the old loop and replace uses of its results with those of the new loop.
  rewriter.replaceOp(op, kokkosRange);
  return kokkosRange;
}

// Convert an scf.parallel to a TeamPolicy parallel_for.
// The outer loop always uses a 1D iteration space where 1 team == 1 iteration.
// We can recover the original ND induction variables from the league rank.
kokkos::TeamParallelOp scfParallelToKokkosTeam(RewriterBase& rewriter, scf::ParallelOp op, Value teamSizeHint, Value vectorLengthHint)
{
  rewriter.setInsertionPoint(op);
  // First, compute the league size as the product of iteration bounds of op
  // (this is easy because the bounds are in [0:N:1] form)
  auto origLoopBounds = op.getUpperBound();
  Value leagueSize = buildIndexProduct(op.getLoc(), rewriter, origLoopBounds);
  // Create the kokkos.parallel but don't populate the body yet
  auto kokkosTeam = rewriter.create<kokkos::TeamParallelOp>(
    op.getLoc(), leagueSize, teamSizeHint, vectorLengthHint, op.getInitVals(), nullptr);
  // Now inline the old loop's operations into the new loop.
  rewriter.setInsertionPointToStart(&kokkosTeam.getLoopBody().front());
  IRMapping irMap;
  auto origInductionVars = op.getInductionVars();
  int n = origInductionVars.size();
  Value leagueRank = kokkosTeam.getLeagueRank();
  if(n == 1)
  {
    irMap.map(origInductionVars[0], leagueRank);
  }
  else
  {
    // If the original loop was not 1D, use arith.remui and arith.divui to get
    // the original (n-dimensional) induction variables in terms of the single league rank.
    // Treat the last dimension the fastest varying.
    Value leagueRankRemaining = leagueSize;
    for(int i = n - 1; i >= 0; i--)
    {
      Value thisLoopSize = origLoopBounds[i];
      // First, use remainder (remui) to get the current induction var
      Value newInductionVar = rewriter.create<arith::RemUIOp>(op.getLoc(), leagueRankRemaining, thisLoopSize).getResult();
      // Then (if there are more loops after this), divide the remaining league rank by the loop size, rounding down
      if(i != 0) {
        leagueRankRemaining = rewriter.create<arith::DivUIOp>(op.getLoc(), leagueRankRemaining, thisLoopSize).getResult();
      }
      // Map old to new, for when we generate the new body
      irMap.map(origInductionVars[i], newInductionVar);
    }
  }
  for(Operation& oldOp : op.getBody()->getOperations()) {
    if(isa<scf::YieldOp>(oldOp))
      continue;
    // If we are inside a TeamThread loop and oldOp has side effects,
    // then we have to wrap the new op in a PerThread kokkos.single
    if(opNeedsSingle(&oldOp)) {
      // The single has the same set of result types as the original op
      auto single = rewriter.create<kokkos::SingleOp>(oldOp.getLoc(), oldOp.getResultTypes(), kokkos::SingleLevel::PerTeam);
      auto singleBody = rewriter.createBlock(&single.getRegion());
      auto ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(singleBody);
      auto singleWrappedOp = rewriter.clone(oldOp, irMap);
      rewriter.create<kokkos::YieldOp>(op->getLoc(), singleWrappedOp->getResults());
      for (auto p : llvm::zip(oldOp.getResults(), single->getResults())) {
        irMap.map(std::get<0>(p), std::get<1>(p));
      }
      rewriter.restoreInsertionPoint(ip);
    }
    else {
      auto newOp = rewriter.clone(oldOp, irMap);
      for (auto p : llvm::zip(oldOp.getResults(), newOp->getResults())) {
        irMap.map(std::get<0>(p), std::get<1>(p));
      }
    }
  }
  // The new loop is fully constructed. Erase the old loop and replace uses of its results with those of the new loop.
  rewriter.replaceOp(op, kokkosTeam);
  return kokkosTeam;
}

// !! TODO: team synchronization requires dataflow analysis with op's body
// to understand what specific memrefs might be in use at each point
// (reading or writing) by other threads executing a team-level loop or single.
//
// So for now, conservatively insert a barrier after each team-wide parallel op and
// single that does not already synchronize.
// This is obviously not ideal for performance of complex kernels, but most of our
// common examples (spmv-like patterns, gemv/axpy fusion) are not affected.
LogicalResult insertTeamSynchronization(RewriterBase& rewriter, kokkos::TeamParallelOp op)
{
  op->walk([&](kokkos::RangeParallelOp nestedOp) {
    auto level = nestedOp.getParallelLevel();
    bool needsPostBarrier = 
    (nestedOp.getNumResults() == 0) &&
      (level == kokkos::ParallelLevel::TeamThread ||
      level == kokkos::ParallelLevel::TeamVector);
    if(needsPostBarrier) {
      rewriter.setInsertionPointAfter(nestedOp);
      rewriter.create<kokkos::TeamBarrierOp>(nestedOp.getLoc());
    }
  });
  op->walk([&](kokkos::SingleOp single) {
    bool needsPostBarrier = 
      single.getNumResults() == 0 && single.getLevel() == kokkos::SingleLevel::PerTeam;
    if(needsPostBarrier) {
      rewriter.setInsertionPointAfter(single);
      rewriter.create<kokkos::TeamBarrierOp>(single.getLoc());
    }
  });
  return success();
}

// Convert an scf.parallel to a kokkos.thread_parallel.
// This represents a common pattern where the top two levels of a TeamPolicy
// (TeamPolicy and TeamThread) are used together to iterate over one flat range.
// This leaves the last level (ThreadVector) available to iterate over additional nested loops.
Operation* scfParallelToKokkosThread(RewriterBase& rewriter, scf::ParallelOp op, Value vectorLengthHint)
{
  rewriter.setInsertionPoint(op);
  // First, compute the league size as the product of iteration bounds of op
  // (this is easy because the bounds are in [0:N:1] form)
  auto origLoopBounds = op.getUpperBound();
  Value numIters = buildIndexProduct(op.getLoc(), rewriter, origLoopBounds);
  // Create the kokkos.parallel but don't populate the body yet
  auto kokkosThread = rewriter.create<kokkos::ThreadParallelOp>(
    op.getLoc(), numIters, vectorLengthHint, op.getInitVals(), nullptr);
  // Now inline the old loop's operations into the new loop.
  rewriter.setInsertionPointToStart(&kokkosThread.getLoopBody().front());
  IRMapping irMap;
  auto origInductionVars = op.getInductionVars();
  int n = origInductionVars.size();
  // Use arith.remui and arith.divui to get the original (n-dimensional) induction variables in terms of the
  // single induction variable passed to the new loop body.
  // Treat the last dimension the fastest varying.
  Value inductionVar = kokkosThread.getInductionVar();
  if(n == 1)
  {
    irMap.map(origInductionVars[0], inductionVar);
  }
  else
  {
    Value iterCountRemaining = numIters;
    for(int i = n - 1; i >= 0; i--)
    {
      Value thisLoopSize = origLoopBounds[i];
      // First, use remainder (remui) to get the current induction var
      Value newInductionVar = rewriter.create<arith::RemUIOp>(op.getLoc(), iterCountRemaining, thisLoopSize).getResult();
      // Then (if there are more loops after this), divide the remaining league rank by the loop size, rounding down
      if(i != 0) {
        iterCountRemaining = rewriter.create<arith::DivUIOp>(op.getLoc(), iterCountRemaining, thisLoopSize).getResult();
      }
      // Map old to new, for when we generate the new body
      irMap.map(origInductionVars[i], newInductionVar);
    }
  }
  for(Operation& oldOp : op.getBody()->getOperations()) {
    if(isa<scf::YieldOp>(oldOp))
      continue;
    // If we are inside a TeamThread loop and oldOp has side effects,
    // then we have to wrap the new op in a PerThread kokkos.single
    if(opNeedsSingle(&oldOp)) {
      // The single has the same set of result types as the original op
      auto single = rewriter.create<kokkos::SingleOp>(oldOp.getLoc(), oldOp.getResultTypes(), kokkos::SingleLevel::PerTeam);
      auto singleBody = rewriter.createBlock(&single.getRegion());
      auto ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(singleBody);
      auto singleWrappedOp = rewriter.clone(oldOp, irMap);
      rewriter.create<kokkos::YieldOp>(op->getLoc(), singleWrappedOp->getResults());
      for (auto p : llvm::zip(oldOp.getResults(), single->getResults())) {
        irMap.map(std::get<0>(p), std::get<1>(p));
      }
      rewriter.restoreInsertionPoint(ip);
    }
    else {
      auto newOp = rewriter.clone(oldOp, irMap);
      for (auto p : llvm::zip(oldOp.getResults(), newOp->getResults())) {
        irMap.map(std::get<0>(p), std::get<1>(p));
      }
    }
  }
  // The new loop is fully constructed.
  // As a last step, erase the old loop and replace uses of its results with those of the new loop.
  rewriter.replaceOp(op, kokkosThread);
  return kokkosThread;
}

// Recursively map the scf.parallel children of op.
// parLevelsRemaining:
// If 2, then op will be inside a TeamParallel.
// If 1, then op will be inside a TeamThread RangeParallel.
// It can never be 0 because the innermost loops use ThreadVector.
void mapNestedLoopsImpl(RewriterBase& rewriter, Operation* op, int parLevelsRemaining)
{
  // Make a list of the directly nested scf.parallel ops inside op, and their respective nesting depths
  int opDepth = getOpParallelDepth(op);
  op->walk([&](scf::ParallelOp child) {
      int childDepth = getOpParallelDepth(child);
      if(childDepth == opDepth + 1) {
        int childNestingLevel = getParallelNumLevels(child);
        // Four cases:
        // parLevelsRemaining == 2 and childNestingLevel == 1: TeamVector
        // parLevelsRemaining == 2 and childNestingLevel > 1: TeamThread
        // parLevelsRemaining == 1 and childNestingLevel > 1: scf.for
        // parLevelsRemaining == 1 and childNestingLevel == 1: ThreadVector
        //
        // The children of child must be mapped first, but use the cases above
        // to determine if child will use a parallelism level
        bool childWillBeSequential = parLevelsRemaining == 1 && childNestingLevel > 1;
        mapNestedLoopsImpl(rewriter, child, parLevelsRemaining - (childWillBeSequential ? 0 : 1));
        // Now map child itself
        if(childWillBeSequential) {
          scfParallelToSequential(rewriter, child);
        }
        else {
          kokkos::ParallelLevel level;
          if(parLevelsRemaining == 2) {
            level = childNestingLevel == 1 ? kokkos::ParallelLevel::TeamVector : kokkos::ParallelLevel::TeamThread;
          }
          else {
            level = kokkos::ParallelLevel::ThreadVector;
          }
          scfParallelToKokkosRange(rewriter, child, kokkos::ExecutionSpace::Device, level);
        }
      }
  });
}

// Map all scf.parallel loops nested within a kokkos.team_parallel.
void mapTeamNestedLoops(RewriterBase& rewriter, kokkos::TeamParallelOp op)
{
  mapNestedLoopsImpl(rewriter, op, 2);
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
    // TODO: is there a more efficient way to check for multiple types of Op in a walk?
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
    // Cases for exec == Device:
    //
    // - Depth 1:   RangePolicy/MDRangePolicy
    // - Depth 2:   TeamPolicy with one thread (simd) per inner work-item (best for spmv-like patterns)
    //              TODO: Write a heuristic to choose TeamPolicy/TeamVector instead, for when the inner loop
    //              requires more parallelism?
    // - Depth >=3: Use TeamPolicy/TeamThread for outermost two loops, and ThreadVector for innermost loop.
    //              Better coalescing that way, if data layout is correct for the loop structure.
    //              Serialize all other loops by replacing them with scf.for.
    //
    // For exec == Host, just parallelize the outermost loop with RangePolicy and serialize any inner loops.
    if(exec == kokkos::ExecutionSpace::Host || nestingLevel == 1)
    {
      Operation* newOp = scfParallelToKokkosRange(rewriter, op, exec, kokkos::ParallelLevel::RangePolicy);
      if(!newOp) return op.emitError("Failed to convert scf.parallel to kokkos.range_parallel (RangePolicy)");
      // NOTE: when walking a tree and replacing ops as they're found, the walk must be PostOrder
      newOp->walk<WalkOrder::PostOrder>([&](scf::ParallelOp innerParOp) {
          scfParallelToSequential(rewriter, innerParOp);
          return WalkResult::skip();
      });
    }
    else if(nestingLevel == 2)
    {
      // Note: zero for vector length hint mean that no hint is given, and Kokkos::AUTO should be used instead
      Value zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
      Operation* newOp = scfParallelToKokkosThread(rewriter, op, zero);
      if(!newOp) return op.emitError("Failed to convert scf.parallel to kokkos.thread_parallel");
      // since the nesting level is 2, we know all inner parallel loops should be at ThreadVector level.
      // There won't be any more scf.parallels nested within those.
      newOp->walk<mlir::WalkOrder::PostOrder>([&](scf::ParallelOp innerParOp) {
          scfParallelToKokkosRange(rewriter, innerParOp, exec, kokkos::ParallelLevel::ThreadVector);
          return WalkResult::skip();
      });
    }
    else if(nestingLevel >= 3)
    {
      // Generate a team policy for the top level. This leaves 2 more levels of parallelism to use.
      // See mapTeamNestedLoops() above for how these are used.
      //
      // Note: zero for team size hint and vector length hint mean that no hint is given,
      // and Kokkos::AUTO should be used instead
      Value zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
      kokkos::TeamParallelOp newOp = scfParallelToKokkosTeam(rewriter, op, zero, zero);
      if(!newOp) return op.emitError("Failed to convert scf.parallel to kokkos.team_parallel");
      mapTeamNestedLoops(rewriter, newOp);
      // The loop structure is now finalized; insert team barriers where needed
      if(failed(insertTeamSynchronization(rewriter, newOp)))
        return newOp->emitError("Failed to insert team synchronization into body of kokkos.team_parallel");
    }
    return success();
  }
};

} // namespace

void mlir::populateKokkosLoopMappingPatterns(RewritePatternSet &patterns)
{
  patterns.add<KokkosLoopRewriter>(patterns.getContext());
}

