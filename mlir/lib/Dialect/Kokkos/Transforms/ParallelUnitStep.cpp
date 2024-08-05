//===- ParallelUnitStep.cpp - define parallel-unit-step pass pattern --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Kokkos/IR/KokkosDialect.h"
#include "mlir/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace {

// Is v a compile-time constant integer with value 0?
bool valueIsIntegerConstantZero(Value v)
{
  // If we don't know what op generated v, can't assume anything about its value
  if (!v.getDefiningOp())
    return false;
  if (auto constantOp = dyn_cast<arith::ConstantOp>(v.getDefiningOp())) {
    auto valAttr = constantOp.getValue();
    if (auto iAttr = valAttr.dyn_cast<IntegerAttr>()) {
      return iAttr.getValue().isZero();
    }
    return false;
  }
  return false;
}

// Is v a compile-time constant integer with value 1?
bool valueIsIntegerConstantOne(Value v)
{
  // If we don't know what op generated v, can't assume anything about its value
  if (!v.getDefiningOp())
    return false;
  if (auto constantOp = dyn_cast<arith::ConstantOp>(v.getDefiningOp())) {
    auto valAttr = constantOp.getValue();
    if (auto iAttr = valAttr.dyn_cast<IntegerAttr>()) {
      return iAttr.getValue().isOne();
    }
    return false;
  }
  return false;
}

struct ParallelUnitStepRewriter : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  ParallelUnitStepRewriter(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {
    // n is the dimensionality of loop (number of lower bounds/upper bounds/steps)
    const int n = op.getNumLoops();
    // If all lower bounds are 0 and all steps are 1, do nothing
    bool allUnitAlready = true;
    for(int i = 0; i < n; i++) {
      if(!valueIsIntegerConstantZero(op.getLowerBound()[i]) ||
         !valueIsIntegerConstantOne(op.getStep()[i])) {
        allUnitAlready = false;
        break;
      }
    }
    if(allUnitAlready) {
      // op did not match pattern; nothing to do
      return failure();
    }
    // Insert zero and one index constants before the ParallelOp
    rewriter.setInsertionPoint(op);
    Value zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    // Given lower, step, upper:
    //  - Replace lower with 0
    //  - Replace upper with (upper - lower + step - 1) / step
    //  - Replace step with 1
    //  - Replace all uses of the old induction variable with "lower + i * step" where i is the new induction variable
    SmallVector<Value> newLowers;
    SmallVector<Value> newSteps;
    SmallVector<Value> newUppers;
    for(int i = 0; i < n; i++) {
      newLowers.push_back(zero);
      newSteps.push_back(one);
      // If this dimension already has lower 0 and step 1, just keep the old upper
      Value step = op.getStep()[i];
      Value lower = op.getLowerBound()[i];
      if(!valueIsIntegerConstantZero(lower) ||
         !valueIsIntegerConstantOne(step)) {
        // Note: folding will automatically simplify this arithmetic where possible
        // upper - lower
        Value newUpper = rewriter.create<arith::SubIOp>(op.getLoc(), op.getUpperBound()[i], lower).getResult();
        // upper - lower + step
        newUpper = rewriter.create<arith::AddIOp>(op.getLoc(), newUpper, step).getResult();
        // upper - lower + step - 1
        newUpper = rewriter.create<arith::SubIOp>(op.getLoc(), newUpper, one).getResult();
        // (upper - lower + step - 1) / step
        newUpper = rewriter.create<arith::DivUIOp>(op.getLoc(), newUpper, step).getResult();
        newUppers.push_back(newUpper);
      }
      else {
        // Leave upper the same
        newUppers.push_back(op.getUpperBound()[i]);
      }
    }
    // Now go into the loop body and compute a mapping from new to old induction variable,
    // and replace all usages of the induction variable with this.
    auto& body = op.getLoopBody();
    rewriter.setInsertionPointToStart(&body.front());
    for(int i = 0; i < n; i++) {
      Value oldLower = op.getLowerBound()[i];
      Value oldStep = op.getStep()[i];
      Value induction = op.getInductionVars()[i];
      // Skip this dimension is nothing is changing
      if(valueIsIntegerConstantZero(oldLower) &&
         valueIsIntegerConstantOne(oldStep)) {
        continue;
      }
      // Compute lower + i * step (old lower and step, and i is the induction var)
      // Then replace all other uses of the old induction variable (except this expression!)
      auto replacementException = rewriter.create<arith::MulIOp>(op.getLoc(), induction, oldStep);
      Value inductionReplacement = rewriter.create<arith::AddIOp>(op.getLoc(), oldLower, replacementException.getResult()).getResult();
      rewriter.replaceAllUsesExcept(induction, inductionReplacement, replacementException);
    }
    //TODO: they renamed this to startOpModification in upstream
    rewriter.startRootUpdate(op);
    //rewriter.startOpModification(op);
    op.getLowerBoundMutable().assign(newLowers);
    op.getUpperBoundMutable().assign(newUppers);
    op.getStepMutable().assign(newSteps);
    //TODO: they renamed this to finalizeOpModification upstream
    rewriter.finalizeRootUpdate(op);
    //rewriter.finalizeOpModification(op);
    return success();
  }
};

} // namespace

void mlir::populateParallelUnitStepPatterns(RewritePatternSet &patterns)
{
  patterns.add<ParallelUnitStepRewriter>(patterns.getContext());
}
