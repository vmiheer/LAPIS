#include <optional>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeUtilities.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "lapis/Dialect/PartTensor/IR/PartTensor.h"
#include "lapis/Dialect/PartTensor/IR/PartTensorType.h"
#include "lapis/Dialect/PartTensor/Transforms/LinalgToPartTensor.h"
#include "lapis/Dialect/PartTensor/Transforms/Passes.h"

#include "CodegenUtils.h"
#include "fmt/core.h"

using namespace mlir;
using namespace mlir::part_tensor;
using mlir::ModuleOp;
using mlir::linalg::generalizeNamedOp;
using mlir::linalg::GenericOp;
using mlir::linalg::LinalgOp;
using std::optional;

struct FuncOpConversion final : OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    using llvm::dbgs;
    auto &region = funcOp.getRegion();
    if (!region.hasOneBlock())
      return failure();
    // if there is any sub-region which is not a linalg op, disable conversion
    size_t numLinalgOps = 0;
    optional<LinalgOp> linalgOp;
    auto walkResult = region.walk([&](Operation *op) {
      if (!isa<LinalgOp>(op) && op->getRegions().size() > 0) {
        dbgs() << fmt::format("Found region so disabling conversion");
        dbgs() << *op;
        return WalkResult::interrupt();
      } else if (isa<LinalgOp>(op)) {
        linalgOp = cast<LinalgOp>(op);
        numLinalgOps++;
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      fmt::println("walk interrupted");
      return failure();
    }
    if (numLinalgOps != 1) {
      fmt::println("Found {} linalg ops so disabling conversion", numLinalgOps);
      return (numLinalgOps == 0) ? success() : failure();
    }

    const bool AllSparse = lapis::part_tensor::hasAllSparseResult(*linalgOp) &&
                           lapis::part_tensor::hasAllSparseOperands(*linalgOp);
    if (!AllSparse) {
      fmt::println("Only supported when all arguments and results are sparse!");
      return failure();
    }

    // Create new function
    // auto module = SymbolTable::getNearestSymbolTable(*linalgOp);
    auto module =
        rewriter.getInsertionBlock()->getParent()->getParentOfType<ModuleOp>();
    MLIRContext *context = module.getContext();
    auto result = SymbolRefAttr::get(context, "dist_op");
    auto opFunc = module.lookupSymbol<func::FuncOp>(result.getAttr());

    if (!opFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());
      // auto opFunctionTy = FunctionType::get(
      //     rewriter.getContext(),
      //     (*linalgOp).getOperation()->getOperands().getTypes(),
      //     (*linalgOp).getOperation()->getResults().getTypes());
      auto opFunctionTy = FunctionType::get(
          rewriter.getContext(),
          (*linalgOp).getOperation()->getOperands().getTypes(), {});
      opFunc = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(),
                                             "dist_op", opFunctionTy);
      Block *entryBB = opFunc.addEntryBlock();
      // opFunc.dump();
    }
    // rewriter.replaceOp(funcOp, opFunc);
    rewriter.eraseOp(funcOp);
    // auto newFuncOp = rewriter.create<func::FuncOp>(
    //     funcOp.getLoc(), funcOp.getName(), funcOp.getType());
    return success();
  }
};

void mlir::populateLinalgToPartTensorPatterns(TypeConverter &typeConverter,
                                              RewritePatternSet &patterns) {
  patterns.add<FuncOpConversion>(typeConverter, patterns.getContext());
}
