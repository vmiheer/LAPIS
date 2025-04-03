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
using mlir::sparse_tensor::createFuncCall;
using mlir::sparse_tensor::genAlloca;
using mlir::sparse_tensor::EmitCInterface;

namespace {
static Value genGetRankCall(OpBuilder &builder, Location loc) {
  StringRef name = "mpi_getRank";
  Type iTp = builder.getIndexType();
  return createFuncCall(builder, loc, name, iTp, {}, EmitCInterface::Off)
      .getResult(0);
}

#define GEN_PASS_DEF_LINALGTOPARTTENSOR
#include "lapis/Dialect/PartTensor/Transforms/Passes.h.inc"
struct LinalgToPartTensorPass
    : public impl::LinalgToPartTensorBase<LinalgToPartTensorPass> {
  LinalgToPartTensorPass() = default;
  LinalgToPartTensorPass(const LinalgToPartTensorPass &pass) = default;

  optional<LinalgOp> getCandidateLinalgOp(func::FuncOp funcOp) {
    using llvm::dbgs;
    auto &region = funcOp.getRegion();
    if (!region.hasOneBlock())
      return std::nullopt;
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
      return std::nullopt;
    }
    if (numLinalgOps != 1) {
      fmt::println("Found {} linalg ops so disabling conversion", numLinalgOps);
      return std::nullopt;
    }

    const bool AllSparse = lapis::part_tensor::hasAllSparseResult(*linalgOp) &&
                           lapis::part_tensor::hasAllSparseOperands(*linalgOp);
    if (!AllSparse) {
      fmt::println("Only supported when all arguments and results are sparse!");
      return std::nullopt;
    }
    return linalgOp;
  }
  LogicalResult processFunction(func::FuncOp funcOp,
                                ImplicitLocOpBuilder &builder) {
    auto linalgOp = getCandidateLinalgOp(funcOp);
    if (!linalgOp)
      return failure();
    auto partTensorTypes = llvm::to_vector(
        llvm::map_range(linalgOp->getOperation()->getOperands().getTypes(),
                        [&](Type t) -> Type {
                          return lapis::part_tensor::getPartTensorType(
                              builder.getContext(), cast<RankedTensorType>(t));
                        }));
    // auto getRankTy =
    //     FunctionType::get(builder.getContext(), {}, {builder.getIndexType()});
    // auto getRankDecl =
    //     builder.create<func::FuncOp>(funcOp.getLoc(), "mpi_getRank", getRankTy);
    // getRankDecl.setPrivate();
    auto indexTp = builder.getIndexType();
    auto opFunctionTy =
        FunctionType::get(builder.getContext(), partTensorTypes, {});
    auto opFunc =
        builder.create<func::FuncOp>(funcOp.getLoc(), "dist_op", opFunctionTy);
    opFunc.setPrivate();
    Block *entryBB = opFunc.addEntryBlock();
    builder.setInsertionPointToEnd(entryBB);
    auto rank = genGetRankCall(builder, funcOp.getLoc());
    auto memref1dDynTp = MemRefType::get({ShapedType::kDynamic}, indexTp);
    auto arg0 = entryBB->getArgument(0);
    auto primaryPartPlan = builder.create<part_tensor::GetPartitionsOp>(
        funcOp.getLoc(), memref1dDynTp, arg0);
    auto const arg0Rank = arg0.getType().cast<RankedTensorType>().getRank();
    auto arg0partspec = genAlloca(builder, funcOp.getLoc(), arg0Rank * 2,
                                 indexTp, false);
    // Let's assume first parameter is going to be primary tensor
    builder.create<func::ReturnOp>(funcOp.getLoc());
    return success();
  }
  void runOnOperation() override {
    // auto *ctx = &getContext();
    ModuleOp module = getOperation();
    Location loc = module->getLoc();
    ImplicitLocOpBuilder builder =
        ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());
    module.walk([&](func::FuncOp funcOp) {
      LogicalResult res = processFunction(funcOp, builder);
      (void)res;
    });

    // RewritePatternSet patterns(ctx);
    // PartTensorTypeToPtrConverter converter;
    // ConversionTarget target(*ctx);
    // // Allow func.call
    // // target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    // //   return converter.isSignatureLegal(op.getFunctionType());
    // // });
    // // target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    // //   return converter.isSignatureLegal(op.getCalleeType());
    // // });
    // // target.addLegalDialect<
    // //     arith::ArithDialect, bufferization::BufferizationDialect,
    // //     LLVM::LLVMDialect, memref::MemRefDialect, scf::SCFDialect,
    // //     sparse_tensor::SparseTensorDialect>();
    // // target.addLegalOp<UnrealizedConversionCastOp>();
    // // Populate with rules and apply rewriting rules.
    // populateLinalgToPartTensorPatterns(converter, patterns);
    // if (failed(applyPartialConversion(getOperation(), target,
    //                                   std::move(patterns))))
    //   signalPassFailure();
  }
};
} // namespace
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
    // auto module =
    //     rewriter.getInsertionBlock()->getParent()->getParentOfType<ModuleOp>();
    // MLIRContext *context = module.getContext();
    // auto result = SymbolRefAttr::get(context, "dist_op");
    // auto opFunc = module.lookupSymbol<func::FuncOp>(result.getAttr());

    // if (!opFunc) {
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(funcOp);
      auto opFunctionTy = FunctionType::get(
          rewriter.getContext(),
          (*linalgOp).getOperation()->getOperands().getTypes(), {});
      auto opFunc = rewriter.create<func::FuncOp>(funcOp.getLoc(), "dist_op",
                                                  opFunctionTy);
      opFunc.setPrivate();
      Block *entryBB = opFunc.addEntryBlock();
      rewriter.setInsertionPointToEnd(entryBB);
      rewriter.create<func::ReturnOp>(funcOp.getLoc());
    }

    // opFunc.dump();
    // }
    rewriter.eraseOp(funcOp);
    // rewriter.replaceOp(funcOp, opFunc);
    // auto newFuncOp = rewriter.create<func::FuncOp>(
    //     funcOp.getLoc(), funcOp.getName(), funcOp.getType());
    return success();
  }
};

void mlir::populateLinalgToPartTensorPatterns(TypeConverter &typeConverter,
                                              RewritePatternSet &patterns) {
  patterns.add<FuncOpConversion>(typeConverter, patterns.getContext());
}

std::unique_ptr<Pass> mlir::createLinalgToPartTensorPass() {
  return std::make_unique<LinalgToPartTensorPass>();
}
