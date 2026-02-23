#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::part_tensor;
using mlir::ModuleOp;
using mlir::linalg::generalizeNamedOp;
using mlir::linalg::GenericOp;
using mlir::linalg::LinalgOp;
using mlir::sparse_tensor::createFuncCall;
using mlir::sparse_tensor::EmitCInterface;
using mlir::sparse_tensor::genAlloca;
using std::optional;

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

  // Helper function to create a wrapper function for the linalg operation
  func::FuncOp createLinalgWrapperFunction(LinalgOp linalgOp,
                                           ImplicitLocOpBuilder &builder,
                                           ModuleOp module) {
    auto ctx = builder.getContext();
    auto sparseTensorTypes = linalgOp.getOperation()->getOperands().getTypes();
    auto linalgOpResTy = linalgOp.getOperation()->getResultTypes();

    // Create function type: (operands) -> (result)
    auto funcTy = FunctionType::get(ctx, sparseTensorTypes, linalgOpResTy);

    // Save current insertion point and create wrapper at module level
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());

    // Create the wrapper function
    auto linalgFunc = builder.create<func::FuncOp>(linalgOp.getLoc(),
                                                   "linalg_op_wrapper", funcTy);
    linalgFunc.setPrivate();

    // Add entry block
    Block *entryBB = linalgFunc.addEntryBlock();
    builder.setInsertionPointToEnd(entryBB);

    // Create the linalg::GenericOp with the function arguments as operands
    auto newLinalgOp = builder.create<linalg::GenericOp>(
        linalgOp.getLoc(), linalgOpResTy, entryBB->getArguments().drop_back(),
        entryBB->getArguments().back(), linalgOp.getIndexingMapsArray(),
        linalgOp.getIteratorTypesArray());

    // Clone the region from the original linalg op
    IRMapping mapping;
    linalgOp.getOperation()->getRegion(0).cloneInto(
        &newLinalgOp.getRegion(), newLinalgOp.getRegion().begin(), mapping);

    // Return the result
    builder.create<func::ReturnOp>(linalgOp.getLoc(), newLinalgOp.getResults());

    return linalgFunc;
  }

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
    auto sparseTensorTypes = linalgOp->getOperation()->getOperands().getTypes();
    auto partTensorTypes =
        llvm::to_vector(llvm::map_range(sparseTensorTypes, [&](Type t) -> Type {
          return lapis::part_tensor::getPartTensorType(
              builder.getContext(), cast<RankedTensorType>(t));
        }));
    // auto getRankTy =
    //     FunctionType::get(builder.getContext(), {},
    //     {builder.getIndexType()});
    // auto getRankDecl =
    //     builder.create<func::FuncOp>(funcOp.getLoc(), "mpi_getRank",
    //     getRankTy);
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
    auto const arg0Rank = cast<RankedTensorType>(arg0.getType()).getRank();
    auto partSpecs = llvm::to_vector(llvm::map_range(
        llvm::seq<size_t>(0, entryBB->getNumArguments()), [&](size_t i) {
          auto arg = entryBB->getArgument(i);
          auto argRank = llvm::cast<RankedTensorType>(arg.getType()).getRank();
          return genAlloca(builder, funcOp.getLoc(), argRank * 2, indexTp,
                           false);
        }));
    auto &arg0partspec = partSpecs[0];
    auto ptStartIdx = builder.create<arith::MulIOp>(
        funcOp.getLoc(), rank,
        builder.create<arith::ConstantIndexOp>(funcOp.getLoc(), arg0Rank * 2));
    llvm::for_each(llvm::seq<size_t>(0, arg0Rank * 2), [&](size_t i) {
      Value ptIdx = builder.create<arith::AddIOp>(
          funcOp.getLoc(), ptStartIdx,
          builder.create<arith::ConstantIndexOp>(funcOp.getLoc(), i));
      Value ptVal = builder.create<memref::LoadOp>(funcOp.getLoc(),
                                                   primaryPartPlan, ptIdx);
      builder.create<memref::StoreOp>(
          funcOp.getLoc(), ptVal, arg0partspec,
          Value(builder.create<arith::ConstantIndexOp>(funcOp.getLoc(), i)));
    });

    // Create a temporary linalg op to compute loop ranges
    auto linalgOpResTy = linalgOp->getOperation()->getResultTypes();
    auto tempLinalgOp = builder.create<linalg::GenericOp>(
        funcOp.getLoc(), linalgOpResTy, entryBB->getArguments().drop_back(),
        entryBB->getArguments().back(), linalgOp->getIndexingMapsArray(),
        linalgOp->getIteratorTypesArray());
    auto extents = llvm::cast<linalg::LinalgOp>(tempLinalgOp.getOperation())
                       .createLoopRanges(builder, funcOp.getLoc());

    // Compute workload bounds based on the access maps
    auto access0 = linalgOp->getIndexingMapsArray()[0];
    auto access0InvPermMap =
        mlir::inverseAndBroadcastProjectedPermutation(access0);
    auto const loopRank = access0.getNumInputs();
    SmallVector<Value> workloadLo(loopRank), workloadHi(loopRank);

    for (auto i : llvm::seq<size_t>(0, loopRank)) {
      auto expr = access0InvPermMap.getResult(i);
      auto constExpr = dyn_cast<AffineConstantExpr>(expr);
      if (constExpr && constExpr.getValue() == 0) {
        auto lo = builder.create<arith::ConstantIndexOp>(funcOp.getLoc(), 0);
        auto hi = extents[i].size;
        workloadLo[i] = lo;
        workloadHi[i] =
            getValueOrCreateConstantIndexOp(builder, funcOp.getLoc(), hi);
        continue;
      }

      auto dim = access0InvPermMap.getDimPosition(i);
      Value arg0Idx =
          builder.create<arith::ConstantIndexOp>(funcOp.getLoc(), dim);
      Value arg0HiIdx = builder.create<arith::ConstantIndexOp>(funcOp.getLoc(),
                                                               arg0Rank + dim);
      auto loExt =
          builder.create<memref::LoadOp>(indexTp, arg0partspec, arg0Idx);
      auto hiExt =
          builder.create<memref::LoadOp>(indexTp, arg0partspec, arg0HiIdx);
      workloadLo[i] = loExt;
      workloadHi[i] = hiExt;
    }

    // Populate partition specs for all other tensors based on access maps
    for (auto i : llvm::seq<size_t>(1, partSpecs.size())) {
      auto pspec = partSpecs[i];
      auto pspecTp = mlir::cast<MemRefType>(pspec.getType());
      auto pspecRank = pspecTp.getRank();
      auto tensorRank =
          llvm::cast<RankedTensorType>(entryBB->getArgument(i).getType())
              .getRank();
      auto accessMap = linalgOp->getIndexingMapsArray()[i];
      for (auto j : llvm::seq<size_t>(0, tensorRank)) {
        auto expr = accessMap.getResult(j);
        auto dim = llvm::cast<AffineDimExpr>(expr).getPosition();
        Value argIdx =
            builder.create<arith::ConstantIndexOp>(funcOp.getLoc(), j);
        Value argHiIdx = builder.create<arith::ConstantIndexOp>(funcOp.getLoc(),
                                                                j + tensorRank);
        builder.create<memref::StoreOp>(funcOp.getLoc(), workloadLo[dim], pspec,
                                        argIdx);
        builder.create<memref::StoreOp>(funcOp.getLoc(), workloadHi[dim], pspec,
                                        argHiIdx);
      }
    }

    // Clean up temporary linalg op
    tempLinalgOp.getOperation()->erase();

    // Extract slices from the partitioned tensors
    SmallVector<Value> slices(partSpecs.size());
    for (auto i : llvm::seq<size_t>(0, partSpecs.size())) {
      auto pspec = partSpecs[i];
      auto ptensor = entryBB->getArgument(i);
      slices[i] = builder.create<part_tensor::GetSliceOp>(
          funcOp.getLoc(), sparseTensorTypes[i], ptensor, pspec);
    }
    // Create the linalg wrapper function
    auto module = funcOp->getParentOfType<ModuleOp>();
    auto linalgWrapperFunc =
        createLinalgWrapperFunction(*linalgOp, builder, module);

    // Call the linalg wrapper function with the slices
    builder.setInsertionPointToEnd(entryBB);
    auto callOp = builder.create<func::CallOp>(
        funcOp.getLoc(), linalgWrapperFunc, llvm::ArrayRef(slices));

    // Store the result back into the partitioned tensor
    auto entryBBArgs = entryBB->getArguments();
    auto setSliceOp = builder.create<part_tensor::SetSliceOp>(
        funcOp.getLoc(), partTensorTypes.back(), entryBBArgs.back(),
        partSpecs.back(), callOp.getResult(0));

    // Return from the distributed function
    builder.create<func::ReturnOp>(funcOp.getLoc());

    // Let's assume first parameter is going to be primary tensor
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
