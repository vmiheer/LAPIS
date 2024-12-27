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
#include "lapis/Dialect/PartTensor/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::part_tensor;
using mlir::ModuleOp;
using mlir::linalg::generalizeNamedOp;
using mlir::linalg::GenericOp;
using mlir::linalg::LinalgOp;

struct FuncOpConversion final : OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(funcOp);
    return success();
  }
};

void mlir::populateLinalgToPartTensorPatterns(TypeConverter &typeConverter,
                                              RewritePatternSet &patterns) {
  patterns.add<FuncOpConversion>(typeConverter, patterns.getContext());
}
