#ifndef LINALGTOPARTTENSOR_H_
#define LINALGTOPARTTENSOR_H_

#include <mlir/Dialect/SparseTensor/IR/SparseTensor.h>

namespace lapis {
namespace part_tensor {

using mlir::Operation;
using mlir::RankedTensorType;
using mlir::Type;
using mlir::TypeRange;
using mlir::sparse_tensor::SparseTensorEncodingAttr;

Type getPartTensorType(mlir::MLIRContext *context,
                       const RankedTensorType &spTensorTy) {
  using mlir::part_tensor::PartTensorEncodingAttr;
  SparseTensorEncodingAttr encoding;
  auto ptEncoding = PartTensorEncodingAttr::get(
      context, 1, mlir::sparse_tensor::getSparseTensorEncoding(spTensorTy));
  return mlir::RankedTensorType::get(spTensorTy.getShape(),
                                     spTensorTy.getElementType(), ptEncoding);
}

// See the any version already defined in sparse_tensor
inline bool hasAllSparseType(TypeRange types) {
  return llvm::any_of(types, [](Type type) {
    return mlir::sparse_tensor::getSparseTensorEncoding(type) != nullptr;
  });
}

/// Returns true iff MLIR operation has all sparse result.
inline bool hasAllSparseResult(Operation *op) {
  return hasAllSparseType(op->getResults().getTypes());
}

/// Returns true iff MLIR operation has all sparse args.
inline bool hasAllSparseOperands(Operation *op) {
  return hasAllSparseType(op->getOperands().getTypes());
}

}
}



#endif // LINALGTOPARTTENSOR_H_
