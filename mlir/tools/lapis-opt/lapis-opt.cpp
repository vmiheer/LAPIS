//===- lapis-opt.cpp - LAPIS pass Driver -------------------------===//
//
#include "mlir/Dialect/Kokkos/IR/KokkosDialect.h"
#include "mlir/Dialect/Kokkos/Pipelines/Passes.h"
#include "mlir/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Pipelines/Passes.h"
#ifdef ENABLE_PART_TENSOR
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/PartTensor/IR/PartTensor.h"
#include "mlir/Dialect/PartTensor/Pipelines/Passes.h"
#include "mlir/Dialect/PartTensor/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#endif
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  // lapis-opt is intended to drive only passes from the custom
  // LAPIS dialects (Kokkos and PartTensor), not builtin dialects.
  // Use mlir-opt for those passes.
  DialectRegistry registry;
  registry.insert<
#ifdef ENABLE_PART_TENSOR
      mlir::arith::ArithDialect, mlir::bufferization::BufferizationDialect,
      mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
      mlir::memref::MemRefDialect, mlir::sparse_tensor::SparseTensorDialect,
      mlir::tensor::TensorDialect,
#endif
      mlir::kokkos::KokkosDialect, mlir::part_tensor::PartTensorDialect>();

  // Register LAPIS pipelines and passes
#ifdef ENABLE_PART_TENSOR
  part_tensor::registerPartTensorPipelines();
  mlir::registerPartTensorPasses();
#endif

  kokkos::registerKokkosPipelines();
  mlir::registerKokkosPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "LAPIS/MLIR pass driver\n", registry));
}
