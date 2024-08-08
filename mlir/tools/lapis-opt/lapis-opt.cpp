//===- lapis-opt.cpp - LAPIS pass Driver -------------------------===//
//
#include "mlir/Dialect/Kokkos/IR/KokkosDialect.h"
#include "mlir/Dialect/Kokkos/Pipelines/Passes.h"
#include "mlir/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Pipelines/Passes.h"
#ifdef ENABLE_PART_TENSOR
#include "mlir/Dialect/PartTensor/IR/PartTensor.h"
#include "mlir/Dialect/PartTensor/Transforms/Passes.h"
#include "mlir/Dialect/PartTensor/Pipelines/Passes.h"
#endif
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

void registerLAPISPipelines() {
}

int main(int argc, char **argv) {
  // lapis-opt is intended to drive only passes from the custom
  // LAPIS dialects (Kokkos and PartTensor), not builtin dialects.
  // Use mlir-opt for those passes.
  DialectRegistry registry;
  registry.insert<
    mlir::kokkos::KokkosDialect,
    mlir::part_tensor::PartTensorDialect
  >();

  // Register LAPIS pipelines and passes
  sparse_tensor::registerSparseTensorPipelines();

#ifdef ENABLE_PART_TENSOR
  part_tensor::registerPartTensorPipelines();
  mlir::registerPartTensorPasses();
#endif

  kokkos::registerKokkosPipelines();
  mlir::registerKokkosPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "LAPIS/MLIR pass driver\n", registry));
}

