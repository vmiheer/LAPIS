//===- lapis-opt.cpp - LAPIS pass Driver -------------------------===//
//
//#include "mlir/InitAllDialects.h"
//#include "mlir/InitAllExtensions.h"
//#include "mlir/InitAllPasses.h"
#include "mlir/Dialect/Kokkos/IR/KokkosDialect.h"
#include "mlir/Dialect/Kokkos/Transforms/Passes.h"
#ifdef ENABLE_PART_TENSOR
#include "mlir/Dialect/PartTensor/IR/PartTensor.h"
#include "mlir/Dialect/PartTensor/Transforms/Passes.h"
#endif
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<
    mlir::kokkos::KokkosDialect,
    mlir::part_tensor::PartTensorDialect
  >();

  mlir::registerKokkosPasses();
#ifdef ENABLE_PART_TENSOR
  mlir::registerPartTensorPasses();
#endif

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "LAPIS/MLIR pass driver\n", registry));
}

