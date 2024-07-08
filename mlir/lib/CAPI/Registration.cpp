#include "mlir/InitAllKokkosPasses.h"
#include "mlir-c/kmDialects.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

void kokkosMlirRegisterAllPasses() {
  std::cout << " registerAllKokkosPasses2 called "<< std::endl;

  mlir::registerAllKokkosPasses();
}

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Kokkos, kokkos, mlir::sparse_tensor::SparseTensorDialect)
