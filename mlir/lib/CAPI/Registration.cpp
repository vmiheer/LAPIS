#include "mlir/InitAllKokkosPasses.h"
#include "mlir-c/kmDialects.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

void lapisRegisterAllPasses() {
  mlir::registerAllKokkosPasses();
}

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Kokkos, kokkos, mlir::sparse_tensor::SparseTensorDialect)
