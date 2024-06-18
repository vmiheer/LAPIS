#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;

PYBIND11_MODULE(_kokkosMlir, m) {
  kokkosMlirRegisterAllPasses();

  m.doc() = "kokkos-mlir";

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__kokkos__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
}
