#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir-c/kmDialects.h"
#include "mlir-c/Registration.h"
#include "mlir-c/EmitKokkos.h"
//#include "mlir-c/Bindings/Python/Interop.h"
//#include "IRModule.h"

#include "mlir/InitAllKokkosPasses.h"

void lapisRegisterAllPasses() {
  printf("*************** registerAllKokkosPasses() !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  mlir::registerAllKokkosPasses();
}

namespace py = pybind11;

using namespace mlir::python;

PYBIND11_MODULE(_lapis, m) {
  lapisRegisterAllPasses();

  m.doc() = "LAPIS dialect registration for Python interface";

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        throw std::runtime_error("Oh shit");
        /*
        printf("Hello from register_dialect! load = %d\n", load ? 1:0);
        MlirDialectHandle handle = mlirGetDialectHandle__kokkos__();
        printf("Registering...\n");
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          printf("Loading...\n");
          mlirDialectHandleLoadDialect(handle, context);
        }
        */
      },
      py::arg("context"), py::arg("load") = true);

//  m.def("emit_kokkos",
//          [](PyModule &module, const char* cxxSourceFile, const char* pySourceFile) {
//            MlirLogicalResult status =
//                lapisEmitKokkos(module.get(), cxxSourceFile, pySourceFile);
//            if (mlirLogicalResultIsFailure(status))
//              throw MLIRError("Failure while raising MLIR to Kokkos C++ source code.");
//          },
//          py::arg("module"), py::arg("cxx_source_file"), py::arg("py_source_file"),
//          "Emit Kokkos C++ and Python wrappers for the given module, and throw a RuntimeError on failure.");
//
//  m.def("emit_kokkos_sparse",
//          [](PyModule &module, const char* cxxSourceFile, const char* pySourceFile, bool useHierarchical, bool isLastKernel) {
//            MlirLogicalResult status =
//                lapisEmitKokkosSparse(module.get(), cxxSourceFile, pySourceFile, useHierarchical, isLastKernel);
//            if (mlirLogicalResultIsFailure(status))
//              throw MLIRError("Failure while raising MLIR to Kokkos C++ source code.");
//          },
//          py::arg("module"), py::arg("cxx_source_file"), py::arg("py_source_file"), py::arg("use_hierarchical"), py::arg("is_final_kernel"),
//          "Emit Kokkos C++ and Python wrappers for the given sparse module, and throw a RuntimeError on failure.");
}

