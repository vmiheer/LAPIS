#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir-c/kmDialects.h"
#include "mlir-c/Registration.h"

#include "IRModule.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/kmPass.h"

#include "mlir/InitAllKokkosPasses.h"

void lapisRegisterAllPasses() {
  mlir::registerAllKokkosPasses();
}

namespace py = pybind11;

namespace lapis {
  namespace python {
    void populatePassManagerSubmodule(py::module &m);
  }
}

PYBIND11_MODULE(_lapis, m) {
  lapisRegisterAllPasses();

  m.doc() = "lapis";

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


  auto passModule =
      m.def_submodule("passmanager", "MLIR Pass Management Bindings");
  lapis::python::populatePassManagerSubmodule(passModule);
}


using namespace mlir;
using namespace mlir::python;

namespace {

/// Owning Wrapper around a PassManager.
class PyPassManager {
public:
  PyPassManager(MlirPassManager passManager) : passManager(passManager) {}
  PyPassManager(PyPassManager &&other) : passManager(other.passManager) {
    other.passManager.ptr = nullptr;
  }
  ~PyPassManager() {
    if (!mlirPassManagerIsNull(passManager))
      mlirPassManagerDestroy(passManager);
  }
  MlirPassManager get() { return passManager; }

  void release() { passManager.ptr = nullptr; }
  pybind11::object getCapsule() {
    return py::reinterpret_steal<py::object>(
        mlirPythonPassManagerToCapsule(get()));
  }

  static pybind11::object createFromCapsule(pybind11::object capsule) {
    MlirPassManager rawPm = mlirPythonCapsuleToPassManager(capsule.ptr());
    if (mlirPassManagerIsNull(rawPm))
      throw py::error_already_set();
    return py::cast(PyPassManager(rawPm), py::return_value_policy::move);
  }

private:
  MlirPassManager passManager;
};

} // namespace

/// Create the `mlir.passmanager` here.
void lapis::python::populatePassManagerSubmodule(py::module &m) {
  //----------------------------------------------------------------------------
  // Mapping of the top-level PassManager
  //----------------------------------------------------------------------------
  py::class_<PyPassManager>(m, "PassManager", py::module_local())
      .def(py::init<>([](const std::string &anchorOp,
                         DefaultingPyMlirContext context) {
             MlirPassManager passManager = mlirPassManagerCreateOnOperation(
                 context->get(),
                 mlirStringRefCreate(anchorOp.data(), anchorOp.size()));
             return new PyPassManager(passManager);
           }),
           py::arg("anchor_op") = py::str("any"),
           py::arg("context") = py::none(),
           "Create a new PassManager for the current (or provided) Context.")
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR,
                             &PyPassManager::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyPassManager::createFromCapsule)
      .def("_testing_release", &PyPassManager::release,
           "Releases (leaks) the backing pass manager (testing)")
      .def(
          "enable_ir_printing",
          [](PyPassManager &passManager) {
            mlirPassManagerEnableIRPrinting(passManager.get());
          },
          "Enable mlir-print-ir-after-all.")
      .def(
          "enable_verifier",
          [](PyPassManager &passManager, bool enable) {
            mlirPassManagerEnableVerifier(passManager.get(), enable);
          },
          py::arg("enable"), "Enable / disable verify-each.")
      .def_static(
          "parse",
          [](const std::string &pipeline, DefaultingPyMlirContext context) {
            MlirPassManager passManager = mlirPassManagerCreate(context->get());
            PyPrintAccumulator errorMsg;
            MlirLogicalResult status = mlirParsePassPipeline(
                mlirPassManagerGetAsOpPassManager(passManager),
                mlirStringRefCreate(pipeline.data(), pipeline.size()),
                errorMsg.getCallback(), errorMsg.getUserData());
            if (mlirLogicalResultIsFailure(status))
              throw py::value_error(std::string(errorMsg.join()));
            return new PyPassManager(passManager);
          },
          py::arg("pipeline"), py::arg("context") = py::none(),
          "Parse a textual pass-pipeline and return a top-level PassManager "
          "that can be applied on a Module. Throw a ValueError if the pipeline "
          "can't be parsed")
      .def(
          "add",
          [](PyPassManager &passManager, const std::string &pipeline) {
            PyPrintAccumulator errorMsg;
            MlirLogicalResult status = mlirOpPassManagerAddPipeline(
                mlirPassManagerGetAsOpPassManager(passManager.get()),
                mlirStringRefCreate(pipeline.data(), pipeline.size()),
                errorMsg.getCallback(), errorMsg.getUserData());
            if (mlirLogicalResultIsFailure(status))
              throw py::value_error(std::string(errorMsg.join()));
          },
          py::arg("pipeline"),
          "Add textual pipeline elements to the pass manager. Throws a "
          "ValueError if the pipeline can't be parsed.")
      .def(
          "run",
          [](PyPassManager &passManager, MlirOperation &op) {
            MlirLogicalResult status = mlirPassManagerRunOnOp(
                passManager.get(), op);
            if (mlirLogicalResultIsFailure(status))
              throw MLIRError("Failure while executing pass pipeline");
          },
          py::arg("operation"),
          "Run the pass manager on the provided operation, raising an "
          "MLIRError on failure.")
      .def(
          "emit_kokkos",
          [](PyPassManager &passManager, PyModule &module, const char* cxxSourceFile, const char* pySourceFile) {
            MlirLogicalResult status =
                lapisEmitKokkos(module.get(), cxxSourceFile, pySourceFile);
            if (mlirLogicalResultIsFailure(status))
              throw MLIRError("Failure while raising MLIR to Kokkos C++ source code.");
          },
          py::arg("module"), py::arg("cxx_source_file"), py::arg("py_source_file"),
          "Emit Kokkos C++ and Python wrappers for the given module, and throw a RuntimeError on failure.")
      .def(
          "emit_kokkos_sparse",
          [](PyPassManager &passManager, PyModule &module, const char* cxxSourceFile, const char* pySourceFile, bool useHierarchical, bool isLastKernel) {
            MlirLogicalResult status =
                lapisEmitKokkosSparse(module.get(), cxxSourceFile, pySourceFile, useHierarchical, isLastKernel);
            if (mlirLogicalResultIsFailure(status))
              throw MLIRError("Failure while raising MLIR to Kokkos C++ source code.");
          },
          py::arg("module"), py::arg("cxx_source_file"), py::arg("py_source_file"), py::arg("use_hierarchical"), py::arg("is_final_kernel"),
          "Emit Kokkos C++ and Python wrappers for the given sparse module, and throw a RuntimeError on failure.")
      .def(
          "__str__",
          [](PyPassManager &self) {
            MlirPassManager passManager = self.get();
            PyPrintAccumulator printAccum;
            mlirPrintPassPipeline(
                mlirPassManagerGetAsOpPassManager(passManager),
                printAccum.getCallback(), printAccum.getUserData());
            return printAccum.join();
          },
          "Print the textual representation for this PassManager, suitable to "
          "be passed to `parse` for round-tripping.");
}

