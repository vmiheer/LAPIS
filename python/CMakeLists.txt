include(AddMLIRPython)

# Disables generation of "version soname" (i.e. libFoo.so.<version>), which
# causes pure duplication as part of Python wheels.
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(KOKKOS_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kokkos_mlir")


# We vendor our own MLIR instance in the `kokkos_mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=kokkos_mlir.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(KokkosMLIRPythonSources)
declare_mlir_python_sources(KokkosMLIRPythonExtensions)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(KokkosMLIRPythonExtensions.Main
  MODULE_NAME _kokkosMlir
  ADD_TO_PARENT KokkosMLIRPythonExtensions
  SOURCES
    KokkosMLIRModule.cpp
  EMBED_CAPI_LINK_LIBS
    KokkosMLIRCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Generate packages and shared library
# Downstreams typically will not use these, but they are useful for local
# testing.
################################################################################

set(_source_components
  # TODO: Core is now implicitly building/registering all dialects, increasing
  # build burden by ~5x. Make it stop.
  # TODO: Reduce dependencies. We need ExecutionEngine and a bunch of passes
  # for the reference backend, but logically they can be separate. But seemingly
  # the only way to handle that is to create a separate mlir python package
  # tree, which seems excessive.
  MLIRPythonSources
  MLIRPythonExtension.Core
  MLIRPythonExtension.RegisterEverything
)

add_mlir_python_common_capi_library(KokkosMLIRAggregateCAPI
  INSTALL_COMPONENT KokkosMLIRPythonModules
  INSTALL_DESTINATION python_packages/kokkos_mlir/kokkos_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${KOKKOS_MLIR_PYTHON_PACKAGES_DIR}/kokkos_mlir/kokkos_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(KokkosMLIRPythonModules
  ROOT_PREFIX "${KOKKOS_MLIR_PYTHON_PACKAGES_DIR}/kokkos_mlir/kokkos_mlir"
  INSTALL_PREFIX "python_packages/kokkos_mlir/kokkos_mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    KokkosMLIRAggregateCAPI
  )
