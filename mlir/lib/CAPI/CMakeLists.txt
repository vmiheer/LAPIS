add_mlir_public_c_api_library(KokkosMLIRCAPI
  Registration.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/mlir-c/

  ENABLE_AGGREGATION
  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSparseTensorKokkosPipelines
)

kokkos_mlir_target_includes(KokkosMLIRCAPI)
add_subdirectory(IR)
