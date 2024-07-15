add_mlir_public_c_api_library(LapisCAPI
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

lapis_target_includes(LapisCAPI)
add_subdirectory(IR)
