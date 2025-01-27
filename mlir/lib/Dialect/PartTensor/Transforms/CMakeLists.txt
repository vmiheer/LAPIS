add_mlir_dialect_library(MLIRPartTensorTransforms
  LinalgToPartTensor.cpp
  PartTensorConversion.cpp
  PartTensorPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/PartTensor

  DEPENDS
  MLIRPartTensorPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRComplexDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSCFUtils
  MLIRPartTensor
  MLIRTensorDialect
  MLIRTransforms
  MLIRVectorDialect
)

if (LAPIS_USE_FMTLIB)
target_link_libraries(MLIRPartTensorTransforms
  PUBLIC
  fmt::fmt
)
endif()
