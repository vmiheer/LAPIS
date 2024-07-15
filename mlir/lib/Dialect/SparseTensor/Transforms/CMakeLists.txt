if(LAPIS_ENABLE_PART_TENSOR)
  add_mlir_dialect_library(Kokkos_MLIRSparseTensorTransforms
    SparseKokkosCodegen.cpp
    SparseTensorPasses.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

    DEPENDS
    LAPISSparseTensorPassIncGen

    LINK_LIBS PUBLIC
    MLIRSparseTensorTransforms
    MLIRPartTensorTransforms
  )
else()
  add_mlir_dialect_library(Kokkos_MLIRSparseTensorTransforms
    SparseKokkosCodegen.cpp
    SparseTensorPasses.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

    DEPENDS
    LAPISSparseTensorPassIncGen

    LINK_LIBS PUBLIC
    MLIRSparseTensorTransforms
  )
endif()