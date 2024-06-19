add_mlir_dialect_library(Kokkos_MLIRSparseTensorTransforms
  SparseKokkosCodegen.cpp
  SparseTensorPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

  DEPENDS
  KOKKOS_MLIRSparseTensorPassIncGen

  LINK_LIBS PUBLIC
  MLIRSparseTensorTransforms
  MLIRPartTensorTransforms
)
