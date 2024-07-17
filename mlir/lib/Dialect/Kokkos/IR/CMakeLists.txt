add_mlir_dialect_library(MLIRKokkosDialect
   KokkosDialect.cpp

   ADDITIONAL_HEADER_DIRS
   ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Kokkos

   DEPENDS
   MLIRKokkosIncGen

   LINK_LIBS PUBLIC
   MLIRDialect
   MLIRIR
   MLIRDialectUtils
   MLIRIR
   MLIRInferTypeOpInterface
   MLIRSupport
   )
