add_mlir_library(MLIRKokkosPass
  Pass.cpp

  DEPENDS
  mlir-generic-headers

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  MLIRPass
  MLIRTargetKokkosCpp
  )
