add_mlir_library(LAPISPass
  Pass.cpp

  DEPENDS
  mlir-generic-headers

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  MLIRPass
  MLIRTargetKokkosCpp
  )
