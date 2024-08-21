# Exclude these from libMLIR.so because the JIT infrastructure
# is a big dependency which most don't need.
if(NOT LAPIS_ENABLE_PART_TENSOR)
  message(STATUS "Not building LapisPartTensorExecutionEngine because \
    MLIR_ENABLE_EXECUTION_ENGINE=OFF")
  return()
endif()

add_mlir_library(LapisPartTensorExecutionEngine
  PartTensorRuntime.cpp
  )
