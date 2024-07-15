# Main API shared library.
add_mlir_public_c_api_library(LAPISCAPIIR
  Pass.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
)

set_property(GLOBAL APPEND PROPERTY LapisCAPI LAPISCAPIIR)