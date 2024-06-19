# Main API shared library.
add_mlir_public_c_api_library(KOKKOS_MLIRCAPIIR
  Pass.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
)

set_property(GLOBAL APPEND PROPERTY KokkosMLIRCAPI KOKKOS_MLIRCAPIIR)