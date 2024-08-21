
set(mlir_cmake_builddir "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/lapis")
get_property(MLIR_EXPORTS GLOBAL PROPERTY MLIR_EXPORTS)
export(TARGETS ${MLIR_EXPORTS} FILE ${mlir_cmake_builddir}/LAPISTargets.cmake)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/LAPISConfig.cmake.in
  ${mlir_cmake_builddir}/LAPISConfig.cmake
  @ONLY)
# configure_file(
#   ${CMAKE_CURRENT_SOURCE_DIR}/MLIRConfigVersion.cmake.in
#   ${mlir_cmake_builddir}/MLIRConfigVersion.cmake
#   @ONLY)

# install_distribution_exports(MLIR)
#   install(FILES
#     ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/LAPISConfig.cmake
#     DESTINATION ${MLIR_INSTALL_PACKAGE_DIR}
#     COMPONENT mlir-cmake-exports)
