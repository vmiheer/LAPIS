set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(lapis-translate
  lapis-translate.cpp
  )
llvm_update_compile_flags(lapis-translate)
target_link_libraries(lapis-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  ${test_libs}
  MLIRIR
  MLIRParser
  LAPISPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRSupport
  )

mlir_check_link_libraries(lapis-translate)
