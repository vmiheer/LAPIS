//===-- mlir-c/Pass.h - C API to Pass Management ------------------*- C -*-===//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to MLIR pass manager.
//
//===----------------------------------------------------------------------===//

#ifndef LAPIS_C_PASS_H
#define LAPIS_C_PASS_H

#include "mlir-c/Pass.h"

#ifdef __cplusplus
extern "C" {
#endif


//MLIR_CAPI_EXPORTED 
MlirLogicalResult
lapisEmitKokkos(MlirModule module, const char* cxxSourceFile, const char* pySourceFile);

//MLIR_CAPI_EXPORTED 
MlirLogicalResult
lapisEmitKokkosSparse(MlirModule module, const char* cxxSourceFile, const char* pySourceFile, bool useHierarchical, bool isLastKernel);

#ifdef __cplusplus
}
#endif

#endif // LAPIS_C_PASS_H
