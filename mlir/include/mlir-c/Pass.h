//===-- mlir-c/Pass.h - C API to Pass Management ------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to MLIR pass manager.
//
//===----------------------------------------------------------------------===//

#ifndef KOKKOS_MLIR_C_PASS_H
#define KOKKOS_MLIR_C_PASS_H

#include "mlir-c/Pass.h"

#ifdef __cplusplus
extern "C" {
#endif


MLIR_CAPI_EXPORTED MlirLogicalResult
mlirPassManagerEmitKokkos(MlirPassManager passManager, MlirModule module, const char* cxxSourceFile, const char* pySourceFile);

MLIR_CAPI_EXPORTED MlirLogicalResult
mlirPassManagerEmitKokkosSparse(MlirPassManager passManager, MlirModule module, const char* cxxSourceFile, const char* pySourceFile, bool useHierarchical, bool isLastKernel);

#ifdef __cplusplus
}
#endif

#endif // KOKKOS_MLIR_C_PASS_H
