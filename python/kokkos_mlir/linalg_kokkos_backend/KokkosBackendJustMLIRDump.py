# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import ctypes
import numpy as np
import sys
import os
import subprocess
from io import StringIO
import tempfile

from kokkos_mlir.ir import *
from kokkos_mlir.passmanager import *
from kokkos_mlir.runtime import *
# Imported for side effects.
import kokkos_mlir.all_passes_registration
import kokkos_mlir.dialects.torch

from .abc import LinalgKokkosBackend

__all__ = [
    "LinalgKokkosBackend",
]

LOWERING_PIPELINE = ",".join([
    "builtin.func(refback-generalize-tensor-pad)",
    "builtin.func(scf-bufferize)",
    "builtin.func(tm-tensor-bufferize)",
    "builtin.func(linalg-bufferize)",
    "func-bufferize",
    "arith-bufferize",
    "builtin.func(tensor-bufferize)",
    "builtin.func(finalizing-bufferize)",
    "refback-insert-rng-globals",
    "builtin.func(lower-affine)"
])

#LOWERING_PIPELINE = ",".join([
#    "builtin.func(refback-generalize-tensor-pad)",
#    "builtin.func(scf-bufferize)",
#    "builtin.func(tm-tensor-bufferize)",
#    "builtin.func(linalg-bufferize)",
#    "func-bufferize",
#    "arith-bufferize",
#    "builtin.func(tensor-bufferize)",
#    "builtin.func(finalizing-bufferize)",
#    "refback-insert-rng-globals",
#    "builtin.func(tm-tensor-to-loops)",
#    "builtin.func(convert-linalg-to-parallel-loops)",
#    #"builtin.func(convert-linalg-to-loops)",
#    "builtin.func(lower-affine)",
#])

### Original RefBackend pipeline for full lowering to LLVM (is more low-level than what we need for Kokkos)
#LOWERING_PIPELINE = ",".join([
#    "builtin.func(refback-generalize-tensor-pad)",
#    # Bufferize.
#    "builtin.func(scf-bufferize)",
#    "builtin.func(tm-tensor-bufferize)",
#    "builtin.func(linalg-bufferize)",
#    "func-bufferize",
#    "arith-bufferize",
#    "builtin.func(tensor-bufferize)",
#    "builtin.func(finalizing-bufferize)",
#    # Munge to make it ExecutionEngine compatible.
#    # Specifically, we rewrite calling convention boundaries to be in terms
#    # of unranked memref, and we rewrite the return to actually be a
#    # callback that consumes the return (the final munged function always
#    # returns void at the C level -- we get the return value by providing the
#    # callback).
#    "refback-munge-calling-conventions",
#    # Insert global variable and instruction sequence for getting the next
#    # global seed used in stateful rng.
#    "refback-insert-rng-globals",
#    # Lower to LLVM
#    "builtin.func(tm-tensor-to-loops)",
#    "builtin.func(refback-munge-memref-copy)",
#    "builtin.func(convert-linalg-to-loops)",
#    "builtin.func(lower-affine)",
#    "convert-scf-to-cf",
#    "builtin.func(refback-expand-ops-for-llvm)",
#    "builtin.func(arith-expand)",
#    "builtin.func(convert-math-to-llvm)",
#    "convert-linalg-to-llvm",
#    "convert-memref-to-llvm",
#    #"builtin.func(convert-arith-to-llvm)",
#    #"convert-func-to-llvm",
#    #"convert-cf-to-llvm",
#    #"reconcile-unrealized-casts",
#])

class KokkosBackendLinalgOnTensorsBackend(LinalgKokkosBackend):
    """Main entry-point for the Kokkos LinAlg backend."""

    def __init__(self):
        super().__init__()

    def compile(self, module: Module):
        """Compiles an imported module, with a flat list of functions.
        The module is expected to be in linalg-on-tensors + scalar code form.
        TODO: More clearly define the backend contract. Generally this will
        extend to support globals, lists, and other stuff.

        Args:
          module: The MLIR module consisting of funcs in the torch
            dialect.
        Returns:
          An opaque, backend specific compiled artifact object that can be
          passed to `load`.
        """

        module_name = "MyModule"
        #original_stderr = sys.stderr
        #sys.stderr = StringIO()
        asm_for_error_report = module.operation.get_asm(
            large_elements_limit=10, enable_debug_info=True)
        # Lower module in place to make it ready for compiler backends.
        with module.context:
            pm = PassManager.parse(LOWERING_PIPELINE)
            pm.run(module)
            moduleAsm = module.operation.get_asm(large_elements_limit=10, enable_debug_info=False)
            filename = "/run/media/bmkelle/6b046acd-4efd-4f43-a567-1699d733ba4c/mlir-trilinos/torch-mlir/examples/mlir_kokkos/dump.mlir"
            with open(filename, 'w') as f:
                f.write(asm_for_error_report)
                print("Wrote out MLIR dump to /run/media/bmkelle/6b046acd-4efd-4f43-a567-1699d733ba4c/mlir-trilinos/torch-mlir/examples/mlir_kokkos/dump.mlir")
            return None
            # TODO: this is hardcoded now, but what should it be in the long term?
            # Make the output location a parameter of compile()? Use temp dir?
            moduleRoot = "/run/media/bmkelle/6b046acd-4efd-4f43-a567-1699d733ba4c/mlir-trilinos/torch-mlir/examples/mlir_kokkos"
            buildDir = moduleRoot + "/build"
            # First, clean existing CMakeCache.txt from build if it exists
            if os.path.isfile(buildDir + '/CMakeCache.txt'):
                os.remove(buildDir + '/CMakeCache.txt')
            # Create the source and build directories
            os.makedirs(buildDir, exist_ok=True)
            # Generate Kokkos C++ source from the module.
            print("Emitting module as Kokkos C++...")
            pm.emit_kokkos(module, moduleRoot + "/mlir_kokkos_module.cpp", moduleRoot + "/mlir_kokkos.py")
            # Now that we have a Kokkos source file, generate the CMake to build it into a shared lib,
            # using $KOKKOS_ROOT as the kokkos installation.
            if 'KOKKOS_ROOT' not in os.environ:
                raise Exception("KOKKOS_ROOT must be defined as an environment variable, and point to a Kokkos installation!")
            kokkosDir = os.environ['KOKKOS_ROOT']
            print("Generating CMakeLists.txt...")
            cmake = open(moduleRoot + "/CMakeLists.txt", "w")
            cmake.write("project(mlir_kokkos)\n")
            cmake.write("cmake_minimum_required(VERSION 3.16 FATAL_ERROR)\n")
            cmake.write("find_package(Kokkos REQUIRED\n")
            cmake.write(" PATHS ")
            cmake.write(kokkosDir)
            cmake.write("/lib64/cmake/Kokkos)\n")
            cmake.write("add_library(mlir_kokkos_module SHARED mlir_kokkos_module.cpp)\n")
            cmake.write("target_link_libraries(mlir_kokkos_module Kokkos::kokkos)\n")
            cmake.close()
            # Now configure the project and build the shared library from the build dir
            print("Configuring build...")
            subprocess.run(['cmake', moduleRoot], cwd=buildDir)
            print("Building module...")
            buildOut = subprocess.run(['make'], cwd=buildDir, shell=True)
            print("Importing module...")
            sys.path.insert(0, moduleRoot)
            import mlir_kokkos
            return mlir_kokkos.MLIRKokkosModule(buildDir + "/libmlir_kokkos_module.so")

