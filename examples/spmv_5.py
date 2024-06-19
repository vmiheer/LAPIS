# RUN: SUPPORTLIB=%mlir_lib_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

import numpy as np
import os
import sys
import tempfile
import shutil

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)

from kokkos_mlir.tools import mlir_pytaco_api as pt
from kokkos_mlir.tools import mlir_pytaco_io as ptio
from kokkos_mlir.tools import testing_utils as utils

###### This PyTACO part is taken from the TACO open-source project. ######
# See http://tensor-compiler.org/docs/scientific_computing/index.html.

compressed = pt.compressed
dense = pt.dense

# Define formats for storing the sparse matrix and dense vectors.
csr = pt.format([dense, compressed])
dv = pt.format([dense])

A = pt.tensor([5, 5], [pt.dense, pt.compressed], dtype=pt.float64)
b = pt.tensor([5], [pt.dense], dtype=pt.float64)
c = pt.tensor([A.shape[0]], [pt.dense], dtype=pt.float64)

A.insert([0,1], 3.0)
A.insert([1,0], 2.0)
A.insert([1,2], 5.0)

b.insert([0], 3.0)
b.insert([1], 4.0)
b.insert([2], 5.0)

i, j = pt.get_index_vars(2) 
c[i] = A[i,j] * b[j]

##########################################################################

# Perform the SpMV computation and write the result to file
with tempfile.TemporaryDirectory() as test_dir:
  print("Compiling, running spmv and writing result to c.tns")
  ptio.write_kokkos("c.tns", c, "parallelization-strategy=any-storage-any-loop kokkos-uses-hierarchical")

shutil.copyfile("mlir_kokkos/mlir_kokkos_module.cpp", "mlir_kokkos/mlir_kokkos_module_5.cpp")
