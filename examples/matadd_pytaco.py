# RUN: SUPPORTLIB=%mlir_lib_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

import numpy as np
import os
import sys
import tempfile

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)

from lapis.tools import mlir_pytaco_api as pt
from lapis.tools import mlir_pytaco_io as ptio
from lapis.tools import testing_utils as utils

###### This PyTACO part is taken from the TACO open-source project. ######
# See http://tensor-compiler.org/docs/scientific_computing/index.html.

m = 50
n = 50

a = pt.tensor([m,n], [pt.dense,pt.dense], dtype=pt.float64)
b = pt.tensor([m,n], [pt.dense,pt.dense], dtype=pt.float64)
c = pt.tensor([m,n], [pt.dense,pt.dense], dtype=pt.float64)

for i in range(0, m):
  for j in range(0, n):
    a.insert([i,j], 5.0)
    b.insert([i,j], 1.0)

i, j = pt.get_index_vars(2) 
c[i,j] = a[i,j]+b[i,j]

##########################################################################

# Perform the SpMV computation and write the result to file
with tempfile.TemporaryDirectory() as test_dir:
  print("Compiling, running spmv and writing result to c.tns")
  ptio.write_kokkos("c.tns", c, "parallelization-strategy=dense-any-loop")

