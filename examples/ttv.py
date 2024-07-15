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

compressed = pt.compressed
dense = pt.dense

# Define formats for storing the sparse matrix and dense vectors.

a = pt.tensor([5, 5, 5], [pt.compressed, pt.compressed, pt.compressed], dtype=pt.float64)
b = pt.tensor([5], [pt.dense], dtype=pt.float64)
c = pt.tensor([5, 5], [pt.compressed, pt.compressed], dtype=pt.float64)

a.insert([0,1,0], 3.0)
a.insert([1,0,1], 2.0)
a.insert([1,2,3], 5.0)
a.insert([2,1,4], 3.0)
a.insert([3,0,1], 2.0)
a.insert([4,2,0], 5.0)

b.insert([0], 3.0)
b.insert([1], 4.0)
b.insert([2], 5.0)

i, j, k = pt.get_index_vars(3) 

c[i,j] = a[i,j,k] * b[k]

##########################################################################

# Perform the SpMV computation and write the result to file
with tempfile.TemporaryDirectory() as test_dir:
  print("Compiling, running spmv and writing result to c.tns")
  ptio.write_kokkos("c.tns", c)
  # Comment above and uncomment below to run standard PyTACO (non-Kokkos pipeline)
  # pt.write("c_gold.tns", c)
  # Then c.tns and c_gold.tns can be compared to test correctness

