# RUN: SUPPORTLIB=%mlir_lib_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

import filecmp
import numpy as np
import os
import sys
import tempfile

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)

from kokkos_mlir.tools import mlir_pytaco_api as pt
from kokkos_mlir.tools import mlir_pytaco_io as ptio
from kokkos_mlir.tools import testing_utils as utils

from kokkos_mlir.linalg_kokkos_backend import KokkosBackend


def main():
    a = pt.tensor([5, 5], [pt.compressed, pt.compressed])
    b = pt.tensor([5, 5], [pt.compressed, pt.compressed])
    c = pt.tensor([a.shape[0], b.shape[1]], [pt.compressed, pt.compressed])
    c_kokkos = pt.tensor([a.shape[0], b.shape[1]], [pt.compressed, pt.compressed])

    a.insert([0,1], 3.0)
    a.insert([1,0], 2.0)
    a.insert([1,2], 5.0)

    b.insert([0,2], 3.0)
    b.insert([1,0], 4.0)
    b.insert([1,2], 5.0)

    i, j = pt.get_index_vars(2) 
    c[i,j] = a[i,j] + b[i,j]
    c_kokkos[i,j] = a[i,j] + b[i,j]

    
    golden_file = "gold_c.tns"
    out_file = "c.tns"
    ptio.write(golden_file, c)
    ptio.write_kokkos(out_file, c_kokkos)
    #
    # CHECK: Compare result True
    #
    print(f"Compare result {utils.compare_sparse_tns(golden_file, out_file)}")

if __name__ == "__main__":
    main()
