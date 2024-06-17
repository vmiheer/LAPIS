# RUN: SUPPORTLIB=%mlir_lib_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

import unittest
import numpy as np
import os
import sys
import tempfile

from multiprocessing import Process, Queue

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)

from tools import mlir_pytaco_api as pt
from tools import testing_utils as utils

###### This PyTACO part is taken from the TACO open-source project. ######
# See http://tensor-compiler.org/docs/scientific_computing/index.html.

def spmv_test(queue, matrix_format, useKokkos, options = ""):
  A = pt.tensor([5, 5], matrix_format, dtype=pt.float64)
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

  filename_PyTACO = "c.tns"
  filename_Kokkos = "c_kokkos.tns"

  if useKokkos:
    if os.path.exists(filename_Kokkos):
        os.remove(filename_Kokkos)    
    try:
      with tempfile.TemporaryDirectory() as test_dir:
        pt.write_kokkos(filename_Kokkos, c, options)
      queue.put(utils.compare_sparse_tns(filename_PyTACO, filename_Kokkos))
    except:
      queue.put(False)
  else:
    if os.path.exists(filename_PyTACO):
        os.remove(filename_PyTACO)       
    with tempfile.TemporaryDirectory() as test_dir:
      pt.write(filename_PyTACO, c)

def compare_spmv(matrix_format, options = ""):
  queue = Queue()
  p_PyTACO = Process(target=spmv_test, args=(queue, matrix_format, False))
  p_Kokkos = Process(target=spmv_test, args=(queue, matrix_format, True, options))
  p_PyTACO.start()
  p_PyTACO.join()
  p_Kokkos.start()
  p_Kokkos.join()  
  return queue.get()

class TestSPMV(unittest.TestCase):

  def test_serial(self):
    compressed = pt.compressed
    dense = pt.dense

    for matrix_format in [pt.format([dense, compressed]), pt.format([compressed, compressed]), pt.format([compressed, dense]), pt.format([dense, dense])]:
      self.assertTrue(compare_spmv(matrix_format, ""))

  def test_parallel(self):
    compressed = pt.compressed
    dense = pt.dense

    for matrix_format in [pt.format([dense, compressed]), pt.format([compressed, compressed]), pt.format([compressed, dense]), pt.format([dense, dense])]:
      self.assertTrue(compare_spmv(matrix_format, "parallelization-strategy=any-storage-outer-loop"))

  def test_parallel_dense(self):
    compressed = pt.compressed
    dense = pt.dense

    for matrix_format in [pt.format([dense, compressed]), pt.format([compressed, compressed]), pt.format([compressed, dense]), pt.format([dense, dense])]:
      self.assertTrue(compare_spmv(matrix_format, "parallelization-strategy=dense-any-loop"))

  def test_parallel_hierarchical(self):
    compressed = pt.compressed
    dense = pt.dense

    for matrix_format in [pt.format([dense, compressed]), pt.format([compressed, compressed]), pt.format([compressed, dense]), pt.format([dense, dense])]:
      self.assertTrue(compare_spmv(matrix_format, "parallelization-strategy=any-storage-any-loop kokkos-uses-hierarchical"))  

  def test_parallel_with_inner_serial_loop(self):
    compressed = pt.compressed
    dense = pt.dense

    for matrix_format in [pt.format([dense, compressed]), pt.format([compressed, compressed]), pt.format([compressed, dense]), pt.format([dense, dense])]:
      self.assertTrue(compare_spmv(matrix_format, "parallelization-strategy=any-storage-any-loop"))  

if __name__ == '__main__':
    unittest.main()
