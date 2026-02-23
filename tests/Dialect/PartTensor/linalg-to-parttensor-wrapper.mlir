// RUN: %lapis-opt --linalg-to-parttensor --part-tensor-conversion=pt-backend=mpi %s | FileCheck %s

#csr = #sparse_tensor.encoding<{ map = (d0, d1) ->
    (d0 : dense, d1 : compressed) }>

#dense = #sparse_tensor.encoding<{ map = (d0) ->
    (d0 : dense) }>

module {
  // Test that linalg operation is extracted into a separate wrapper function
  // instead of being inlined into the distributed function
  
  func.func @spmv(%A: tensor<?x?xf32, #csr>,
                  %x: tensor<?xf32, #dense>) -> tensor<?xf32, #dense> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j) -> (i, j)>,
        affine_map<(i, j) -> (j)>,
        affine_map<(i, j) -> (i)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%A, %x : tensor<?x?xf32, #csr>, tensor<?xf32, #dense>)
      outs(%x : tensor<?xf32, #dense>) {
      ^bb0(%a: f32, %xi: f32, %yi: f32):
        %mul = arith.mulf %a, %xi : f32
        %add = arith.addf %mul, %yi : f32
        linalg.yield %add : f32
    } -> tensor<?xf32, #dense>
    
    return %result : tensor<?xf32, #dense>
  }
}

// CHECK: call @linalg_op_wrapper
// CHECK: func.func private @linalg_op_wrapper
// CHECK: linalg.generic
