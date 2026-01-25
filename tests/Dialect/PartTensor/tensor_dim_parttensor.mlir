// RUN: %lapis-opt --linalg-to-parttensor --part-tensor-conversion=pt-backend=mpi %s
// This test ensures that `tensor.dim` on a PartTensor is considered illegal
// by the PartTensor conversion pass and thus causes the conversion to fail.

#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
#partEncoding = #part_tensor.encoding<{
  partConst = 1,
  sparseAttributes = #sparse
}>

module {
  func.func @test(%A: tensor<?x?xf32, #partEncoding>) {
    %c0 = arith.constant 0 : index
    // This should be considered illegal by the conversion pass if %A is a
    // PartTensor value and force the pass to fail when no conversion pattern
    // for tensor.dim on PartTensors is provided.
    %sz = tensor.dim %A, %c0 : tensor<?x?xf32, #partEncoding>
    return
  }
}
