#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#csrv = #sparse_tensor.encoding<{ map = (d0, d1, d2) ->
    (d0 : dense, d1 : compressed, d2 : dense) }>
#dense = #sparse_tensor.encoding<{ map = (d0, d1) ->
    (d0 : dense, d1 : dense) }>
#densev = #sparse_tensor.encoding<{ map = (d0, d1, d2) ->
    (d0 : dense, d1 : dense, d2 : dense) }>
#csr = #sparse_tensor.encoding<{ map = (d0, d1) ->
    (d0 : dense, d1 : compressed) }>
#partCsr = #part_tensor.encoding<{
  partConst = 1,
  sparseAttributes = #csr
}>
#partDensev = #part_tensor.encoding<{
  partConst = 1,
  sparseAttributes = #densev
}>
#proj_tensor_map = affine_map<(dh, nh)[Nh] -> (dh * Nh + nh)>
#input_proj_map = {
  indexing_maps = [
    affine_map<(n, f, dh, nh) -> (n, f)>,  // X (in)
    affine_map<(n, f, dh, nh) -> (dh, nh, f)>,  // Q_Proj (in)
    affine_map<(n, f, dh, nh) -> (n, dh, nh)>  // Q (out)
  ],
  iterator_types = ["parallel", "reduction", "parallel", "parallel"]
}
#output_proj_map = {
  indexing_maps = [
    affine_map<(n, f, dh, nh) -> (n, dh, nh)>,  // Attn (in)
    affine_map<(n, f, dh, nh) -> (dh, nh, f)>,  // O_Proj (in)
    affine_map<(n, f, dh, nh) -> (n, f)>  // O (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction", "reduction"]
}
#bsddmm_map = {
  indexing_maps = [
    affine_map<(n1, n2, dh, nh) -> (n1, dh, nh)>,  // q (in)
    affine_map<(n1, n2, dh, nh) -> (n2, dh, nh)>,  // k (in)
    affine_map<(n1, n2, dh, nh) -> (n1, n2)>,  // A (in)
    affine_map<(n1, n2, dh, nh) -> (n1, n2, nh)>   // attn (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction", "parallel"],
  doc = "attn(n1, n2, nh) = q(n1, dh, nh) * k(n2, dh, nh)"
}
#bspmm_map = {
  indexing_maps = [
    affine_map<(n1, n2, dh, nh) -> (n1, n2, nh)>,  // attn (in)
    affine_map<(n1, n2, dh, nh) -> (n2, dh, nh)>,  // v (in)
    affine_map<(n1, n2, dh, nh) -> (n1, dh, nh)>   // out (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction", "parallel"],
  doc = "out(n1, dh, nh) = attn(n1, n2, nh) * v(n2, dh, nh)"
}

module {
  func.func @pte_local_sparse_mha(%A: tensor<?x?xf32, #csr>,
    %Q: tensor<?x?x?xf32, #densev>, %K: tensor<?x?x?xf32, #densev>
  )
      ->  tensor<?x?x?xf32, #csrv>
  {
    %c0_index = arith.constant 0 : index
    %c1_index = arith.constant 1 : index
    %c2_index = arith.constant 2 : index
    %c0_f32 = arith.constant 0.0 : f32
    %N1 = tensor.dim %A, %c0_index : tensor<?x?xf32, #csr>
    %N2 = tensor.dim %A, %c1_index : tensor<?x?xf32, #csr>
    %dh = tensor.dim %Q, %c1_index : tensor<?x?x?xf32, #densev>
    %nh = tensor.dim %Q, %c2_index : tensor<?x?x?xf32, #densev>
    // attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
    %attn0 = tensor.empty (%N1, %N2, %nh) : tensor<?x?x?xf32, #csrv>
    %attn2 = linalg.generic #bsddmm_map ins(%Q, %K, %A: tensor<?x?x?xf32, #densev>,
      tensor<?x?x?xf32, #densev>, tensor<?x?xf32, #csr>)
      outs(%attn0: tensor<?x?x?xf32, #csrv>) {
      ^bb0(%q: f32, %k: f32, %a: f32, %attn: f32):  // no predecessors
        %0 = arith.mulf %q, %k : f32
        %1 = arith.mulf %0, %a : f32
        %2 = arith.addf %1, %attn: f32
        linalg.yield %2 : f32
    } -> tensor<?x?x?xf32, #csrv>
    return %attn2 : tensor<?x?x?xf32, #csrv>

  }
}


// Local Variables:
// rmsbolt-command: "lapis-opt-old --sparse-compiler-kokkos='pt-backend=mpi'"
// rmsbolt-automatic-recompile: on-save
// End:
