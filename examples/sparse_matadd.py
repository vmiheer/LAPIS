import torch
from torch import Tensor
import torch_mlir
from kokkos_mlir.linalg_kokkos_backend import KokkosBackend
from torch import nn


class Adder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return a + b


def main():
    indices_1 = torch.tensor([[0, 1], [1, 0], [1, 2]])
    indices_2 = torch.tensor([[0, 2], [1, 0], [1, 2]])
    v_1 = torch.tensor([3,      2,      5    ], dtype=torch.float32)
    v_2 = torch.tensor([3,      4,      5    ], dtype=torch.float32)

    a = torch.sparse_coo_tensor(indices_1.t(), v_1, (5, 5))
    b = torch.sparse_coo_tensor(indices_2.t(), v_2, (5, 5))

    m = Adder()
    m.train(False)

    mlir_module = torch_mlir.compile(
        m, (a, b), output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
    )

    c = m.forward(a, b)
    print(c.to_dense())

    backend = KokkosBackend.KokkosBackendLinalgOnTensorsBackend(dump_mlir=True)
    k_backend = backend.compile(mlir_module)

    c = k_backend.forward(a.to_dense(), b.to_dense())
    print(c)


if __name__ == "__main__":
    main()
