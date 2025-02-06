from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor, TangentTensor
from torch import Tensor
from torch.nn import Module


class ToManifold(Module):
    """Lift a tensor to a ManifoldTensor."""

    def __init__(self, manifold: Manifold, *args, man_dim: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifold = manifold
        self.man_dim = man_dim

    def forward(self, x: Tensor) -> ManifoldTensor:
        x_tangent = TangentTensor(data=x, manifold=self.manifold, man_dim=self.man_dim)
        return self.manifold.expmap(x_tangent)
