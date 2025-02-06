from hypll.nn.modules.linear import HLinear
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import check_if_man_dims_match, check_if_manifolds_match
from torch import Tensor


class HMLR(HLinear):
    """Poincare multinomial logistic regression layer"""

    def forward(self, x: ManifoldTensor) -> Tensor:
        check_if_manifolds_match(layer=self, input=x)
        check_if_man_dims_match(layer=self, man_dim=-1, input=x)
        return self.manifold.hyperplane_dists(x=x, z=self.z, r=self.bias)
