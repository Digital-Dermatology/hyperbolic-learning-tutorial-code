from hypll.manifolds.poincare_ball import PoincareBall, Curvature
from torch.nn import Sequential

from .euclidean import make_euclidean_backbone
from ..layers.hmlr import HMLR
from ..layers.to_manifold import ToManifold


def make_last_hyperbolic_net(
    out_channels: int = 1, *args, **kwargs
) -> Sequential:
    backbone, backbone_channels = make_euclidean_backbone(*args, **kwargs)
    manifold = PoincareBall(c=Curvature(requires_grad=True))
    neck = ToManifold(manifold=manifold)
    head = HMLR(
        in_features=backbone_channels, out_features=out_channels, manifold=manifold, bias=True
    )
    return Sequential(backbone, neck, head)
