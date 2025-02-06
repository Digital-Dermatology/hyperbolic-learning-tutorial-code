from typing import Sequence, Tuple

import torch
from hypll.manifolds import Manifold
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.nn import HLinear, HConvolution2d, HMaxPool2d, HReLU
from torch.nn import Flatten, Sequential

from ..layers.hmlr import HMLR
from ..layers.to_manifold import ToManifold


def make_hyperbolic_backbone(
    manifold: Manifold,
    in_channels: int = 3,
    conv_channels: Sequence[int] = tuple(),
    fc_channels: Sequence[int] = tuple(),
    conv_kernel_size: int = 3,
    pool_kernel_size: int = 2,
    pool_stride: int = 2,
    image_size: Tuple[int, int] = (32, 32),
) -> Tuple[Sequential, int]:
    layers = [ToManifold(manifold=manifold)]
    all_conv_channels = (in_channels, *conv_channels)
    pool = HMaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, manifold=manifold)
    activation = HReLU(manifold=manifold)
    current_image_size = torch.tensor(image_size)
    for i in range(len(conv_channels)):
        layers.append(
            HConvolution2d(
                in_channels=all_conv_channels[i],
                out_channels=all_conv_channels[i + 1],
                kernel_size=conv_kernel_size,
                manifold=manifold,
            )
        )
        current_image_size -= conv_kernel_size - 1
        layers.append(activation)
        layers.append(pool)
        current_image_size //= pool_stride
    layers.append(Flatten())
    all_fc_channels = (all_conv_channels[-1] * current_image_size.prod(), *fc_channels)
    for i in range(len(fc_channels)):
        layers.append(
            HLinear(
                in_features=all_fc_channels[i], out_features=all_fc_channels[i + 1], manifold=manifold
            )
        )
        layers.append(activation)
    return Sequential(*layers), all_fc_channels[-1]


def make_fully_hyperbolic_net(
    out_channels: int = 1,
    *args,
    **kwargs,
) -> Sequential:
    manifold = PoincareBall(c=Curvature(requires_grad=True))
    backbone, backbone_channels = make_hyperbolic_backbone(*args, manifold=manifold, **kwargs)
    head = HMLR(in_features=backbone_channels, out_features=out_channels, manifold=manifold, bias=True)
    return Sequential(backbone, head)
