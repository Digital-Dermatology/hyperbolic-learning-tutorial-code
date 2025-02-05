from typing import Sequence, Tuple

import torch
from torch import nn


def make_euclidean_backbone(
    in_channels: int = 3,
    conv_channels: Sequence[int] = tuple(),
    fc_channels: Sequence[int] = tuple(),
    conv_kernel_size: int = 3,
    pool_kernel_size: int = 2,
    pool_stride: int = 2,
    image_size: Tuple[int, int] = (32, 32),
) -> Tuple[nn.Sequential, int]:
    all_conv_channels = (in_channels, *conv_channels)
    pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
    activation = nn.ReLU()
    current_image_size = torch.tensor(image_size)
    layers = []
    for i in range(len(conv_channels)):
        layers.append(
            nn.Conv2d(
                in_channels=all_conv_channels[i],
                out_channels=all_conv_channels[i + 1],
                kernel_size=conv_kernel_size,
            )
        )
        layers.append(activation)
        layers.append(pool)
        current_image_size //= pool_stride
    layers.append(nn.Flatten())
    all_fc_channels = (all_conv_channels[-1] * current_image_size.prod(), *fc_channels)
    for i in range(len(fc_channels)):
        layers.append(
            nn.Linear(
                in_features=all_fc_channels[i], out_features=all_fc_channels[i + 1]
            )
        )
        layers.append(activation)
    return nn.Sequential(*layers), all_fc_channels[-1]


def make_euclidean_convnet(
    out_channels: int = 1,
    *args,
    **kwargs,
) -> nn.Sequential:
    backbone, backbone_channels = make_euclidean_backbone(*args, **kwargs)
    return nn.Sequential(
        backbone, nn.Linear(in_features=backbone_channels, out_features=out_channels)
    )
