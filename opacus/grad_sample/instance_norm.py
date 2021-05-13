#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import create_or_extend_grad_sample, register_grad_sampler


@register_grad_sampler(
    [
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    ]
)
def compute_instance_norm_grad_sample(
    layer: Union[
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    ],
    A: torch.Tensor,
    B: torch.Tensor,
    batch_dim: int = 0,
) -> None:
    """
    Computes per sample gradients for InstanceNorm layers

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    gs = F.instance_norm(A, eps=layer.eps) * B
    create_or_extend_grad_sample(layer.weight, torch.einsum("ni...->ni", gs), batch_dim)
    if layer.bias is not None:
        create_or_extend_grad_sample(
            layer.bias, torch.einsum("ni...->ni", B), batch_dim
        )
