#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.utils.tensor_utils import sum_over_all_but_batch_and_last_n

from .utils import create_or_extend_grad_sample, register_grad_sampler


@register_grad_sampler(nn.LayerNorm)
def compute_layer_norm_grad_sample(
    layer: nn.LayerNorm,
    A: torch.Tensor,
    B: torch.Tensor,
    batch_dim: int = 0,
) -> None:
    """
    Computes per sample gradients for LayerNorm

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    create_or_extend_grad_sample(
        layer.weight,
        sum_over_all_but_batch_and_last_n(
            F.layer_norm(A, layer.normalized_shape, eps=layer.eps) * B,
            layer.weight.dim(),
        ),
        batch_dim,
    )
    create_or_extend_grad_sample(
        layer.bias,
        sum_over_all_but_batch_and_last_n(B, layer.bias.dim()),
        batch_dim,
    )
