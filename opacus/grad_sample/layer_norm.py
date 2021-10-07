#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.utils.tensor_utils import sum_over_all_but_batch_and_last_n

from .utils import register_grad_sampler


@register_grad_sampler(nn.LayerNorm)
def compute_layer_norm_grad_sample(
    layer: nn.LayerNorm,
    A: torch.Tensor,
    B: torch.Tensor,
) -> Dict[torch.Tensor, torch.Tensor]:
    """
    Computes per sample gradients for LayerNorm

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
    """
    return {
        layer.weight: sum_over_all_but_batch_and_last_n(
            F.layer_norm(A, layer.normalized_shape, eps=layer.eps) * B,
            layer.weight.dim(),
        ),
        layer.bias: sum_over_all_but_batch_and_last_n(B, layer.bias.dim()),
    }
