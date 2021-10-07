#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import register_grad_sampler


@register_grad_sampler(nn.GroupNorm)
def compute_group_norm_grad_sample(
    layer: nn.GroupNorm,
    A: torch.Tensor,
    B: torch.Tensor,
) -> Dict[torch.Tensor, torch.Tensor]:
    """
    Computes per sample gradients for GroupNorm

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
    """
    gs = F.group_norm(A, layer.num_groups, eps=layer.eps) * B
    ret = {layer.weight: torch.einsum("ni...->ni", gs)}
    if layer.bias is not None:
        ret[layer.bias] = torch.einsum("ni...->ni", B)
    return ret
