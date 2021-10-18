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
    activations: torch.Tensor,
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for GroupNorm

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    gs = F.group_norm(activations, layer.num_groups, eps=layer.eps) * backprops
    ret = {layer.weight: torch.einsum("ni...->ni", gs)}
    if layer.bias is not None:
        ret[layer.bias] = torch.einsum("ni...->ni", backprops)
    return ret
