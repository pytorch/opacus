#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict

import torch
import torch.nn as nn

from .utils import register_grad_sampler


@register_grad_sampler(nn.Linear)
def compute_linear_grad_sample(
    layer: nn.Linear, A: torch.Tensor, B: torch.Tensor
) -> Dict[torch.Tensor, torch.Tensor]:
    """
    Computes per sample gradients for ``nn.Linear`` layer

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
    """
    gs = torch.einsum("n...i,n...j->nij", B, A)
    ret = {layer.weight: gs}
    if layer.bias is not None:
        ret[layer.bias] = torch.einsum("n...k->nk", B)

    return ret
