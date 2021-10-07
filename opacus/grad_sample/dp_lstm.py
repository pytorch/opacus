#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Dict

import torch
from opacus.layers.dp_lstm import LSTMLinear

from .utils import register_grad_sampler


@register_grad_sampler(LSTMLinear)
def compute_lstm_linear_grad_sample(
    layer: LSTMLinear,
    A: torch.Tensor,
    B: torch.Tensor,
) -> Dict[torch.Tensor, torch.Tensor]:
    """
    Computes per sample gradients for ``LSTMLinear`` layer. The DPLSTM class is written using
    this layer as its building block.

    class

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """

    gs = torch.einsum("n...i,n...j->nij", B, A)

    ret = {layer.weight: gs}
    if layer.bias is not None:
        ret[layer.bias] = torch.einsum("n...k->nk", B)

    return ret
