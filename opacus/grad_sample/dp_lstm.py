#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch
from opacus.layers.dp_lstm import LSTMLinear

from .utils import create_or_accumulate_grad_sample, register_grad_sampler


@register_grad_sampler(LSTMLinear)
def compute_lstm_linear_grad_sample(
    layer: LSTMLinear, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
) -> None:
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

    gs = torch.einsum("n...i,n...j->n...ij", B, A)
    create_or_accumulate_grad_sample(
        layer.weight, torch.einsum("n...ij->nij", gs), layer
    )

    if layer.bias is not None:
        create_or_accumulate_grad_sample(
            layer.bias,
            torch.einsum("n...k->nk", B),
            layer,
        )
