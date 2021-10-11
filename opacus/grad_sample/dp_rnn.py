#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch
from opacus.layers.dp_rnn import RNNLinear

from .utils import create_or_accumulate_grad_sample, register_grad_sampler


@register_grad_sampler(RNNLinear)
def compute_rnn_linear_grad_sample(
    layer: RNNLinear, A: torch.Tensor, B: torch.Tensor
) -> None:
    """
    Computes per sample gradients for ``RNNLinear`` layer. The RNN-like (DPLSTM, DPGRU) models
    are written using this layer as its building block.

    class

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
    """

    gs = torch.einsum("n...i,n...j->nij", B, A)
    create_or_accumulate_grad_sample(layer.weight, gs, layer)

    if layer.bias is not None:
        create_or_accumulate_grad_sample(
            layer.bias,
            torch.einsum("n...k->nk", B),
            layer,
        )
