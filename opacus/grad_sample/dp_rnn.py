#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Dict

import torch
import torch.nn as nn
from opacus.layers.dp_rnn import RNNLinear

from .utils import register_grad_sampler


@register_grad_sampler(RNNLinear)
def compute_rnn_linear_grad_sample(
    layer: RNNLinear, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``RNNLinear`` layer. The RNN-like (DPLSTM, DPGRU) models
    are written using this layer as its building block.

    class

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """

    gs = torch.einsum("n...i,n...j->nij", backprops, activations)

    ret = {layer.weight: gs}
    if layer.bias is not None:
        ret[layer.bias] = torch.einsum("n...k->nk", backprops)

    return ret
