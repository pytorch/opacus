#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Dict

import torch
import torch.nn as nn
from opacus.layers.dp_multihead_attention import SequenceBias

from .utils import register_grad_sampler


@register_grad_sampler(SequenceBias)
def compute_sequence_bias_grad_sample(
    layer: SequenceBias, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``SequenceBias`` layer

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    return {layer.bias: backprops[:, -1]}
