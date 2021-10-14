#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Dict

import torch
from opacus.layers.dp_multihead_attention import SequenceBias

from .utils import register_grad_sampler


@register_grad_sampler(SequenceBias)
def compute_sequence_bias_grad_sample(
    layer: SequenceBias, A: torch.Tensor, B: torch.Tensor
) -> Dict[torch.Tensor, torch.Tensor]:
    """
    Computes per sample gradients for ``SequenceBias`` layer

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
    """
    return {layer.bias: B[:, -1]}
