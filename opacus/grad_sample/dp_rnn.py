#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Dict, List

import torch
import torch.nn as nn
from opacus.layers.dp_rnn import RNNLinear
from opt_einsum import contract

from .utils import register_grad_sampler


@register_grad_sampler(RNNLinear)
def compute_rnn_linear_grad_sample(
    layer: RNNLinear, activations: List[torch.Tensor], backprops: torch.Tensor
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
    activations = activations[0]
    ret = {}
    if layer.weight.requires_grad:
        gs = contract("n...i,n...j->nij", backprops, activations)
        ret[layer.weight] = gs
    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = contract("n...k->nk", backprops)
    return ret
