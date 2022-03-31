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


from typing import Dict

import torch
import torch.nn as nn
from opacus.layers.dp_multihead_attention import InputProjection, SequenceBias

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


@register_grad_sampler(InputProjection)
def compute_input_projection_grad_sample(
    layer: InputProjection, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``InputProjection`` layer

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    ret = {}

    # TODO: since these calculations are taken from the linear module, perhaps they should
    # be imported from there and reused between both methods?
    def linear_weight_grad(activations: torch.Tensor, backprops: torch.Tensor):
        return torch.einsum("n...i,n...j->nij", backprops, activations)

    def linear_bias_grad(backprops: torch.Tensor):
        return torch.einsum("n...k->nk", backprops)

    q_bp, k_bp, v_bp = backprops.unbind(-1)

    q_end_index = layer.qlinear_weight.shape[1]
    k_end_index = q_end_index + layer.klinear_weight.shape[1]
    q_a = activations[:, :, :q_end_index]
    k_a = activations[:, :, q_end_index:k_end_index]
    v_a = activations[:, :, k_end_index:]

    ret[layer.qlinear_weight] = linear_weight_grad(q_a, q_bp)
    ret[layer.klinear_weight] = linear_weight_grad(k_a, k_bp)
    ret[layer.vlinear_weight] = linear_weight_grad(v_a, v_bp)

    if layer.bias is not None:
        ret[layer.bias] = torch.cat(
            (linear_bias_grad(q_bp), linear_bias_grad(k_bp), linear_bias_grad(v_bp)),
            axis=-1,
        )

    return ret
