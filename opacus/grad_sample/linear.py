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

import logging
from typing import Dict, List

import torch
import torch.nn as nn

from .utils import register_grad_sampler, register_norm_sampler


logger = logging.getLogger(__name__)
logging.disabled = False


@register_grad_sampler(nn.Linear)
def compute_linear_grad_sample(
    layer: nn.Linear, activations: List[torch.Tensor], backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``nn.Linear`` layer

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    activations = activations[0]
    ret = {}
    if layer.weight.requires_grad:
        gs = torch.einsum("n...i,n...j->nij", backprops, activations)
        ret[layer.weight] = gs
    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = torch.einsum("n...k->nk", backprops)
    return ret


@register_norm_sampler(nn.Linear)
def compute_linear_norm_sample(
    layer: nn.Linear, activations: List[torch.Tensor], backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradient norms for ``nn.Linear`` layer

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    activations = activations[0]
    ret = {}

    if backprops.dim() == 2:
        if layer.weight.requires_grad:
            g = torch.einsum("n...i,n...i->n", backprops, backprops)
            a = torch.einsum("n...j,n...j->n", activations, activations)
            ret[layer.weight] = torch.sqrt((g * a).flatten())
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.sqrt(
                torch.einsum("n...i,n...i->n", backprops, backprops).flatten()
            )
    elif backprops.dim() == 3:
        if layer.weight.requires_grad:

            ggT = torch.einsum("nik,njk->nij", backprops, backprops)  # batchwise g g^T
            aaT = torch.einsum(
                "nik,njk->nij", activations, activations
            )  # batchwise a a^T
            ga = torch.einsum("n...i,n...i->n", ggT, aaT).clamp(min=0)

            ret[layer.weight] = torch.sqrt(ga)
        if layer.bias is not None and layer.bias.requires_grad:
            ggT = torch.einsum("nik,njk->nij", backprops, backprops)
            gg = torch.einsum("n...i,n...i->n", ggT, ggT).clamp(min=0)
            ret[layer.bias] = torch.sqrt(gg)
    return ret
