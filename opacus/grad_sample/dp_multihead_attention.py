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
from opacus.layers.dp_multihead_attention import SequenceBias

from .utils import register_grad_sampler


@register_grad_sampler(SequenceBias)
def compute_sequence_bias_grad_sample(
    layer: SequenceBias, activations: List[torch.Tensor], backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``SequenceBias`` layer

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    ret = {}
    if layer.bias.requires_grad:
        ret[layer.bias] = backprops[:, -1]
    return ret
