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

from typing import List

import torch.nn as nn
from opacus.layers import DPMultiheadAttention

from .errors import ShouldReplaceModuleError, UnsupportedModuleError
from .utils import register_module_fixer, register_module_validator


@register_module_validator(nn.MultiheadAttention)
def validate(module: nn.MultiheadAttention) -> List[UnsupportedModuleError]:
    return [
        ShouldReplaceModuleError(
            "We do not support nn.MultiheadAttention because its implementation uses special "
            "modules. We have written a DPMultiheadAttention class that is a drop-in replacement "
            "which is compatible with our Grad Sample hooks. Please run the recommended "
            "replacement!"
        )
    ]


@register_module_fixer(nn.MultiheadAttention)
def fix(module: nn.MultiheadAttention) -> DPMultiheadAttention:
    dp_attn = DPMultiheadAttention(
        embed_dim=module.embed_dim,
        num_heads=module.num_heads,
        dropout=module.dropout,
        bias=module.in_proj_bias is not None,
        add_bias_kv=module.bias_k is not None,
        add_zero_attn=module.add_zero_attn,
        kdim=module.kdim,
        vdim=module.vdim,
    )
    dp_attn.load_state_dict(module.state_dict())
    return dp_attn
