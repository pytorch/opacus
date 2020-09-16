#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn
from opacus.dp_module.modules import DPMultiheadAttention

from .errors import ShouldReplaceModuleError
from .module_checker import ModuleChecker


class MultiheadAttentionChecker(ModuleChecker):
    def __init__(self):
        super().__init__([nn.MultiheadAttention])

    def is_valid(self, module: nn.MultiheadAttention) -> bool:
        return False

    def validate(self, module: nn.MultiheadAttention) -> None:
        if self.is_watching(module) and not self.is_valid(module):
            raise ShouldReplaceModuleError(
                "We do not support nn.MultiheadAttention because its implementation uses special "
                "modules. We have written a DPMultiheadAttention class that is a drop-in replacement "
                "which is compatible with our Grad Sample hooks. Please run the recommended "
                "replacement!"
            )

    def recommended_replacement(
        self, module: nn.MultiheadAttention
    ) -> DPMultiheadAttention:
        return DPMultiheadAttention(
            embed_dim=module.embed_dim,
            num_heads=module.num_heads,
            dropout=module.dropout,
            bias=module.in_proj_bias is not None,
            add_bias_kv=module.bias_k is not None,
            add_zero_attn=module.add_zero_attn,
            kdim=module.kdim,
            vdim=module.vdim,
        )
