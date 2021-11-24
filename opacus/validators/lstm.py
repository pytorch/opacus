#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import torch.nn as nn
from opacus.layers import DPLSTM

from .errors import ShouldReplaceModuleError, UnsupportedModuleError
from .utils import register_module_fixer, register_module_validator


@register_module_validator(nn.LSTM)
def validate(module: nn.LSTM) -> List[UnsupportedModuleError]:
    return [
        ShouldReplaceModuleError(
            "We do not support nn.LSTM because its implementation uses special "
            "modules. We have written a DPLSTM class that is a drop-in replacement "
            "which is compatible with our Grad Sample hooks. Please run the recommended "
            "replacement!"
        )
    ]


@register_module_fixer(nn.LSTM)
def fix(module: nn.LSTM) -> DPLSTM:
    dplstm = DPLSTM(
        input_size=module.input_size,
        hidden_size=module.hidden_size,
        num_layers=module.num_layers,
        bias=module.bias,
        batch_first=module.batch_first,
        dropout=module.dropout,
        bidirectional=module.bidirectional,
    )
    dplstm.load_state_dict(module.state_dict())
    return dplstm
