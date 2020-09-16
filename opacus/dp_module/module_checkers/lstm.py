#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn
from opacus.dp_module.modules import DPLSTM

from .errors import ShouldReplaceModuleError
from .module_checker import ModuleChecker


class LSTMChecker(ModuleChecker):
    def __init__(self):
        super().__init__([nn.LSTM])

    def is_valid(self, module: nn.LSTM) -> bool:
        return False

    def validate(self, module: nn.LSTM) -> None:
        if self.is_watching(module) and not self.is_valid(module):
            raise ShouldReplaceModuleError(
                "We do not support nn.LSTM because its implementation uses special "
                "modules. We have written a DPLSTM class that is a drop-in replacement "
                "which is compatible with our Grad Sample hooks. Please run the recommended "
                "replacement!"
            )

    def recommended_replacement(self, module: nn.LSTM) -> DPLSTM:
        return DPLSTM(
            input_size=module.input_size,
            hidden_size=module.hidden_size,
            num_layers=module.num_layers,
            bias=module.bias,
            batch_first=module.batch_first,
            dropout=module.dropout,
            bidirectional=module.bidirectional,
        )
