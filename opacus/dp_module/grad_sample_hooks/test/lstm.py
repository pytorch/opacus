#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
from opacus.dp_module.modules import DPLSTM

from .common import GradSampleHooks_test, ModelWithLoss


class DPSLTMWrapper(nn.Module):
    """
    Convenience function to use with Sequential.
    Takes the item at [pos]. Useful with eg nn.LSTM whose forward returns a tuple.
    This lets you take only one item in that tuple.

    Args:
        pos (int): The position to grab.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dplstm = DPLSTM(*args, **kwargs)

    def forward(self, x):
        out, _rest = self.dplstm(x)
        return out


class LSTM_test(GradSampleHooks_test):
    def test_batch_first(self):
        N, T, D, H = 32, 20, 8, 16
        lstm = DPSLTMWrapper(D, H, num_layers=1, batch_first=True)
        module = ModelWithLoss(lstm, n_classes=H + 2)
        x = torch.randn([N, T, D])
        self.run_test(x, module, batch_first=True)

    def test_batch_second(self):
        N, T, D, H = 32, 20, 8, 16
        lstm = DPSLTMWrapper(D, H, num_layers=1, batch_first=False)
        module = ModelWithLoss(lstm, n_classes=H + 2)
        x = torch.randn([T, N, D])
        self.run_test(x, module, batch_first=False)
