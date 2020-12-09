#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
from opacus.layers import DPLSTM

from .common import GradSampleHooks_test


class DPSLTMAdapter(nn.Module):
    """
    Adapter for DPLSTM.
    LSTM returns a tuple, but our testing tools need the model to return a single tensor in output.
    We do this adaption here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dplstm = DPLSTM(*args, **kwargs)

    def forward(self, x):
        out, _rest = self.dplstm(x)
        return out


class LSTM_test(GradSampleHooks_test):
    def test_batch_first_bias(self):
        N, T, D, H = 32, 20, 8, 16
        lstm = DPSLTMAdapter(D, H, num_layers=1, batch_first=True, bias=True)
        x = torch.randn([N, T, D])
        self.run_test(x, lstm, batch_first=True)

    def test_batch_second_bias(self):
        N, T, D, H = 32, 20, 8, 16
        lstm = DPSLTMAdapter(D, H, num_layers=1, batch_first=False, bias=True)
        x = torch.randn([T, N, D])
        self.run_test(x, lstm, batch_first=False)

    def test_batch_first_nobias(self):
        N, T, D, H = 32, 20, 8, 16
        lstm = DPSLTMAdapter(D, H, num_layers=1, batch_first=True, bias=False)
        x = torch.randn([N, T, D])
        self.run_test(x, lstm, batch_first=True)

    def test_batch_second_nobias(self):
        N, T, D, H = 32, 20, 8, 16
        lstm = DPSLTMAdapter(D, H, num_layers=1, batch_first=False, bias=False)
        x = torch.randn([T, N, D])
        self.run_test(x, lstm, batch_first=False)

    def test_batch_first_bias_two_layers(self):
        N, T, D, H = 32, 20, 8, 16
        lstm = DPSLTMAdapter(D, H, num_layers=2, batch_first=True, bias=True)
        x = torch.randn([N, T, D])
        self.run_test(x, lstm, batch_first=True)

    def test_batch_second_bias_two_layers(self):
        N, T, D, H = 32, 20, 8, 16
        lstm = DPSLTMAdapter(D, H, num_layers=2, batch_first=False, bias=True)
        x = torch.randn([T, N, D])
        self.run_test(x, lstm, batch_first=False)

    def test_batch_first_nobias_two_layers(self):
        N, T, D, H = 32, 20, 8, 16
        lstm = DPSLTMAdapter(D, H, num_layers=2, batch_first=True, bias=False)
        x = torch.randn([N, T, D])
        self.run_test(x, lstm, batch_first=True)

    def test_batch_second_nobias_two_layers(self):
        N, T, D, H = 32, 20, 8, 16
        lstm = DPSLTMAdapter(D, H, num_layers=2, batch_first=False, bias=False)
        x = torch.randn([T, N, D])
        self.run_test(x, lstm, batch_first=False)
