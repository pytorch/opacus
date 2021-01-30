#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
from opacus.layers import DPLSTM

from typing import Optional, Tuple, Callable, Union

import hypothesis.strategies as st
from hypothesis import given, settings

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
    @given(
        N=st.sampled_from([32, 1]),
        T=st.sampled_from([20]),
        D=st.sampled_from([8]),
        H=st.sampled_from([16]),
        num_layers=st.sampled_from([1, 2]),
        bias=st.booleans(),
        batch_first=st.booleans(),
    )
    @settings(deadline=10000)
    def test_batch_bias(
        self,
        N: int,
        T: int,
        D: int,
        H: int,
        num_layers: int,
        bias: bool,
        batch_first: bool,
    ):

        lstm = DPSLTMAdapter(
            D, H, num_layers=num_layers, batch_first=batch_first, bias=bias
        )
        if batch_first:
            x = torch.randn([N, T, D])
        else:
            x = torch.randn([T, N, D])
        self.run_test(x, lstm, batch_first=batch_first)
