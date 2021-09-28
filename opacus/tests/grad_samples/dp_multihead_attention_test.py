#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings
from opacus.layers import DPMultiheadAttention

from .common import GradSampleHooks_test


class DPMultiheadAttentionAdapter(nn.Module):
    """
    Adapter for DPMultiHeadAttention.
    This module takes three inputs, but our testing tools need that the model is given a single
    tensor, and returns a single tensor in output.

    To adapt for this, we stack the three input tensors required (q, k, v) over the LAST dimension,
    because our testing tools need to handle the `batch_first` argument which will manipulate x
    over the first (and potentially second) dimension.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.attn = DPMultiheadAttention(*args, **kwargs)

    def forward(self, x):
        q, k, v = x.unbind(-1)
        out, _attn_weights = self.attn(q, k, v)
        return out


class MultiHeadAttention_test(GradSampleHooks_test):
    @given(
        N=st.integers(1, 4),
        T=st.integers(16, 20),
        D=st.sampled_from([4]),
        P=st.sampled_from([1, 2]),
        bias=st.booleans(),
        add_bias_kv=st.booleans(),
        add_zero_attn=st.booleans(),
        kv_dim=st.booleans(),
    )
    @settings(deadline=10000)
    def test_multihead_attention(
        self,
        N: int,
        T: int,
        D: int,
        P: int,
        bias: bool,
        add_bias_kv: bool,
        add_zero_attn: bool,
        kv_dim: bool,
    ):

        if kv_dim:
            kdim, vdim = D, D
        else:
            kdim, vdim = None, None
        attn = DPMultiheadAttentionAdapter(
            D,
            P,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            dropout=0.0,
            kdim=kdim,
            vdim=vdim,
        )
        q = torch.randn([T, N, D])
        k = torch.randn([T, N, D])
        v = torch.randn([T, N, D])
        x = torch.stack((q, k, v), dim=-1)

        self.run_test(x, attn, batch_first=False)
