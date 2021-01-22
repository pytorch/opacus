#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
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
    def test_batch_second_no_extras_one_head(self):
        N, T, D, P = 32, 20, 8, 1
        attn = DPMultiheadAttentionAdapter(
            D,
            P,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            dropout=0.0,
        )
        q = torch.randn([T, N, D])
        k = torch.randn([T, N, D])
        v = torch.randn([T, N, D])

        x = torch.stack((q, k, v), dim=-1)

        self.run_test(x, attn, batch_first=False)

    def test_batch_second_no_extras_two_heads(self):
        N, T, D, P = 32, 20, 8, 2
        attn = DPMultiheadAttentionAdapter(
            D,
            P,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            dropout=0.0,
        )
        q = torch.randn([T, N, D])
        k = torch.randn([T, N, D])
        v = torch.randn([T, N, D])
        x = torch.stack((q, k, v), dim=-1)

        self.run_test(x, attn, batch_first=False)

    def test_batch_second_just_bias(self):
        N, T, D, P = 32, 20, 8, 1
        attn = DPMultiheadAttentionAdapter(
            D,
            P,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            dropout=0.0,
        )
        q = torch.randn([T, N, D])
        k = torch.randn([T, N, D])
        v = torch.randn([T, N, D])
        x = torch.stack((q, k, v), dim=-1)

        self.run_test(x, attn, batch_first=False)

    def test_batch_second_just_bias_kv(self):
        N, T, D, P = 32, 20, 8, 1
        attn = DPMultiheadAttentionAdapter(
            D,
            P,
            bias=False,
            add_bias_kv=True,
            add_zero_attn=False,
            dropout=0.0,
        )
        q = torch.randn([T, N, D])
        k = torch.randn([T, N, D])
        v = torch.randn([T, N, D])
        x = torch.stack((q, k, v), dim=-1)

        self.run_test(x, attn, batch_first=False)

    def test_batch_second_just_zero_attn(self):
        N, T, D, P = 32, 20, 8, 1
        attn = DPMultiheadAttentionAdapter(
            D,
            P,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=True,
            dropout=0.0,
        )
        q = torch.randn([T, N, D])
        k = torch.randn([T, N, D])
        v = torch.randn([T, N, D])
        x = torch.stack((q, k, v), dim=-1)

        self.run_test(x, attn, batch_first=False)

    def test_batch_second_Just_kdim_vdim(self):
        N, T, D, P = 32, 20, 8, 1
        attn = DPMultiheadAttentionAdapter(
            D,
            P,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            dropout=0.0,
            kdim=D,
            vdim=D,
        )
        q = torch.randn([T, N, D])
        k = torch.randn([T, N, D])
        v = torch.randn([T, N, D])
        x = torch.stack((q, k, v), dim=-1)

        self.run_test(x, attn, batch_first=False)

    def test_batch_second_all_options(self):
        N, T, D, P = 32, 20, 8, 1
        attn = DPMultiheadAttentionAdapter(
            D,
            P,
            bias=True,
            add_bias_kv=True,
            add_zero_attn=True,
            dropout=0.0,  # We can't repro dropout so we don't test it at the moment
            kdim=D,
            vdim=D,
        )
        q = torch.randn([T, N, D])
        k = torch.randn([T, N, D])
        v = torch.randn([T, N, D])
        x = torch.stack((q, k, v), dim=-1)

        self.run_test(x, attn, batch_first=False)
