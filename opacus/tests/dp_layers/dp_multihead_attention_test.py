#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional

import hypothesis.strategies as st
import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings
from opacus.layers import DPMultiheadAttention

from .common import DPModules_test


def attn_train_fn(
    model: nn.Module,
    *args,
    **kwargs,
):
    model.train()
    criterion = nn.MSELoss()
    logits, attn_weights = model(*args, **kwargs)
    y = torch.zeros_like(logits)
    loss = criterion(logits, y)
    loss.backward()


class DPMultiheadAttention_test(DPModules_test):
    @given(
        batch_size=st.integers(1, 5),
        src_seq_len=st.integers(1, 6),
        tgt_seq_len=st.integers(1, 6),
        num_heads=st.integers(1, 3),
        bias=st.booleans(),
        add_bias_kv=st.booleans(),
        add_zero_attn=st.booleans(),
        kdim=st.integers(2, 8) | st.none(),
        vdim=st.integers(2, 8) | st.none(),
    )
    @settings(deadline=10000)
    @pytest.mark.skip(
        "Failing due to a known problem. Should be enabled after issue #123 is fixed"
    )
    def test_attn(
        self,
        batch_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        num_heads: int,
        bias: bool,
        add_bias_kv: bool,
        add_zero_attn: bool,
        kdim: Optional[int],
        vdim: Optional[int],
    ):
        embed_dim = 4 * num_heads  # embed_dim must be divisible by num_heads

        attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.0,  # Untestable between two different implementations
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
        )
        dp_attn = DPMultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.0,  # Untestable between two different implementations
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
        )

        dp_attn.load_state_dict(attn.state_dict())

        q = torch.randn(tgt_seq_len, batch_size, embed_dim)
        k = torch.randn(
            src_seq_len, batch_size, kdim if kdim is not None else embed_dim
        )
        v = torch.randn(
            src_seq_len, batch_size, vdim if vdim is not None else embed_dim
        )

        self.compare_forward_outputs(
            attn,
            dp_attn,
            q,
            k,
            v,
            output_names=("attn_out", "attn_out_weights"),
            atol=1e-5,
            rtol=1e-3,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
        )

        self.compare_gradients(
            attn,
            dp_attn,
            attn_train_fn,
            q,
            k,
            v,
            atol=1e-5,
            rtol=1e-3,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
        )
