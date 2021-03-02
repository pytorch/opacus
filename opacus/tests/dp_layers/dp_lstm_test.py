#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional, Tuple

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings
from opacus.layers import DPLSTM

from .common import DPModules_test


def lstm_train_fn(
    model: nn.Module,
    x: torch.Tensor,
    state_init: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    model.train()
    criterion = nn.MSELoss()
    logits, (hn, cn) = model(x, state_init)
    y = torch.zeros_like(logits)
    loss = criterion(logits, y)
    loss.backward()


class DPLSTM_test(DPModules_test):
    @given(
        batch_size=st.integers(1, 5),
        seq_len=st.integers(1, 6),
        emb_size=st.integers(5, 10),
        hidden_size=st.integers(3, 7),
        num_layers=st.integers(1, 3),
        bidirectional=st.booleans(),
        bias=st.booleans(),
        batch_first=st.booleans(),
        zero_init=st.booleans(),
    )
    @settings(deadline=10000)
    def test_lstm(
        self,
        batch_size: int,
        seq_len: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        bias: bool,
        batch_first: bool,
        zero_init: bool,
    ):
        lstm = nn.LSTM(
            emb_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            bias=bias,
        )
        dp_lstm = DPLSTM(
            emb_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            bias=bias,
        )

        dp_lstm.load_state_dict(lstm.state_dict())

        x = (
            torch.randn([batch_size, seq_len, emb_size])
            if batch_first
            else torch.randn([seq_len, batch_size, emb_size])
        )
        if zero_init:
            self.compare_forward_outputs(
                lstm,
                dp_lstm,
                x,
                output_names=("out", "hn", "cn"),
                atol=1e-5,
                rtol=1e-3,
            )

            self.compare_gradients(
                lstm,
                dp_lstm,
                lstm_train_fn,
                x,
                atol=1e-5,
                rtol=1e-3,
            )

        else:
            num_directions = 2 if bidirectional else 1
            h0 = torch.randn([num_layers * num_directions, batch_size, hidden_size])
            c0 = torch.randn([num_layers * num_directions, batch_size, hidden_size])
            self.compare_forward_outputs(
                lstm,
                dp_lstm,
                x,
                (h0, c0),
                output_names=("out", "hn", "cn"),
                atol=1e-5,
                rtol=1e-3,
            )
            self.compare_gradients(
                lstm,
                dp_lstm,
                lstm_train_fn,
                x,
                (h0, c0),
                atol=1e-5,
                rtol=1e-3,
            )
