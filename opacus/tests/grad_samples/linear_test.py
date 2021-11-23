#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings

from .common import GradSampleHooks_test


class Linear_test(GradSampleHooks_test):
    @given(
        N=st.integers(1, 4),
        Z=st.integers(1, 4),
        H=st.integers(1, 3),
        W=st.integers(10, 17),
        input_dim=st.integers(2, 4),
        bias=st.booleans(),
        batch_first=st.booleans(),
    )
    @settings(deadline=10000)
    def test_input_bias(
        self,
        N: int,
        Z: int,
        W: int,
        H: int,
        input_dim: int,
        bias: bool,
        batch_first: bool,
    ):

        if input_dim == 2:
            if not batch_first:
                return  # see https://github.com/pytorch/opacus/pull/265
            else:
                x_shape = [N, W]
        if input_dim == 3:
            x_shape = [N, Z, W]
        if input_dim == 4:
            x_shape = [N, Z, H, W]

        linear = nn.Linear(W, W + 2, bias=bias)
        x = torch.randn(x_shape)
        if not batch_first:
            x = x.transpose(0, 1)
        self.run_test(x, linear, batch_first=batch_first)
