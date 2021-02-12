#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional, Tuple, Callable

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings

from .common import GradSampleHooks_test

expander = lambda x: 2 * x
shrinker = lambda x: x // 2


class Conv2d_test(GradSampleHooks_test):
    @given(
        N=st.integers(32, 48),
        C=st.integers(3, 32),
        H=st.integers(9, 24),
        W=st.integers(9, 24),
        out_channels_mapper=st.sampled_from([expander, shrinker]),
        kernel_size=st.integers(2, 3),
        stride=st.integers(1, 2),
        padding=st.sampled_from([0, 2]),
        groups=st.integers(1, 16),
    )
    @settings(deadline=10000)
    def test_conv2d(
        self,
        N: int,
        C: int,
        H: int,
        W: int,
        out_channels_mapper: Callable[[int], int],
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
    ):

        out_channels = out_channels_mapper(C)
        if (
            C % groups != 0 or out_channels % groups != 0
        ):  # since in_channels and out_channels must be divisible by groups
            return

        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(
            in_channels=C,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.run_test(x, conv, batch_first=True)
