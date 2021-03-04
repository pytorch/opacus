#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Callable

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings

from .common import GradSampleHooks_test, expander, shrinker


class Conv1d_test(GradSampleHooks_test):
    @given(
        N=st.integers(1, 4),
        C=st.sampled_from([1, 3, 32]),
        W=st.integers(6, 10),
        out_channels_mapper=st.sampled_from([expander, shrinker]),
        kernel_size=st.integers(2, 3),
        stride=st.integers(1, 2),
        padding=st.integers(0, 2),
        dilation=st.integers(1, 2),
        groups=st.integers(1, 12),
    )
    @settings(deadline=10000)
    def test_conv1d(
        self,
        N: int,
        C: int,
        W: int,
        out_channels_mapper: Callable[[int], int],
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
    ):

        out_channels = out_channels_mapper(C)
        if (
            C % groups != 0 or out_channels % groups != 0
        ):  # since in_channels and out_channels must be divisible by groups
            return

        x = torch.randn([N, C, W])
        conv = nn.Conv1d(
            in_channels=C,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.run_test(x, conv, batch_first=True, atol=10e-5, rtol=10e-4)
