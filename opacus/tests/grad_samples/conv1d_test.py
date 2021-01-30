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


class Conv1d_test(GradSampleHooks_test):
    @given(
        N=st.sampled_from([32]),
        C=st.sampled_from([3, 4, 12, 24]),
        W=st.sampled_from([11, 10]),
        out_channels_mapper=st.sampled_from([expander, shrinker]),
        kernel_size=st.sampled_from([2, 3]),
        stride=st.sampled_from([1, 2]),
        padding=st.sampled_from([0, 2]),
        groups=st.sampled_from([1, 2, 3, 12]),
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
            groups=groups,
        )
        self.run_test(x, conv, batch_first=True)
