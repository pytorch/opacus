#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Tuple, Union

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings

from .common import GradSampleHooks_test, expander, shrinker


class Conv3d_test(GradSampleHooks_test):
    @given(
        N=st.integers(1, 4),
        C=st.sampled_from([1, 3, 32]),
        D=st.integers(3, 6),
        H=st.integers(6, 10),
        W=st.integers(6, 10),
        out_channels_mapper=st.sampled_from([expander, shrinker]),
        kernel_size=st.sampled_from([2, 3, (1, 2, 3)]),
        stride=st.sampled_from([1, 2, (1, 2, 3)]),
        padding=st.sampled_from([0, 2, (1, 2, 3)]),
        dilation=st.just(1),
        groups=st.integers(1, 16),
    )
    @settings(deadline=10000)
    def test_conv3d(
        self,
        N: int,
        C: int,
        D: int,
        H: int,
        W: int,
        out_channels_mapper: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]],
        padding: Union[int, Tuple[int]],
        dilation: int,
        groups: int,
    ):

        out_channels = out_channels_mapper(C)
        if (
            C % groups != 0 or out_channels % groups != 0
        ):  # since in_channels and out_channels must be divisible by groups
            return
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(
            in_channels=C,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.run_test(x, conv, batch_first=True, atol=10e-5, rtol=10e-3)
