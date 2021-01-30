#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional, Tuple, Callable, Union

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings

from .common import GradSampleHooks_test

expander = lambda x: 2 * x
shrinker = lambda x: x // 2


class Conv3d_test(GradSampleHooks_test):
    @given(
        N=st.sampled_from([32]),
        C=st.sampled_from([3]),
        D=st.sampled_from([4, 5, 10]),
        H=st.sampled_from([9, 10, 16]),
        W=st.sampled_from([9, 10, 16]),
        out_channels_mapper=st.sampled_from([expander, shrinker]),
        kernel_size=st.sampled_from([2, 3, (1, 2, 3)]),
        stride=st.sampled_from([1, 2, (1, 2, 3)]),
        padding=st.sampled_from([0, 2, (1, 2, 3)]),
    )
    @settings(deadline=10000)
    def test_images(
        self,
        N: int,
        C: int,
        D: int,
        H: int,
        W: int,
        out_channels_mapper: Callable[[int], int],
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]],
        padding: Union[int, Tuple[int]],
    ):

        out_channels = out_channels_mapper(C)
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(
            in_channels=C,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.run_test(x, conv, batch_first=True, atol=10e-6, rtol=10e-3)

    @given(
        N=st.sampled_from([4]),
        C=st.sampled_from([16, 32]),
        D=st.sampled_from([12]),
        H=st.sampled_from([12]),
        W=st.sampled_from([12]),
        out_channels_mapper=st.sampled_from([expander, shrinker]),
        kernel_size=st.sampled_from([3]),
        stride=st.sampled_from([1]),
        groups=st.sampled_from([2, 4, 16]),
    )
    @settings(deadline=10000)
    def test_4d_inputs(
        self,
        N: int,
        C: int,
        D: int,
        H: int,
        W: int,
        out_channels_mapper: Callable[[int], int],
        kernel_size: int,
        stride: int,
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
            groups=groups,
        )
        self.run_test(x, conv, batch_first=True, atol=10e-5, rtol=10e-3)
